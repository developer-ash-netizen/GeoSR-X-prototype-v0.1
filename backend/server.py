"""GeoSR-X prototype server. Standard library only (+ numpy/scipy/scikit-image/Pillow/tifffile).

    python run.py            ->  http://127.0.0.1:8000

API
  GET  /api/status                       readiness + calibration proof (reliability data)
  POST /api/scene    {seed,n_dates,cloud_level,noise,change}   synthetic scene -> {sid, meta}
  POST /api/upload?sid=&name=  (raw body) add one GeoTIFF acquisition   |  POST /api/upload/build {sid}
  POST /api/run      {sid, gate}         start pipeline job -> {job}
  GET  /api/job/<id>                     stage-by-stage progress
  GET  /api/session/<sid>/summary        all metrics
  GET  /api/session/<sid>/layer/<name>   PNG   (sr,bicubic,fid,prior,gt,sr_fcc,provenance,risk,uncertainty,alpha,E1..E4,err,regions,truth_change,lr?i=)
  GET  /api/session/<sid>/region?x=&y=   region inspector
  GET  /api/session/<sid>/regionmask?id= PNG outline of one region
  GET  /api/session/<sid>/download/<geotiff|sidecar|coverage>
"""
import json, os, re, sys, threading, time, uuid, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np

from . import pipeline as P, validation as V, export as E, render as RN
from .synth import make_scene, CLASS_NAMES
from .sensor import SensorModel
from .train import load_or_train, ART
from .l4_l6 import PROV_NAMES
from .l7_calibration import CLASSES
from .pipeline_io import scene_from_geotiffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "frontend")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

STATE = dict(prior=None, calib=None, report=None, ready=False, msg="starting")
SESS = {}
JOBS = {}
LOCK = threading.Lock()


def jdump(o):
    return json.dumps(E._jsonable(o)).encode()


def boot():
    def log(m):
        STATE["msg"] = m; print(m, flush=True)
    try:
        STATE["prior"], STATE["calib"], STATE["report"] = load_or_train(log)
        STATE["ready"] = True; STATE["msg"] = "ready"
    except Exception as e:
        traceback.print_exc(); STATE["msg"] = f"boot failed: {e}"


def scene_meta(sc):
    return dict(name=sc.name, seed=sc.seed, n_dates=len(sc.dates), dates=sc.dates,
                cloud_fraction=[float(x) for x in sc.cloud.mean((1, 2))], size_hr=[sc.lr.shape[2] * sc.scale, sc.lr.shape[3] * sc.scale],
                scale=sc.scale, gsd_hr=sc.gsd_lr / sc.scale, has_truth=sc.gt is not None,
                has_change_truth=bool(sc.change_mask is not None and sc.change_mask.any()), epsg=sc.epsg)


def run_job(jid, sid, gate):
    job = JOBS[jid]; sess = SESS[sid]; sc = sess["scene"]
    stages = {sid_: dict(id=sid_, name=n, desc=d, status="wait", time=None, info={}) for sid_, n, d in P.STAGES}
    job["stages"] = stages

    def cb(stage, status, info, dt):
        st = stages[stage]; st["status"] = status; st["info"] = E._jsonable(info)
        if dt is not None:
            st["time"] = round(dt, 2)
    try:
        t0 = time.time()
        R = P.process(sc, STATE["prior"], STATE["calib"], cb=cb, gate=gate)
        cb("L8", "run", {}, None); t1 = time.time()
        d = os.path.join(OUT, sid); os.makedirs(d, exist_ok=True)
        tif = os.path.join(d, f"geosrx_{sc.name}.tif")
        E.write_geotiff(tif, sc, R)
        cb("L8", "done", dict(file=os.path.basename(tif)), t1)
        cb("L9", "run", {}, None); t2 = time.time()
        Vres = V.run_validation(sc, R, STATE["prior"])
        cb("L9", "done", {}, t2)
        side = E.sidecar(sc, R, Vres, STATE["report"], dict(gate=gate))
        json.dump(side, open(os.path.join(d, f"geosrx_{sc.name}_sidecar.json"), "w"), indent=1)
        json.dump(STATE["report"], open(os.path.join(d, "coverage_report.json"), "w"), indent=1)
        sess.update(R=R, V=Vres, gate=gate, cache={}, dir=d, tif=tif)
        job.update(status="done", total=round(time.time() - t0, 1))
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}")


def build_summary(sess):
    sc, R, Vres = sess["scene"], sess["R"], sess["V"]
    s = sc.scale
    prov_area = {PROV_NAMES[k]: float((R.prov[R.lab] == k).mean()) for k in range(3)}
    bound_by_prov = {PROV_NAMES[k]: (float(np.nanmean(R.bound[R.prov == k])) if (R.prov == k).any() else None) for k in range(3)}
    cls_area = {CLASSES[c]: float((R.cls[R.lab] == c).mean()) for c in range(4)}
    layers = ["sr", "bicubic", "fid", "prior", "sr_fcc", "provenance", "risk", "uncertainty", "alpha", "E1", "E2", "E3", "E4", "regions"]
    if sc.gt is not None:
        layers += ["gt", "err"]
    if sc.change_mask is not None and sc.change_mask.any():
        layers += ["truth_change"]
    return dict(
        meta=scene_meta(sc), gate=sess["gate"], target=int(R.ref), clean=R.clean, ncc=R.ncc, usable=R.usable,
        shifts=R.shifts, reg_err=R.reg_err, gain=R.gain, sigma=R.ev["sigma"],
        E=dict(E1_mean=float(R.ev["E1"].mean()), E3_frac=float(R.ev["E3"].mean()), E4_mean=float(R.ev["E4"].mean()),
               E2_med_raw=float(np.median(R.ev["E2"]))),
        n_regions=int(R.n), alpha_mean=float(R.alpha_hr.mean()),
        provenance=dict(area=prov_area, mean_bound=bound_by_prov, true_mae=Vres.get("prov_error")),
        classes=cls_area, bound_mean=float(np.nanmean(R.bound)), fid_rms=R.fid_resid,
        fusion_residual=R.hist, validation=Vres, calibration=STATE["report"], layers=layers,
        levels=[0.9])


def make_handler():
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def _json(self, o, code=200):
            self._send(code, jdump(o))

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        # ------------------------------------------------------------- GET
        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query); p = u.path
            try:
                if p == "/api/status":
                    return self._json(dict(ready=STATE["ready"], msg=STATE["msg"], calibration=STATE["report"]))
                m = re.match(r"^/api/job/([\w-]+)$", p)
                if m:
                    j = JOBS.get(m.group(1))
                    if not j:
                        return self._json(dict(error="no such job"), 404)
                    return self._json(dict(status=j["status"], stages=list(j.get("stages", {}).values()),
                                           error=j.get("error"), total=j.get("total")))
                m = re.match(r"^/api/session/([\w-]+)/(\w+)(?:/(\w+))?$", p)
                if m:
                    sid, what, arg = m.groups()
                    sess = SESS.get(sid)
                    if not sess or "R" not in sess:
                        return self._json(dict(error="session not processed"), 404)
                    return self._session_get(sess, sid, what, arg, q)
                # static
                rel = "index.html" if p in ("/", "") else p.lstrip("/")
                fp = os.path.normpath(os.path.join(WEB, rel))
                if not fp.startswith(WEB) or not os.path.isfile(fp):
                    return self._send(404, b"not found", "text/plain")
                ct = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8",
                      "svg": "image/svg+xml", "png": "image/png"}.get(fp.rsplit(".", 1)[-1], "application/octet-stream")
                return self._send(200, open(fp, "rb").read(), ct)
            except Exception as e:
                traceback.print_exc(); self._json(dict(error=str(e)), 500)

        def _session_get(self, sess, sid, what, arg, q):
            sc, R = sess["scene"], sess["R"]
            if what == "summary":
                return self._json(build_summary(sess))
            if what == "layer":
                key = (arg, q.get("i", [""])[0])
                if key not in sess["cache"]:
                    sess["cache"][key] = RN.render_layer(arg, sc, R, q.get("i", [0])[0])
                return self._send(200, sess["cache"][key], "image/png", {"Cache-Control": "max-age=60"})
            if what == "regionmask":
                rid = int(q.get("id", [0])[0])
                from skimage.segmentation import find_boundaries
                mk = R.lab == rid
                b = find_boundaries(mk, mode="thick") & np.pad(mk, 1)[1:-1, 1:-1] | find_boundaries(mk, mode="inner")
                a = np.zeros(mk.shape + (4,), np.uint8); a[b] = (255, 240, 60, 255); a[mk & ~b] = (255, 240, 60, 45)
                return self._send(200, RN._png(a), "image/png")
            if what == "region":
                x = int(np.clip(int(float(q.get("x", [0])[0])), 0, R.lab.shape[1] - 1)); y = int(np.clip(int(float(q.get("y", [0])[0])), 0, R.lab.shape[0] - 1))
                i = int(R.lab[y, x]); rf = R.rf
                d = dict(id=i, x=x, y=y, area_px=int(rf["size"][i]), provenance=PROV_NAMES[int(R.prov[i])], prov_id=int(R.prov[i]),
                         land_cover=CLASSES[int(R.cls[i])],
                         evidence=dict(E1=float(rf["E1"][i]), E1_max=len(R.clean), E2=float(rf["E2"][i]), E3=float(rf["E3f"][i]), E4=float(rf["E4"][i]),
                                       diversity=float(rf["div"][i])),
                         gate_alpha=float(rf["alpha"][i]), prior_share=float(rf["fp"][i]),
                         risk=dict(total=float(rf["risk"][i]), temporal=float(rf["r_t"][i]), spectral=float(rf["r_s"][i]), texture=float(rf["r_x"][i])),
                         self_consistency=dict(residual_sigma=float(R.res_r[i]), inflation=float(R.infl[i])),
                         uncertainty=dict(predicted=float(R.u0[i]), p90_bound=float(R.bound[i])),
                         spectra=[float(R.sr[b][R.lab == i].mean()) for b in range(4)])
                if sc.gt is not None:
                    d["true_mae"] = float(R.mae[i]); d["covered"] = bool(R.mae[i] <= R.bound[i])
                return self._json(d)
            if what == "download":
                fn = {"geotiff": (sess["tif"], "image/tiff"),
                      "sidecar": (os.path.join(sess["dir"], f"geosrx_{sc.name}_sidecar.json"), "application/json"),
                      "coverage": (os.path.join(sess["dir"], "coverage_report.json"), "application/json")}.get(arg)
                if not fn:
                    return self._json(dict(error="unknown download"), 404)
                return self._send(200, open(fn[0], "rb").read(), fn[1],
                                  {"Content-Disposition": f'attachment; filename="{os.path.basename(fn[0])}"'})
            return self._json(dict(error="unknown"), 404)

        # ------------------------------------------------------------- POST
        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query); p = u.path
            try:
                if p == "/api/scene":
                    if not STATE["ready"]:
                        return self._json(dict(error="server still calibrating"), 503)
                    b = json.loads(self._body() or b"{}")
                    sc = make_scene(seed=int(b.get("seed", 7)), n_dates=int(np.clip(b.get("n_dates", 8), 3, 12)),
                                    cloud_level=float(np.clip(b.get("cloud_level", 1.0), 0, 3)),
                                    noise=float(np.clip(b.get("noise", 0.004), 0.001, 0.02)), with_change=bool(b.get("change", True)))
                    sid = uuid.uuid4().hex[:10]
                    SESS[sid] = dict(scene=sc)
                    return self._json(dict(sid=sid, meta=scene_meta(sc)))
                if p == "/api/upload":
                    sid = q.get("sid", [None])[0] or uuid.uuid4().hex[:10]
                    s = SESS.setdefault(sid, dict(scene=None, files=[]))
                    s.setdefault("files", []).append((q.get("name", ["file.tif"])[0], self._body()))
                    return self._json(dict(sid=sid, n_files=len(s["files"])))
                if p == "/api/upload/build":
                    b = json.loads(self._body() or b"{}"); s = SESS.get(b.get("sid"))
                    if not s or not s.get("files"):
                        return self._json(dict(error="no files uploaded"), 400)
                    try:
                        sc = scene_from_geotiffs(s["files"])
                    except Exception as e:
                        return self._json(dict(error=str(e)), 400)
                    s["scene"] = sc
                    return self._json(dict(sid=b["sid"], meta=scene_meta(sc)))
                if p == "/api/run":
                    b = json.loads(self._body() or b"{}"); sid = b.get("sid")
                    if sid not in SESS or SESS[sid].get("scene") is None:
                        return self._json(dict(error="unknown session"), 404)
                    jid = uuid.uuid4().hex[:10]
                    JOBS[jid] = dict(status="running", stages={}, sid=sid)
                    threading.Thread(target=run_job, args=(jid, sid, b.get("gate", "evidence")), daemon=True).start()
                    return self._json(dict(job=jid))
                return self._json(dict(error="unknown"), 404)
            except Exception as e:
                traceback.print_exc(); self._json(dict(error=str(e)), 500)
    return H


def serve(host="127.0.0.1", port=8000):
    threading.Thread(target=boot, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), make_handler())
    print(f"GeoSR-X on http://{host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
