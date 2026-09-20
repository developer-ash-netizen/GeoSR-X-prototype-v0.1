(() => {
const $ = s => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(r => r.json());
const post = (p, b) => api(p, {method: "POST", body: JSON.stringify(b || {})});
const f = (x, d = 3) => x == null || isNaN(x) ? "–" : (+x).toFixed(d);
const pct = x => x == null ? "–" : (100 * x).toFixed(1) + "%";
const PROV = ["multi-date observed", "single-date observed", "prior-generated"];
const PC = ["var(--teal)", "var(--amber)", "var(--crim)"];
const LAYERS = {
  sr: "GeoSR-X output", bicubic: "Single-date bicubic (what S2 shows)", fid: "Fidelity path", prior: "Prior path (ungated)",
  sr_fcc: "GeoSR-X false colour", gt: "Ground truth 2.5 m", err: "Absolute error", risk: "Hallucination risk",
  uncertainty: "Calibrated 90% bound", alpha: "Gate weight α", E1: "E1 temporal support", E2: "E2 disagreement",
  E3: "E3 structured change", E4: "E4 spectral coherence", provenance: "Provenance"
};
let S = {src: "synth", sid: null, sum: null, tab: "trust", region: null, split: 0.5};

// ---------- boot
async function boot() {
  const st = await api("/api/status");
  $("#bootMsg").textContent = st.ready ? "Ready. Calibration loaded." : st.msg;
  if (!st.ready) return setTimeout(boot, 1500);
  $("#run").disabled = false; S.cal = st.calibration;
}
boot();

// ---------- controls
document.querySelectorAll("#srcSeg button").forEach(b => b.onclick = () => {
  S.src = b.dataset.src;
  document.querySelectorAll("#srcSeg button").forEach(x => x.classList.toggle("on", x === b));
  $("#synthCtl").hidden = S.src !== "synth"; $("#upCtl").hidden = S.src !== "upload";
});
$("#nd").oninput = e => $("#ndO").textContent = e.target.value;
$("#cl").oninput = e => $("#clO").textContent = (+e.target.value).toFixed(1);

function renderRail(stages) {
  $("#rail").innerHTML = stages.map(s => `<li data-s="${s.status}"><b>${s.id}</b><span class="nm">${s.name}</span><span class="tm">${s.time != null ? s.time + " s" : ""}</span><span class="ds">${s.desc}</span></li>`).join("");
}
const STAGE0 = [["L0","Data foundation","cloud mask, quality, sub-pixel co-registration"],["L1","Evidence tensor","E1 support, E2 disagreement, E3 change, E4 spectral"],["L2","Temporal fusion","multi-date fusion through the sensor model"],["L3","Fidelity + prior paths","constrained path and learned-detail path"],["L4","Evidence gate","alpha from L1 only"],["L5","Provenance + risk","3-class provenance, independent hallucination risk"],["L6","Self-consistency","forward-degrade SR against the observation"],["L7","Calibration","region-level split-conformal bound"],["L8","Analysis-ready output","GeoTIFF + sidecar"],["L9","Validation","Tier A / B / C"]].map(a => ({id: a[0], name: a[1], desc: a[2], status: "wait"}));
renderRail(STAGE0);

$("#run").onclick = async () => {
  $("#run").disabled = true; renderRail(STAGE0);
  try {
    let sc;
    if (S.src === "synth") {
      sc = await post("/api/scene", {seed: +$("#seed").value, n_dates: +$("#nd").value, cloud_level: +$("#cl").value, change: $("#chg").checked});
    } else {
      const fl = [...$("#files").files];
      if (fl.length < 3) throw new Error("choose at least 3 GeoTIFFs");
      let sid = null;
      for (const file of fl) {
        const r = await api(`/api/upload?name=${encodeURIComponent(file.name)}${sid ? "&sid=" + sid : ""}`, {method: "POST", body: await file.arrayBuffer()});
        sid = r.sid;
      }
      sc = await post("/api/upload/build", {sid});
    }
    if (sc.error) throw new Error(sc.error);
    S.sid = sc.sid;
    const j = await post("/api/run", {sid: S.sid, gate: $("#gate").value});
    await poll(j.job);
  } catch (e) { $("#bootMsg").textContent = "Error: " + e.message; }
  $("#run").disabled = false;
};

async function poll(jid) {
  for (;;) {
    const j = await api("/api/job/" + jid);
    renderRail(j.stages);
    if (j.status === "error") throw new Error(j.error);
    if (j.status === "done") break;
    await new Promise(r => setTimeout(r, 450));
  }
  S.sum = await api(`/api/session/${S.sid}/summary`);
  $("#bootMsg").textContent = "Done.";
  setupViewer(); renderTab();
}

// ---------- viewer
const L = n => `/api/session/${S.sid}/layer/${n}?t=${S.sid}`;
function setupViewer() {
  const ls = S.sum.layers.filter(l => LAYERS[l]);
  const opts = ls.map(l => `<option value="${l}">${LAYERS[l]}</option>`).join("");
  $("#selA").innerHTML = opts; $("#selB").innerHTML = opts;
  $("#selA").value = "bicubic"; $("#selB").value = "sr";
  $("#empty").hidden = true; $("#frame").hidden = false; $("#legend").hidden = false;
  $("#ovTruthL").hidden = !S.sum.layers.includes("truth_change");
  $("#imgProv").src = L("provenance"); $("#imgReg").src = L("regions");
  if (S.sum.layers.includes("truth_change")) $("#imgTruth").src = L("truth_change");
  S.region = null; $("#imgSel").style.display = "none";
  $("#inspect").innerHTML = "<h2>Region inspector</h2><p class='hint'>Click the image to inspect a region.</p>";
  loadLayers(); applySplit();
}
function loadLayers() {
  $("#imgA").src = L($("#selA").value); $("#imgB").src = L($("#selB").value);
  $("#labA").textContent = LAYERS[$("#selA").value]; $("#labB").textContent = LAYERS[$("#selB").value];
}
$("#selA").onchange = $("#selB").onchange = loadLayers;
function applySplit() {
  const p = S.split * 100;
  $("#imgB").style.clipPath = `inset(0 0 0 ${p}%)`;
  $("#handle").style.left = p + "%";
}
function ov() {
  const o = $("#ovOp").value / 100;
  [["#ovProv", "#imgProv"], ["#ovReg", "#imgReg"], ["#ovTruth", "#imgTruth"]].forEach(([c, i]) => {
    $(i).style.display = $(c).checked ? "block" : "none"; $(i).style.opacity = i === "#imgReg" ? 1 : o;
  });
}
["#ovProv", "#ovReg", "#ovTruth", "#ovOp"].forEach(s => $(s).oninput = ov);

let drag = false;
const frame = $("#frame");
const posOf = e => { const r = frame.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
$("#handle").onpointerdown = e => { drag = true; $("#handle").setPointerCapture(e.pointerId); e.stopPropagation(); };
$("#handle").onpointermove = e => { if (drag) { S.split = Math.min(1, Math.max(0, posOf(e)[0])); applySplit(); } };
$("#handle").onpointerup = () => drag = false;
frame.onclick = async e => {
  if (!S.sum || e.target === $("#handle") || e.target.parentNode === $("#handle")) return;
  const [u, v] = posOf(e), n = S.sum.meta.size_hr[1];
  const x = Math.floor(u * n), y = Math.floor(v * n);
  const r = await api(`/api/session/${S.sid}/region?x=${x}&y=${y}`);
  S.region = r; $("#imgSel").src = `/api/session/${S.sid}/regionmask?id=${r.id}&s=${S.sid}`; $("#imgSel").style.display = "block";
  renderInspector(r);
};
frame.onpointermove = e => {
  if (!S.sum) return; const [u, v] = posOf(e), n = S.sum.meta.size_hr[1], g = S.sum.meta.gsd_hr;
  $("#px").textContent = `px (${Math.floor(u * n)}, ${Math.floor(v * n)})  ·  ${(u * n * g).toFixed(0)} m E, ${(v * n * g).toFixed(0)} m S`;
};

function meter(v, max = 1) { return `<div class="meter"><i style="width:${Math.min(100, 100 * v / max)}%"></i></div>`; }
function renderInspector(r) {
  const c = PC[r.prov_id];
  let h = `<h2>Region ${r.id}</h2><span class="badge" style="background:${c}">${r.provenance}</span>
  <p class="hint">${r.land_cover} · ${r.area_px} px (${(r.area_px * 6.25).toFixed(0)} m²)</p>
  <h3 style="font:600 13px var(--serif);margin:10px 0 2px">Evidence (L1)</h3>
  <div class="kv"><span>E1 clean dates</span><span>${f(r.evidence.E1, 1)} of ${r.evidence.E1_max}</span>
  <span>E2 disagreement</span><span>${f(r.evidence.E2, 2)}×</span>
  <span>E3 change share</span><span>${pct(r.evidence.E3)}</span>
  <span>E4 spectral</span><span>${f(r.evidence.E4, 2)}</span>
  <span>Sub-pixel phases</span><span>${f(r.evidence.diversity, 1)}</span></div>
  <h3 style="font:600 13px var(--serif);margin:12px 0 2px">Gate and source</h3>
  <div class="kv"><span>Gate α</span><span>${f(r.gate_alpha, 2)}</span><span>Detail from prior</span><span>${pct(r.prior_share)}</span></div>
  <h3 style="font:600 13px var(--serif);margin:12px 0 2px">Hallucination risk (independent of gate)</h3>
  <div class="kv"><span>Total</span><span>${f(r.risk.total, 2)}</span><span>Temporal</span><span>${f(r.risk.temporal, 2)}</span><span>Spectral</span><span>${f(r.risk.spectral, 2)}</span><span>Texture novelty</span><span>${f(r.risk.texture, 2)}</span></div>
  ${meter(r.risk.total)}
  <h3 style="font:600 13px var(--serif);margin:12px 0 2px">Uncertainty</h3>
  <div class="kv"><span>Self-consistency</span><span>${f(r.self_consistency.residual_sigma, 2)}σ</span><span>90% bound (mean abs. refl. error)</span><span>±${f(r.uncertainty.p90_bound, 4)}</span></div>`;
  if (r.true_mae != null) h += `<div class="kv" style="margin-top:6px"><span>True error (synthetic)</span><span>${f(r.true_mae, 4)} ${r.covered ? "· inside bound" : "· <b style='color:var(--crim)'>outside bound</b>"}</span></div>`;
  $("#inspect").innerHTML = h;
}

// ---------- tabs
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {
  S.tab = b.dataset.t; document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b)); renderTab();
});
function renderTab() {
  if (!S.sum) return;
  ({trust, val, chg, cal, acq, exp})[S.tab]();
}
const P = html => $("#panel").innerHTML = html;

function trust() {
  const s = S.sum, a = s.provenance.area, mb = s.provenance.mean_bound, tm = s.provenance.true_mae;
  const bar = PROV.map((n, i) => `<div style="width:${100 * a[n]}%;background:${PC[i]}" title="${n}"></div>`).join("");
  const rows = PROV.map((n, i) => `<tr><td><i class="sw" style="background:${PC[i]}"></i>${n}</td><td>${pct(a[n])}</td><td>${f(mb[n], 4)}</td><td>${tm ? f(tm[n], 4) : "–"}</td></tr>`).join("");
  P(`<h3>Where the pixels come from</h3>
  <div class="bar">${bar}</div>
  <table><tr><th>Provenance class</th><th>Area</th><th>Mean 90% bound</th><th>True error</th></tr>${rows}</table>
  <p class="hint">Provenance is decided by the evidence tensor (E1 support, sub-pixel phase diversity) and by how much of a region's detail came from the prior path, not by a single confidence scalar. Single-date and prior-generated regions carry visibly larger errors: that separation is the point.</p>
  <div class="two"><div><h3>Evidence summary</h3><div class="kv"><span>Mean clean dates (E1)</span><span>${f(s.E.E1_mean, 2)}</span><span>Structured-change area (E3)</span><span>${pct(s.E.E3_frac)}</span><span>Spectral coherence (E4)</span><span>${f(s.E.E4_mean, 2)}</span><span>Mean gate α</span><span>${f(s.alpha_mean, 2)}</span><span>Regions</span><span>${s.n_regions}</span></div></div>
  <div><h3>Fidelity constraint</h3><div class="kv"><span>degrade(SR) vs LR (RMS)</span><span>${f(s.fid_rms, 5)}</span><span>Noise σ (per band)</span><span>${s.sigma.map(x => f(x, 4)).join(", ")}</span><span>Gate mode</span><span>${s.gate}</span></div></div></div>`);
}

function scatter(rows) {
  const W = 360, H = 240, m = 38, mx = Math.max(...rows.map(r => r.omission), .1) * 1.1, my = Math.max(...rows.map(r => r.hallucination), .1) * 1.1;
  const X = v => m + (W - m - 10) * v / mx, Y = v => H - m - (H - m - 10) * v / my;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%"><line x1="${m}" y1="${H - m}" x2="${W - 10}" y2="${H - m}" stroke="#9aa6b1"/><line x1="${m}" y1="10" x2="${m}" y2="${H - m}" stroke="#9aa6b1"/>
  <text x="${W / 2 - 40}" y="${H - 8}">omission (misses real detail)</text><text transform="translate(12 ${H / 2 + 40}) rotate(-90)">hallucination</text>`;
  rows.forEach((r, i) => { const hl = r.product.startsWith("GeoSR"); s += `<circle cx="${X(r.omission)}" cy="${Y(r.hallucination)}" r="${hl ? 7 : 5}" fill="${hl ? "var(--accent)" : "#8b98a6"}"/><text x="${X(r.omission) + 8}" y="${Y(r.hallucination) + (i % 2 ? 12 : -6)}" ${hl ? 'style="fill:var(--ink);font-weight:600"' : ""}>${r.product}</text>`; });
  return s + "</svg>";
}
function val() {
  const v = S.sum.validation; let h = "";
  if (v.tier_a) {
    h += `<h3>Tier A — absolute quality against HR truth</h3><div class="two"><table><tr><th>Product</th><th>PSNR</th><th>SSIM</th><th>SAM°</th><th>Halluc.</th><th>Omiss.</th><th>Δ dB</th></tr>` +
      v.tier_a.map(r => `<tr class="${r.product.startsWith("GeoSR") ? "hl" : ""}"><td>${r.product}</td><td>${f(r.psnr, 2)}</td><td>${f(r.ssim, 3)}</td><td>${f(r.sam_deg, 2)}</td><td>${f(r.hallucination, 3)}</td><td>${f(r.omission, 3)}</td><td>${f(r.improvement_db, 2)}</td></tr>`).join("") + `</table><div>${scatter(v.tier_a)}</div></div>
      <p class="caveat">Hallucination/omission are edge-based proxies in the spirit of ESA opensr-test, not that library. On this in-distribution synthetic set the ungated prior scores slightly higher PSNR than the gated output: the gate costs some accuracy here and buys conservatism where evidence is thin. That trade-off is measured, not assumed to favour us.</p>`;
  }
  if (v.tier_b) {
    h += `<h3>Tier B — leave-one-date-out (date ${v.tier_b.hidden_date + 1} hidden, ${v.tier_b.n_pixels} px)</h3><table><tr><th>Predictor</th><th>RMSE (refl.)</th><th>RMSE (noise σ)</th></tr>` +
      v.tier_b.rows.map((r, i) => `<tr class="${i === 0 ? "hl" : ""}"><td>${r.method}</td><td>${f(r.rmse, 5)}</td><td>${f(r.rmse_sigma, 1)}</td></tr>`).join("") + `</table><p class="caveat">${v.tier_b.caveat}</p>`;
  } else h += `<h3>Tier B</h3><p class="hint">Not enough clean revisits to hide one.</p>`;
  if (v.tier_c) {
    h += `<h3>Tier C — downstream field-boundary delta</h3><table><tr><th>Input product</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr>` +
      v.tier_c.map(r => `<tr class="${r.product.startsWith("GeoSR") ? "hl" : ""}"><td>${r.product}</td><td>${f(r.precision, 2)}</td><td>${f(r.recall, 2)}</td><td>${f(r.f1, 2)}</td><td>${f(r.iou, 2)}</td></tr>`).join("") + `</table><p class="hint">Same Canny-on-NDVI boundary extractor for every input; 2-px tolerance to the true parcel boundaries.</p>`;
  }
  if (!v.tier_a) h += `<p class="hint">Tiers A and C need HR truth; uploaded stacks get Tier B only.</p>`;
  P(h);
}
function chg() {
  const c = S.sum.validation.change;
  if (!c) return P(`<p class="hint">No known change in this scene (or no truth available). Enable “Real change” in the synthetic controls.</p>`);
  P(`<h3>Change-aware evidence (E3) protects real change</h3>
  <div class="kv"><span>True change pixels flagged as structured</span><span>${pct(c.change_recall)}</span><span>Stable area wrongly flagged</span><span>${pct(c.false_flag_rate)}</span></div>
  <h3>Reconstruction error inside the changed area</h3>
  <table><tr><th>Method</th><th>RMSE vs truth</th></tr>${c.rows.map((r, i) => `<tr class="${i === 2 ? "hl" : ""}"><td>${r.method}</td><td>${f(r.rmse, 4)}</td></tr>`).join("")}</table>
  <p class="hint">A naive temporal-consistency fusion averages the pre- and post-change dates and blurs exactly the signal a disaster or intelligence use case exists to detect. Toggle “true change” on the viewer to see where it is.</p>`);
}
function reliab(rep, sceneCov) {
  const W = 340, H = 300, m = 40, X = v => m + (W - m - 12) * (v - .45) / .55, Y = v => H - m - (H - m - 12) * (v - .45) / .55;
  const col = {"water": "#2a6fb0", "vegetation": "#3d8b3d", "built-up": "#7a5c9e", "bare / other": "#b07a2a"};
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%"><line x1="${X(.45)}" y1="${Y(.45)}" x2="${X(1)}" y2="${Y(1)}" stroke="#9aa6b1" stroke-dasharray="4"/>
  <line x1="${m}" y1="${H - m}" x2="${W - 12}" y2="${H - m}" stroke="#9aa6b1"/><line x1="${m}" y1="12" x2="${m}" y2="${H - m}" stroke="#9aa6b1"/>
  <text x="${W / 2 - 50}" y="${H - 8}">nominal coverage</text><text transform="translate(12 ${H / 2 + 40}) rotate(-90)">observed coverage</text>`;
  [.5, .7, .9].forEach(t => s += `<text x="${X(t) - 8}" y="${H - m + 14}">${t}</text><text x="${m - 28}" y="${Y(t) + 4}">${t}</text>`);
  const line = (arr, c, w, d) => `<polyline fill="none" stroke="${c}" stroke-width="${w}" ${d ? 'stroke-dasharray="3 3"' : ""} points="${rep.levels.map((l, i) => arr[i] == null ? "" : X(l) + "," + Y(arr[i])).join(" ")}"/>`;
  Object.entries(rep.per_class).forEach(([k, v]) => s += line(v.coverage, col[k], 1.4));
  s += line(rep.global_only, "#8b98a6", 1.6, true) + line(rep.overall, "var(--accent)", 3);
  return s + "</svg>";
}
function cal() {
  const r = S.sum.calibration, i90 = r.levels.indexOf(.9), v = S.sum.validation;
  const rows = Object.entries(r.per_class).map(([k, c]) => `<tr><td>${k}</td><td>${c.n_cal}</td><td>${c.n_test}</td><td>${c.coverage[i90] == null ? "–" : pct(c.coverage[i90])}</td><td>${f(c.q90, 2)}</td></tr>`).join("");
  P(`<h3>Reliability: stated coverage vs observed coverage</h3><div class="two"><div>${reliab(r)}<p class="hint">Bold line: all regions, class-adaptive. Dashed: one global quantile. Thin lines: per land-cover class. Diagonal = perfect.</p></div>
  <div><table><tr><th>Class</th><th>Cal. regions</th><th>Test regions</th><th>Coverage @ 90%</th><th>q₉₀</th></tr>${rows}
  <tr class="hl"><td>All regions</td><td>${r.n_cal_regions}</td><td>${r.n_test}</td><td>${pct(r.overall[i90])}</td><td>–</td></tr></table>
  ${v.scene_coverage90 != null ? `<p class="hint">This scene: ${pct(v.scene_coverage90)} of regions have true error inside the stated 90% bound.</p>` : ""}
  <p class="caveat">Split-conformal at region level over held-out synthetic scenes (disjoint from prior, uncertainty-model and calibration data). Conformal validity assumes exchangeable regions; spatial correlation makes that approximate, and per-class coverage on small classes is noisy. Classes with under 40 calibration regions fall back to the pooled quantile.</p></div></div>`);
}
function acq() {
  const s = S.sum, m = s.meta;
  P(`<h3>Acquisitions and co-registration</h3><div class="thumbs">${m.dates.map((d, i) => `<figure class="${i === s.target ? "ref" : ""}"><img src="/api/session/${S.sid}/layer/lr?i=${i}&t=${S.sid}"><div>${d}${i === s.target ? " · target" : ""}</div><div>clean ${pct(s.clean[i])} · NCC ${f(s.ncc[i], 2)}${s.usable[i] ? "" : " · dropped"}</div><div>shift ${f(s.shifts[i][0], 2)}, ${f(s.shifts[i][1], 2)} px${s.reg_err ? " · err " + f(s.reg_err[i], 2) : ""}</div></figure>`).join("")}</div>
  <p class="hint">Blue tint = pixels a cloud mask flagged. Alignment is verified before any metric is trusted; dates whose post-registration correlation is too low are dropped from fusion. “err” is the registration error against the synthetic truth.</p>`);
}
function exp() {
  const b = `/api/session/${S.sid}/download/`;
  P(`<h3>Analysis-ready output</h3><a class="dl" href="${b}geotiff">GeoTIFF (SR + provenance + uncertainty)</a><a class="dl" href="${b}sidecar">Sidecar: error budget, lineage</a><a class="dl" href="${b}coverage">Coverage report</a>
  <div class="kv" style="margin-top:12px"><span>Bands 1–4</span><span>SR reflectance B02, B03, B04, B08 at ${S.sum.meta.gsd_hr} m</span><span>Band 5</span><span>provenance class (0 multi-date, 1 single-date, 2 prior-generated)</span><span>Band 6</span><span>calibrated 90% upper bound on region-mean absolute error</span><span>CRS</span><span>EPSG:${S.sum.meta.epsg}, transform preserved</span></div>
  <p class="hint">Open in QGIS and switch on band 5 to see measurement versus reconstruction.</p>`);
}
})();
