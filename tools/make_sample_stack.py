"""Write a synthetic multi-date Sentinel-2-like stack as GeoTIFFs (x10000 integers) to try the Upload path."""
import os, sys
import numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.synth import make_scene

out = sys.argv[1] if len(sys.argv) > 1 else "sample_stack"
os.makedirs(out, exist_ok=True)
sc = make_scene(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 31)
keys = [1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, sc.epsg, 3076, 0, 1, 9001]
for i, d in enumerate(sc.dates):
    arr = (sc.lr[i] * 10000).astype(np.uint16)
    ex = [(33550, "d", 3, (10.0, 10.0, 0.0), True), (33922, "d", 6, (0, 0, 0, sc.origin[0], sc.origin[1], 0), True),
          (34735, "H", len(keys), tuple(keys), True)]
    tifffile.imwrite(os.path.join(out, f"S2_{d}.tif"), arr, photometric="minisblack", planarconfig="separate", extratags=ex)
print(f"wrote {len(sc.dates)} GeoTIFFs to {out}/")
