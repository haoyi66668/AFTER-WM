import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image

ROBOTS = ("panda", "ur5e")
KS = (1, 5, 10, 20, 40)
ROWS = ("afterwm", "compared")


def frames(outputs):
    d = os.path.join(outputs, "examples", "frames")
    return {(ds, row, k): np.asarray(Image.open(os.path.join(d, f"{ds}_{row}_k{k}.png")).convert("RGB"))
            for ds in ROBOTS for row in ROWS for k in KS}


def crop(render_dir, outputs):
    out = {}
    for ds in ROBOTS:
        with open(os.path.join(outputs, "examples", f"{ds}.json")) as f:
            ex = json.load(f)
        img = np.asarray(Image.open(os.path.join(render_dir, f"render_{ds}.png")).convert("RGB"))
        for row in ROWS:
            for k in KS:
                x0, y0, x1, y1 = ex["frames"][f"{row}|k{k}"]["crop_box_png"]
                out[(ds, row, k)] = np.ascontiguousarray(img[y0:y1, x0:x1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cut the 20 example frames out of the two rendered example figures.")
    ap.add_argument("--render-dir", required=True, help="directory with render_panda.png and render_ur5e.png")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    shipped = frames(a.outputs) if os.path.isdir(os.path.join(a.outputs, "examples", "frames")) else {}
    same = 0
    for (ds, row, k), x in crop(a.render_dir, a.outputs).items():
        Image.fromarray(x).save(os.path.join(a.out, f"{ds}_{row}_k{k}.png"))
        if (ds, row, k) in shipped:
            same += int(hashlib.sha256(x.tobytes()).hexdigest() ==
                        hashlib.sha256(np.ascontiguousarray(shipped[(ds, row, k)]).tobytes()).hexdigest())
    print(json.dumps({"frames": 20, "equal_to_released_frames": same}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
