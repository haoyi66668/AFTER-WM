from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(a):
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def npz_digest(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: array_digest(z[k]) for k in sorted(z.files)}


def read_sums(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                h, name = line.rstrip("\n").split("  ", 1)
                out[name] = h
    return out


def build(data_root, files, sums_out, arrays_out):
    sums, arrays = [], {}
    for rel in files:
        p = os.path.join(data_root, rel)
        sums.append(f"{file_sha256(p)}  data/{rel}")
        arrays[f"data/{rel}"] = npz_digest(p)
    with open(sums_out, "w") as f:
        f.write("\n".join(sums) + "\n")
    with open(arrays_out, "w") as f:
        json.dump(arrays, f, indent=1, sort_keys=True)
        f.write("\n")


def check(data_root, sums_path, arrays_path, only=None):
    sums = read_sums(sums_path)
    with open(arrays_path) as f:
        ref = json.load(f)
    rows, ok = [], True
    for name in sorted(ref):
        rel = name[len("data/"):]
        ds = rel.split("/", 1)[0]
        if only and ds not in only:
            continue
        p = os.path.join(data_root, rel)
        if not os.path.exists(p):
            rows.append({"file": name, "present": False, "arrays_identical": False})
            ok = False
            continue
        got = npz_digest(p)
        diff = sorted(k for k in set(ref[name]) | set(got) if ref[name].get(k) != got.get(k))
        row = {"file": name, "present": True, "arrays": len(got), "arrays_identical": not diff,
               "differing_arrays": diff, "file_sha256": file_sha256(p), "file_identical": file_sha256(p) == sums.get(name)}
        ok &= not diff
        rows.append(row)
    return ok, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare npz files with the released file and array sha256 lists.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--sums", default=os.path.join(HERE, "SHA256SUMS"))
    ap.add_argument("--arrays", default=os.path.join(HERE, "ARRAY_SHA256.json"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--build", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.build is not None:
        build(a.data, a.build, a.sums, a.arrays)
        return 0
    only = set(a.datasets.split(",")) if a.datasets else None
    ok, rows = check(a.data, a.sums, a.arrays, only)
    res = {"all_arrays_identical": ok, "files": rows}
    if a.report:
        with open(a.report, "w") as f:
            json.dump(res, f, indent=1)
    for r in rows:
        print(f"{'OK  ' if r['arrays_identical'] else 'FAIL'} {r['file']} arrays={r.get('arrays', 0)} "
              f"file_identical={r.get('file_identical', False)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
