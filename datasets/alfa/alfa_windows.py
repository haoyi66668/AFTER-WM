import argparse, csv, hashlib, io, json, math, os, re, sys, zipfile
import numpy as np

VERSION = "alfa_def v1 (2026-10-21)"
DT_NS = 50_000_000
T_STATES, T_ACTS = 301, 300
STRIDE = 160
GROUP_RE = re.compile(r"^carbonZ_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_(\d))?_(.+)$")
TOPICS = {
    "att": ("mavros-imu-data", 1_000_000_000),
    "gyro": ("mavros-imu-data_raw", 350_000_000),
    "vel": ("mavros-nav_info-velocity", 350_000_000),
    "aspd": ("mavros-nav_info-airspeed", 350_000_000),
    "alt": ("mavros-global_position-rel_alt", 750_000_000),
    "roll": ("mavros-nav_info-roll", 350_000_000),
    "pitch": ("mavros-nav_info-pitch", 350_000_000),
}
STATE_SUFFIX = "mavros-state"
OBS_NAMES = ("gbx", "gby", "sin_yaw", "cos_yaw", "p", "q", "r", "vN", "vE", "vD", "h", "airspeed")
ACT_NAMES = ("roll_cmd", "pitch_cmd", "airspeed_cmd")
AUTONOMOUS_MODES = ("GUIDED", "AUTO")
STATE_TOL_NS = 1_500_000_000
DEG = math.pi / 180.0
SURFACE_WORDS = ("aileron", "rudder", "elevator")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_sha256(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write_npz_deterministic(path, arrays):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists (never overwritten)")
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, buf.getvalue())
    os.replace(tmp, path)


def write_json(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists (never overwritten)")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def parse_name(seq):
    m = GROUP_RE.match(seq)
    if not m:
        raise SystemExit(f"unexpected sequence name {seq}")
    stamp, part, tail = m.group(1), m.group(2), m.group(3)
    return {"group": "carbonZ_" + stamp, "part": int(part) if part else 0, "tail": tail}


def failure_family(tail):
    if tail == "no_ground_truth":
        return "unlabelled"
    if tail.startswith("no_failure"):
        return "none"
    if tail.startswith("engine_failure"):
        return "engine"
    if any(w in tail for w in SURFACE_WORDS):
        return "surface"
    raise SystemExit(f"unknown failure tail {tail}")


def stratum_of_group(families):
    fs = set(families)
    if "unlabelled" in fs:
        return "unlabelled"
    if "engine" in fs:
        return "engine"
    if "surface" in fs:
        return "surface"
    return "none"


class Archive:
    def __init__(self, zpath):
        self.z = zipfile.ZipFile(zpath)
        self.names = self.z.namelist()
        seqs = set()
        for n in self.names:
            parts = n.split("/")
            if len(parts) >= 3 and parts[0] == "processed" and parts[1].startswith("carbonZ_"):
                seqs.add(parts[1])
        self.sequences = sorted(seqs)

    def member(self, seq, suffix):
        n = f"processed/{seq}/{seq}-{suffix}.csv"
        return n if n in self.names else None

    def rows(self, seq, suffix):
        n = self.member(seq, suffix)
        if n is None:
            return None, None
        with self.z.open(n) as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
            hdr = next(r)
            return hdr, [row for row in r]

    def failure_members(self, seq):
        pre = f"processed/{seq}/{seq}-failure_status-"
        return sorted(n for n in self.names if n.startswith(pre) and n.endswith(".csv"))


def column(hdr, rows, name, conv=float):
    j = hdr.index(name)
    return [conv(r[j]) for r in rows]


def load_sequence(ar, seq, with_signals=True):
    out = {}
    for key, (suf, _) in TOPICS.items():
        hdr, rows = ar.rows(seq, suf)
        if hdr is None:
            out[key] = None
            continue
        d = {"t": column(hdr, rows, "%time", int)}
        if with_signals:
            if key == "att":
                for c in ("x", "y", "z", "w"):
                    d["q" + c] = column(hdr, rows, f"field.orientation.{c}")
            elif key == "gyro":
                for c in ("x", "y", "z"):
                    d["w" + c] = column(hdr, rows, f"field.angular_velocity.{c}")
            elif key == "vel":
                for c in ("x", "y", "z"):
                    d["v" + c] = column(hdr, rows, f"field.meas_{c}")
            elif key == "aspd":
                d["meas"] = column(hdr, rows, "field.measured")
                d["cmd"] = column(hdr, rows, "field.commanded")
            elif key == "alt":
                d["h"] = column(hdr, rows, "field.data")
            elif key in ("roll", "pitch"):
                d["cmd"] = column(hdr, rows, "field.commanded")
        out[key] = d
    hdr, rows = ar.rows(seq, STATE_SUFFIX)
    out["state"] = None if hdr is None else {"t": column(hdr, rows, "%time", int),
                                            "armed": column(hdr, rows, "field.armed", str),
                                            "mode": column(hdr, rows, "field.mode", str)}
    onset, ftopics = None, []
    for n in ar.failure_members(seq):
        with ar.z.open(n) as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
            hdr = next(r)
            rws = [row for row in r]
        ftopics.append(n.rsplit("failure_status-", 1)[1][:-4])
        for row in rws:
            if row[1] not in ("0", "0.0", "False", "false"):
                t = int(row[0])
                onset = t if onset is None else min(onset, t)
                break
    out["fail_onset_ns"] = onset
    out["fail_topics"] = ftopics
    return out


def bracket(ts, a, b):
    import bisect
    lo = bisect.bisect_right(ts, a) - 1
    hi = bisect.bisect_left(ts, b)
    if lo < 0 or hi >= len(ts):
        return None
    return lo, hi


def seq_grid(data):
    t0 = max(data[k]["t"][0] for k in TOPICS)
    t1 = min(data[k]["t"][-1] for k in TOPICS)
    if t1 <= t0:
        return t0, 0
    return t0, (t1 - t0) // DT_NS + 1


def window_eligibility(data, t0, k, with_signals):
    a = t0 + STRIDE * k * DT_NS
    b = a + (T_STATES - 1) * DT_NS
    reasons, gaps = [], {}
    for key, (_, gmax) in TOPICS.items():
        ts = data[key]["t"]
        br = bracket(ts, a, b)
        if br is None:
            reasons.append(f"{key}:not_covered")
            continue
        lo, hi = br
        seg = ts[lo:hi + 1]
        mg = max((y - x) for x, y in zip(seg, seg[1:])) if len(seg) > 1 else 0
        gaps[key] = mg
        if any(y <= x for x, y in zip(seg, seg[1:])):
            reasons.append(f"{key}:non_increasing_time")
        if mg > gmax:
            reasons.append(f"{key}:gap_{mg / 1e9:.3f}s")
        if with_signals:
            cols = [c for c in data[key] if c != "t"]
            for c in cols:
                if not all(math.isfinite(v) for v in data[key][c][lo:hi + 1]):
                    reasons.append(f"{key}:{c}:nonfinite")
            if key == "att":
                for i in range(lo, hi + 1):
                    n2 = (data[key]["qx"][i] ** 2 + data[key]["qy"][i] ** 2 + data[key]["qz"][i] ** 2
                          + data[key]["qw"][i] ** 2)
                    if not (n2 > 0.25):
                        reasons.append("att:invalid_quaternion")
                        break
    st = data["state"]
    if st is None:
        reasons.append("state:missing")
    else:
        near = [i for i, t in enumerate(st["t"]) if abs(t - a) <= STATE_TOL_NS]
        if not near:
            reasons.append("state:no_message_near_window_start")
        idx = [i for i, t in enumerate(st["t"]) if a - STATE_TOL_NS <= t <= b]
        for i in idx:
            if st["armed"][i] not in ("1", "True", "true") or st["mode"][i] not in AUTONOMOUS_MODES:
                reasons.append(f"state:{st['mode'][i]}/{st['armed'][i]}")
                break
    return reasons, gaps


def build_pool(zpath, with_signals=False):
    ar = Archive(zpath)
    seqs, windows = {}, []
    for seq in ar.sequences:
        nm = parse_name(seq)
        data = load_sequence(ar, seq, with_signals=with_signals)
        fam = failure_family(nm["tail"])
        rec = {"group": nm["group"], "part": nm["part"], "family": fam, "tail": nm["tail"],
               "fail_topics": data["fail_topics"], "fail_onset_ns": data["fail_onset_ns"]}
        missing = [k for k in TOPICS if data[k] is None or len(data[k]["t"]) < 2]
        if missing:
            rec.update({"eligible_windows": 0, "reason": f"missing topics {missing}"})
            seqs[seq] = rec
            continue
        t0, L = seq_grid(data)
        K = 0 if L < T_STATES else (L - T_STATES) // STRIDE + 1
        rec.update({"grid_t0_ns": t0, "grid_states": L, "windows": K,
                    "duration_s": (L - 1) * DT_NS / 1e9 if L else 0.0})
        if fam in ("engine", "surface") and data["fail_onset_ns"] is None:
            rec["label_problem"] = "failure sequence without a non-zero failure_status message"
        ne = 0
        for k in range(K):
            reasons, gaps = window_eligibility(data, t0, k, with_signals)
            w = {"sequence": seq, "group": nm["group"], "k": k, "tau0_ns": t0 + STRIDE * k * DT_NS,
                 "eligible": not reasons, "reasons": reasons, "max_gap_s": {g: v / 1e9 for g, v in gaps.items()}}
            windows.append(w)
            ne += int(not reasons)
        rec["eligible_windows"] = ne
        seqs[seq] = rec
    groups = {}
    for seq, r in seqs.items():
        g = groups.setdefault(r["group"], {"sequences": [], "families": [], "eligible_windows": 0})
        g["sequences"].append(seq)
        g["families"].append(r["family"])
        g["eligible_windows"] += r.get("eligible_windows", 0)
    for g in groups.values():
        g["stratum"] = stratum_of_group(g["families"])
    return {"version": VERSION, "zip_sha256": sha256_file(zpath), "sequences": seqs, "groups": groups,
            "windows": windows}


def cmd_pool(a):
    os.makedirs(a.out, exist_ok=True)
    pool = build_pool(a.zip, with_signals=a.with_signals)
    write_json(os.path.join(a.out, "POOL.json"), pool)
    by = {}
    for g, r in pool["groups"].items():
        s = by.setdefault(r["stratum"], {"groups": 0, "eligible_windows": 0})
        s["groups"] += 1
        s["eligible_windows"] += r["eligible_windows"]
    nw = len(pool["windows"])
    ne = sum(w["eligible"] for w in pool["windows"])
    print(json.dumps({"sequences": len(pool["sequences"]), "groups": len(pool["groups"]), "windows": nw,
                      "eligible": ne, "by_stratum": by}, indent=1))
    return 0


SPLITS = ("test", "test_b", "val", "train")
FILE_TAG = {"train": "train", "val": "val", "test": "test", "test_b": "testB"}


def select(pool, quotas):
    seed = int(quotas["seed"])
    probes = set(quotas["probe_groups"])
    out = {s: [] for s in SPLITS}
    log = {}
    for si, stratum in enumerate(quotas["strata"]):
        q = quotas["quotas"][stratum]
        gs = sorted(g for g, r in pool["groups"].items() if r["stratum"] == stratum and r["eligible_windows"] > 0)
        if len(gs) != sum(q[s] for s in SPLITS):
            raise SystemExit(4)
        perm = np.random.default_rng([seed, si]).permutation(len(gs))
        order = [gs[i] for i in perm]
        left = list(order)
        for s in SPLITS:
            need = q[s]
            take = []
            for g in list(left):
                if len(take) == need:
                    break
                if s != "train" and g in probes:
                    continue
                take.append(g)
                left.remove(g)
            if len(take) != need:
                raise SystemExit(4)
            out[s] += take
        log[stratum] = {"order": order, "assigned": {s: [g for g in out[s] if g in gs] for s in SPLITS}}
    return out, log


def interp(ts, vs, t, lo):
    t_a, t_b = ts[lo], ts[lo + 1] if lo + 1 < len(ts) else ts[lo]
    if t == t_a or t_b == t_a:
        return vs[lo]
    w = (t - t_a) / (t_b - t_a)
    return vs[lo] + (vs[lo + 1] - vs[lo]) * w


def grid_values(ts, vs, times):
    import bisect
    out = []
    for t in times:
        lo = bisect.bisect_right(ts, t) - 1
        if lo < 0 or (t > ts[-1]):
            raise ValueError("grid time outside the topic coverage")
        out.append(interp(ts, vs, t, lo))
    return out


def interval_mean(ts, vs, a, b):
    import bisect
    pts = [a] + [t for t in ts[bisect.bisect_right(ts, a): bisect.bisect_left(ts, b)]] + [b]
    vals = grid_values(ts, vs, pts)
    acc = 0.0
    for i in range(len(pts) - 1):
        acc += (pts[i + 1] - pts[i]) * (vals[i] + vals[i + 1]) * 0.5
    return acc / (b - a)


def window_arrays(data, tau0):
    times = [tau0 + i * DT_NS for i in range(T_STATES)]
    att = data["att"]
    n = len(att["t"])
    r20, r21, r00, r10 = [], [], [], []
    for i in range(n):
        x, y, z, w = att["qx"][i], att["qy"][i], att["qz"][i], att["qw"][i]
        nn = math.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / nn, y / nn, z / nn, w / nn
        r20.append(2.0 * (x * z - w * y))
        r21.append(2.0 * (y * z + w * x))
        r00.append(1.0 - 2.0 * (y * y + z * z))
        r10.append(2.0 * (x * y + w * z))
    g20 = grid_values(att["t"], r20, times)
    g21 = grid_values(att["t"], r21, times)
    g00 = grid_values(att["t"], r00, times)
    g10 = grid_values(att["t"], r10, times)
    syaw, cyaw = [], []
    for c, s in zip(g00, g10):
        nn = math.sqrt(c * c + s * s)
        syaw.append(s / nn)
        cyaw.append(c / nn)
    cols = [g20, g21, syaw, cyaw]
    for c in ("wx", "wy", "wz"):
        cols.append(grid_values(data["gyro"]["t"], data["gyro"][c], times))
    for c in ("vx", "vy", "vz"):
        cols.append(grid_values(data["vel"]["t"], data["vel"][c], times))
    cols.append(grid_values(data["alt"]["t"], data["alt"]["h"], times))
    cols.append(grid_values(data["aspd"]["t"], data["aspd"]["meas"], times))
    obs = [[cols[j][i] for j in range(len(cols))] for i in range(T_STATES)]
    roll = [v * DEG for v in data["roll"]["cmd"]]
    pitch = [v * DEG for v in data["pitch"]["cmd"]]
    act = []
    for i in range(T_ACTS):
        a, b = times[i], times[i + 1]
        act.append([interval_mean(data["roll"]["t"], roll, a, b), interval_mean(data["pitch"]["t"], pitch, a, b),
                    interval_mean(data["aspd"]["t"], data["aspd"]["cmd"], a, b)])
    return obs, act, times


def build_split(ar, pool, groups, stratum_of):
    wins = [w for w in pool["windows"] if w["eligible"] and w["group"] in set(groups)]
    wins.sort(key=lambda w: (w["group"], w["sequence"], w["k"]))
    cache = {}
    O, A, meta = [], [], []
    for w in wins:
        seq = w["sequence"]
        if seq not in cache:
            cache.clear()
            cache[seq] = load_sequence(ar, seq, with_signals=True)
        data = cache[seq]
        obs, act, times = window_arrays(data, w["tau0_ns"])
        O.append(obs)
        A.append(act)
        meta.append((w, data["fail_onset_ns"], times))
    N = len(wins)
    obs = np.asarray(O, dtype=np.float64).astype(np.float32).reshape(N, T_STATES, len(OBS_NAMES))
    act = np.asarray(A, dtype=np.float64).astype(np.float32).reshape(N, T_ACTS, len(ACT_NAMES))
    fail_step = np.full(N, np.nan, dtype=np.float64)
    onset_index = np.full(N, -1, dtype=np.int64)
    seg = np.zeros((N, T_STATES), dtype=np.int8)
    for i, (w, on, times) in enumerate(meta):
        if on is not None:
            fs = (on - w["tau0_ns"]) / DT_NS
            fail_step[i] = fs
            for j, t in enumerate(times):
                seg[i, j] = 1 if t >= on else 0
            first = [j for j, t in enumerate(times) if t >= on]
            if first and first[0] > 0 and times[0] < on:
                onset_index[i] = first[0]
    seqinfo = pool["sequences"]
    fam = np.asarray([seqinfo[w["sequence"]]["family"] for w, _, _ in meta], dtype="<U16")
    tail = np.asarray([seqinfo[w["sequence"]]["tail"] for w, _, _ in meta], dtype="<U64")
    arrays = {
        "obs": obs, "act": act,
        "parent": np.asarray([w["group"] for w, _, _ in meta], dtype="<U40"),
        "cohort": np.asarray(["real"] * N, dtype="<U8"),
        "sequence": fam,
        "failure_tail": tail,
        "stratum": np.asarray([stratum_of[w["group"]] for w, _, _ in meta], dtype="<U16"),
        "seq_name": np.asarray([w["sequence"] for w, _, _ in meta], dtype="<U80"),
        "window_k": np.asarray([w["k"] for w, _, _ in meta], dtype=np.int64),
        "tau0_ns": np.asarray([w["tau0_ns"] for w, _, _ in meta], dtype=np.int64),
        "fail_step": fail_step, "onset_index": onset_index, "seg": seg,
        "max_gap_s": np.asarray([[w["max_gap_s"][k] for k in TOPICS] for w, _, _ in meta], dtype=np.float64)
        .reshape(N, len(TOPICS)),
    }
    return arrays


def cmd_define(a):
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (never overwritten)")
    with open(a.quotas) as f:
        quotas = json.load(f)
    pool = build_pool(a.zip, with_signals=True)
    sel, log = select(pool, quotas)
    stratum_of = {g: r["stratum"] for g, r in pool["groups"].items()}
    os.makedirs(a.out)
    ar = Archive(a.zip)
    hashes, sums = {}, {}
    for s in SPLITS:
        arrays = build_split(ar, pool, sel[s], stratum_of)
        p = os.path.join(a.out, f"alfa_{FILE_TAG[s]}.npz")
        write_npz_deterministic(p, arrays)
        hashes[s] = {k: array_sha256(v) for k, v in sorted(arrays.items())}
        sums[os.path.basename(p)] = sha256_file(p)
    write_json(os.path.join(a.out, "POOL.json"), pool)
    write_json(os.path.join(a.out, "SELECTION.json"), {"version": VERSION, "quotas": quotas, "splits": sel, "log": log})
    write_json(os.path.join(a.out, "ARRAY_HASHES.json"), hashes)
    for n in ("POOL.json", "SELECTION.json", "ARRAY_HASHES.json"):
        sums[n] = sha256_file(os.path.join(a.out, n))
    write_json(os.path.join(a.out, "DEF_RECORD.json"), {
        "version": VERSION, "zip": os.path.basename(a.zip), "zip_sha256": pool["zip_sha256"],
        "python": sys.version, "numpy": np.__version__, "obs_names": OBS_NAMES, "act_names": ACT_NAMES,
        "dt_s": DT_NS / 1e9, "stride": STRIDE, "topics": {k: list(v) for k, v in TOPICS.items()},
        "counts": {s: int(len([w for w in pool["windows"] if w["eligible"] and w["group"] in set(sel[s])]))
                   for s in SPLITS}})
    with open(os.path.join(a.out, "SHA256SUMS.txt"), "w") as f:
        for n in sorted(sums):
            f.write(f"{sums[n]}  {n}\n")
    print(json.dumps({s: len(sel[s]) for s in SPLITS}), json.dumps(
        {s: int(np.load(os.path.join(a.out, f"alfa_{FILE_TAG[s]}.npz"))["obs"].shape[0]) for s in SPLITS}))
    return 0


def cmd_hashes(a):
    out = {}
    for s in SPLITS:
        z = np.load(os.path.join(a.def_, f"alfa_{FILE_TAG[s]}.npz"), allow_pickle=False)
        out[s] = {k: array_sha256(z[k]) for k in sorted(z.files)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def cmd_xcheck(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    diffs = []
    for s in SPLITS:
        ka, kb = set(A.get(s, {})), set(B.get(s, {}))
        if ka != kb:
            diffs.append(f"{s}: member sets differ {sorted(ka ^ kb)}")
        for k in sorted(ka & kb):
            if A[s][k] != B[s][k]:
                diffs.append(f"{s}.{k}")
    fd = []
    for n in ("POOL.json", "SELECTION.json") + tuple(f"alfa_{FILE_TAG[s]}.npz" for s in SPLITS):
        ha, hb = sha256_file(os.path.join(a.pa, n)), sha256_file(os.path.join(a.pb, n))
        if ha != hb:
            fd.append(n)
    res = {"version": VERSION, "array_differences": diffs, "file_differences": fd,
           "passed": not diffs and not fd}
    write_json(a.out, res)
    print(json.dumps(res))
    return 0 if res["passed"] else 5


def cmd_audit(a):
    with open(a.quotas) as f:
        quotas = json.load(f)
    hard, desc = {}, {}
    Z = {s: dict(np.load(os.path.join(a.def_, f"alfa_{FILE_TAG[s]}.npz"), allow_pickle=False)) for s in SPLITS}
    sel = json.load(open(os.path.join(a.def_, "SELECTION.json")))
    pool = json.load(open(os.path.join(a.def_, "POOL.json")))
    exp_groups = {s: sum(quotas["quotas"][st][s] for st in quotas["strata"]) for s in SPLITS}
    ok = True
    for s in SPLITS:
        o, ac = Z[s]["obs"], Z[s]["act"]
        ok &= o.dtype == np.float32 and ac.dtype == np.float32 and o.shape[1:] == (T_STATES, 12) \
            and ac.shape[1:] == (T_ACTS, 3) and o.shape[0] == ac.shape[0] and o.shape[0] > 0
    hard["1_shapes"] = bool(ok)
    hard["2_finite"] = bool(all(np.isfinite(Z[s]["obs"]).all() and np.isfinite(Z[s]["act"]).all() for s in SPLITS))
    ok = True
    for s in SPLITS:
        for st in quotas["strata"]:
            n = len(set(Z[s]["parent"][Z[s]["stratum"] == st].tolist()))
            ok &= n == quotas["quotas"][st][s]
        ok &= len(sel["splits"][s]) == exp_groups[s]
    hard["3_quotas"] = bool(ok)
    elig = {(w["sequence"], w["k"]) for w in pool["windows"] if w["eligible"]}
    keys = [(sq, int(k)) for s in SPLITS for sq, k in zip(Z[s]["seq_name"].tolist(), Z[s]["window_k"].tolist())]
    hard["4_eligible_unique"] = bool(all(k in elig for k in keys) and len(keys) == len(set(keys)))
    sel_groups = {g for s in SPLITS for g in sel["splits"][s]}
    want = {(w["sequence"], w["k"]) for w in pool["windows"] if w["eligible"] and w["group"] in sel_groups}
    hard["4b_complete"] = bool(want == set(keys))
    ok = True
    for s in SPLITS:
        fs, seg, oi = Z[s]["fail_step"], Z[s]["seg"], Z[s]["onset_index"]
        for i in range(fs.shape[0]):
            if np.isnan(fs[i]):
                ok &= not seg[i].any() and oi[i] == -1
            else:
                exp = (np.arange(T_STATES) >= fs[i]).astype(np.int8)
                ok &= np.array_equal(seg[i], exp)
                if 0 < fs[i] <= T_STATES - 1:
                    ok &= oi[i] == int(math.ceil(fs[i]))
                else:
                    ok &= oi[i] == -1
    hard["5_labels"] = bool(ok)
    ok = True
    for s in SPLITS:
        o = Z[s]["obs"].astype(np.float64)
        ok &= bool(np.all(np.abs(o[..., 2] ** 2 + o[..., 3] ** 2 - 1) < 1e-5))
        ok &= bool(np.all(np.abs(o[..., 0]) <= 1 + 1e-6) and np.all(np.abs(o[..., 1]) <= 1 + 1e-6))
    hard["6_attitude_ranges"] = bool(ok)
    ok = True
    for s in SPLITS:
        ac = Z[s]["act"].astype(np.float64)
        ok &= bool(np.all(np.abs(ac[..., :2]) <= math.pi / 2) and np.all((ac[..., 2] >= 0) & (ac[..., 2] <= 60)))
    hard["7_action_ranges"] = bool(ok)
    ok = True
    probes = set(quotas["probe_groups"])
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i + 1:]:
            ok &= not (set(Z[s]["parent"].tolist()) & set(Z[t]["parent"].tolist()))
            ok &= not (set(Z[s]["seq_name"].tolist()) & set(Z[t]["seq_name"].tolist()))
            hs = {hashlib.sha256(x.tobytes()).hexdigest() for x in Z[s]["obs"]}
            ht = {hashlib.sha256(x.tobytes()).hexdigest() for x in Z[t]["obs"]}
            ok &= not (hs & ht)
        if s != "train":
            ok &= not (set(Z[s]["parent"].tolist()) & probes)
    hard["8_disjoint"] = bool(ok)
    ar = Archive(a.zip)
    ok = True
    cache = {}
    for s in SPLITS:
        for i, (sq, tau) in enumerate(zip(Z[s]["seq_name"].tolist(), Z[s]["tau0_ns"].tolist())):
            if sq not in cache:
                cache.clear()
                cache[sq] = load_sequence(ar, sq, with_signals=True)
            obs, act, _ = window_arrays(cache[sq], int(tau))
            o = np.asarray(obs, np.float64).astype(np.float32)
            ac = np.asarray(act, np.float64).astype(np.float32)
            ok &= np.array_equal(o, Z[s]["obs"][i]) and np.array_equal(ac, Z[s]["act"][i])
    hard["9_replay"] = bool(ok)
    sd = Z["train"]["obs"].astype(np.float64).reshape(-1, 12).std(0)
    hard["11_train_std"] = bool(np.all(sd >= 1e-3))
    desc["windows"] = {s: int(Z[s]["obs"].shape[0]) for s in SPLITS}
    desc["groups"] = {s: len(set(Z[s]["parent"].tolist())) for s in SPLITS}
    desc["sequences"] = {s: len(set(Z[s]["seq_name"].tolist())) for s in SPLITS}
    desc["family_windows"] = {s: {f: int((Z[s]["sequence"] == f).sum()) for f in sorted(set(Z[s]["sequence"].tolist()))}
                              for s in SPLITS}
    desc["windows_with_onset_inside"] = {s: int((Z[s]["onset_index"] >= 0).sum()) for s in SPLITS}
    desc["windows_post_failure_states"] = {s: int(Z[s]["seg"].any(1).sum()) for s in SPLITS}
    desc["train_obs_std"] = sd.tolist()
    desc["train_act_std"] = Z["train"]["act"].astype(np.float64).reshape(-1, 3).std(0).tolist()
    lo = np.quantile(Z["train"]["obs"].reshape(-1, 12), 0.001, axis=0)
    hi = np.quantile(Z["train"]["obs"].reshape(-1, 12), 0.999, axis=0)
    desc["outside_train_0.1_99.9pct"] = {s: float(np.mean((Z[s]["obs"] < lo) | (Z[s]["obs"] > hi)))
                                         for s in ("test", "test_b")}
    desc["tilt_over_60deg_state_fraction"] = {s: float(np.mean(Z[s]["obs"][..., 0].astype(np.float64) ** 2 +
                                                               Z[s]["obs"][..., 1].astype(np.float64) ** 2 > 0.75))
                                              for s in SPLITS}
    res = {"version": VERSION, "hard": hard, "all_hard_passed": all(hard.values()), "describe": desc,
           "note": "hard check 10 (cross-machine identity) is the separate xcheck step"}
    write_json(a.out, res)
    print(json.dumps({"hard": hard, "all": res["all_hard_passed"]}))
    return 0 if res["all_hard_passed"] else 6


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pool"); p.add_argument("--zip", required=True); p.add_argument("--out", required=True)
    p.add_argument("--with-signals", action="store_true", help="also check finiteness / quaternion validity")
    p = sp.add_parser("define"); p.add_argument("--zip", required=True); p.add_argument("--quotas", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("hashes"); p.add_argument("--def", dest="def_", required=True)
    p = sp.add_parser("xcheck"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--pa", required=True); p.add_argument("--pb", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("audit"); p.add_argument("--def", dest="def_", required=True); p.add_argument("--zip", required=True)
    p.add_argument("--quotas", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"pool": cmd_pool, "define": cmd_define, "hashes": cmd_hashes, "xcheck": cmd_xcheck,
            "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
