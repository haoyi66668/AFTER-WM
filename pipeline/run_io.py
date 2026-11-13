from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo", ".lock", ".tmp", ".swp")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def array_sha256(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return to_jsonable(x.detach().cpu().numpy())
    if isinstance(x, (str, type(None))):
        return x
    return str(x)


def _fsync_dir(d: str) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(d)


def atomic_json(path: str, obj, indent: int = 1) -> None:
    text = json.dumps(to_jsonable(obj), indent=indent, sort_keys=False, allow_nan=False, ensure_ascii=False)
    atomic_write_bytes(path, (text + "\n").encode())


def atomic_npz(path: str, compressed: bool = False, **arrays) -> None:
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def atomic_torch_save(path: str, obj) -> None:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def torch_load(path: str, map_location="cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def append_jsonl(path: str, obj) -> None:
    line = json.dumps(to_jsonable(obj), sort_keys=False, allow_nan=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        out.append({"_unparseable": line[:200]})
    except FileNotFoundError:
        pass
    return out


def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def next_free_path(path_pattern: str, start: int = 1) -> tuple:
    n = start
    while os.path.exists(path_pattern.format(n=n)):
        n += 1
    return path_pattern.format(n=n), n


def move_aside(path: str, tag: str = "prev") -> str | None:
    if not os.path.exists(path):
        return None
    dst, _ = next_free_path(path + "." + tag + "_{n}")
    os.rename(path, dst)
    return dst


def code_manifest(root: str = CODE_ROOT) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            if fn.endswith(_SKIP_SUFFIXES) or fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            if os.path.isfile(p) and not os.path.islink(p):
                out[os.path.relpath(p, root)] = sha256_file(p)
    return dict(sorted(out.items()))


def imported_code_files(root: str = CODE_ROOT) -> dict:
    root = os.path.abspath(root)
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        if f.endswith((".pyc", ".pyo")):
            f = f[:-1]
        if f.startswith(root + os.sep) and os.path.isfile(f):
            out[os.path.relpath(f, root)] = sha256_file(f)
    return dict(sorted(out.items()))


def module_source_path(module) -> str:
    return os.path.abspath(module.__file__)


def _cmd(args, timeout=10):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def cpu_quota_cores():
    for qf, pf in (("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),):
        try:
            q = int(open(qf).read().strip())
            p = int(open(pf).read().strip())
            if q > 0 and p > 0:
                return q / p
        except Exception:
            pass
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return int(q) / int(p)
    except Exception:
        pass
    return None


def env_info(device: str | None = None) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pid": os.getpid(),
        "cpu_quota_cores": cpu_quota_cores(),
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                                                         "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")},
        "time": now_iso(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
            idx = torch.cuda.current_device()
            info["gpu"] = torch.cuda.get_device_name(idx)
            info["gpu_total_mib"] = round(torch.cuda.get_device_properties(idx).total_memory / 2 ** 20)
        info["device"] = device
    except Exception as exc:
        info["torch"] = f"unavailable: {exc!r}"
    return info


def gpu_driver() -> str:
    return _cmd(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])


def peak_rss_mib() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


class LockBusy(RuntimeError):
    pass


class FileLock:

    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def acquire(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise LockBusy(f"lock busy: {self.path}") from exc
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {socket.gethostname()} {now_iso()}\n".encode())
        self.fd = fd
        return self

    def release(self):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def json_exactness_problems(x, path="cfg") -> list:
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                out.append(f"{path}: non-string key {k!r}")
            out += json_exactness_problems(v, f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out += json_exactness_problems(v, f"{path}[{i}]")
    elif isinstance(x, bool) or x is None or isinstance(x, (str, int)):
        pass
    elif isinstance(x, float):
        if not math.isfinite(x):
            out.append(f"{path}: non-finite float {x!r}")
    else:
        out.append(f"{path}: non-JSON type {type(x).__name__}")
    return out


def parse_variant(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"variant must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out
