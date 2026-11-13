import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT = "ab971c0ec43df7664c4b727ca69d5b8dd0cab5da"
FILES = (
    "agent/worldModels/MTS3.py",
    "agent/worldModels/SensorEncoders/propEncoder.py",
    "agent/worldModels/Decoders/propDecoder.py",
    "agent/worldModels/gaussianTransformations/gaussian_conditioning.py",
    "agent/worldModels/gaussianTransformations/gaussian_marginalization.py",
)
PACKAGES = (
    "agent",
    "agent/worldModels",
    "agent/worldModels/SensorEncoders",
    "agent/worldModels/Decoders",
    "agent/worldModels/gaussianTransformations",
)
IMPORTS = (
    "SensorEncoders.propEncoder import Encoder",
    "gaussianTransformations.gaussian_marginalization import Predict",
    "gaussianTransformations.gaussian_conditioning import Update",
    "Decoders.propDecoder import SplitDiagGaussianDecoder",
)
NOTE_IMPORTS = "# [patch 1/2] package-relative imports (installed under baselines/third_party/mts3)\n"
NOTE_MASK = (
    "        action_valid_seqs: [patch 2/2] optional (batch x time x 1) bool mask for the\n"
    "            actions that the MANAGER's abstract-action Bayesian aggregation may use. None reproduces\n"
    "            upstream exactly (all actions valid). The worker always receives every action.\n"
)
MASK_BLOCK = (
    "                if action_valid_seqs is None:\n"
    "                    current_act_valid = None ## upstream behaviour: BA with all actions valid\n"
    "                else:\n"
    "                    current_act_valid = action_valid_seqs[:, k+self.H: k+2*self.H, :]\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_sums(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, name = line.split(None, 1)
                out[name.strip()] = digest
    return out


def edit(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"apply_patch: anchor not found exactly once: {old[:70]!r}")
    return text.replace(old, new)


def patch_mts3(text: str, note_imports: str = NOTE_IMPORTS, note_mask: str = NOTE_MASK) -> str:
    for imp in IMPORTS:
        text = edit(text, f"from agent.worldModels.{imp}\n", f"from .{imp}\n")
    text = edit(text, "import torch\nfrom .SensorEncoders", "import torch\n" + note_imports + "from .SensorEncoders")
    text = edit(text, "    def forward(self, obs_seqs, action_seqs, obs_valid_seqs):\n",
                "    def forward(self, obs_seqs, action_seqs, obs_valid_seqs, action_valid_seqs=None):\n")
    anchor = "        obs_valid_seqs: sequences of timeseries of actions (batch x time)\n"
    text = edit(text, anchor, anchor + note_mask)
    anchor = "                abs_act_mean, abs_act_var = self._action_Infer(skill_prior_mean, skill_prior_cov, alpha_k_mean, \\\n"
    text = edit(text, anchor, MASK_BLOCK + anchor)
    text = edit(text, "alpha_k_var, None) ##BA with all actions valid\n",
                "alpha_k_var, current_act_valid) ##BA over valid actions only\n")
    return text


def build(upstream: str, sums: dict, note_imports: str = NOTE_IMPORTS, note_mask: str = NOTE_MASK) -> dict:
    out = {}
    for rel in FILES:
        with open(os.path.join(upstream, rel), "rb") as f:
            raw = f.read()
        got = sha256_bytes(raw)
        want = sums.get("upstream/" + rel)
        if got != want:
            raise SystemExit(f"apply_patch: upstream {rel} has sha256 {got}, expected {want} (commit {COMMIT})")
        if rel.endswith("/MTS3.py"):
            raw = patch_mts3(raw.decode("utf-8"), note_imports, note_mask).encode("utf-8")
        out[rel] = raw
    return out


def install(files: dict, dest: str, sums: dict) -> None:
    for rel, raw in files.items():
        want = sums.get(rel)
        got = sha256_bytes(raw)
        if got != want:
            raise SystemExit(f"apply_patch: patched {rel} has sha256 {got}, expected {want}")
    for pkg in PACKAGES:
        os.makedirs(os.path.join(dest, pkg), exist_ok=True)
        init = os.path.join(dest, pkg, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
    for rel, raw in files.items():
        with open(os.path.join(dest, rel), "wb") as f:
            f.write(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install the five MTS3 model modules used by baselines/mts3.py "
                                             "from an upstream checkout, with the two-part edit, and verify sha256.")
    ap.add_argument("upstream", help="checkout of the upstream MTS3 repository at commit " + COMMIT)
    ap.add_argument("--dest", default=HERE, help="install folder (default: this folder)")
    ap.add_argument("--sums", default=os.path.join(HERE, "SHA256SUMS"))
    args = ap.parse_args(argv)
    sums = read_sums(args.sums)
    files = build(args.upstream, sums)
    install(files, args.dest, sums)
    print(f"installed {len(files)} files under {os.path.abspath(args.dest)} (sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
