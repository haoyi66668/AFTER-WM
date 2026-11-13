from __future__ import annotations

import argparse
import copy
import os

CONSTRUCTOR_STREAM = "tsd"
STAGE1_CHECKPOINT_FORMAT = "tsd.lib.experts.stage1a_checkpoint"
STAGE1_SOURCE = "tsd.stage1"
LEGACY_PACKAGE = "tsd"
PACKAGE = "afterwm"
LEGACY_ARCHITECTURE = "neural_LF_TSD"
ARCHITECTURE = "neural_LF_AFTERWM"
ID_FIELDS = ("model_id", "budget_class", "name", "family")


def identifier(value):
    if isinstance(value, str) and (value == LEGACY_PACKAGE or value.startswith(LEGACY_PACKAGE + "_")):
        return PACKAGE + value[len(LEGACY_PACKAGE):]
    return value


def module_name(value):
    if isinstance(value, str) and value.startswith(LEGACY_PACKAGE + "."):
        return PACKAGE + value[len(LEGACY_PACKAGE):]
    return value


def architecture(value):
    if isinstance(value, str) and value.startswith(LEGACY_ARCHITECTURE):
        return ARCHITECTURE + value[len(LEGACY_ARCHITECTURE):]
    return value


def upgrade_protocol(prot):
    out = copy.deepcopy(prot)
    for block in (out, out.get("cfg")):
        if not isinstance(block, dict):
            continue
        if "module" in block:
            block["module"] = module_name(block["module"])
        for key in ID_FIELDS:
            if key in block:
                block[key] = identifier(block[key])
        if "architecture" in block:
            block["architecture"] = architecture(block["architecture"])
    return out


def link_archive(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    made = []
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        parts = [] if rel == "." else rel.split(os.sep)
        target = os.path.join(dst, *[identifier(p) for p in parts])
        os.makedirs(target, exist_ok=True)
        for name in sorted(filenames):
            link = os.path.join(target, name)
            if not os.path.lexists(link):
                os.symlink(os.path.join(dirpath, name), link)
                made.append(link)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a runs root with the current folder names from a runs root "
                                             "that uses the legacy folder names (symbolic links, no file is changed).")
    ap.add_argument("command", choices=("link-archive",))
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args(argv)
    print(len(link_archive(a.src, a.dst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
