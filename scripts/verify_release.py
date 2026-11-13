#!/usr/bin/env python3
"""Check the distributed data/results and the published main-table values."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def verify_sums(manifest, base):
    count = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = base / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Checksum mismatch: {path.relative_to(ROOT)}")
        count += 1
    return count


def table_rows(text):
    return [re.findall(r"\d+\.\d+", line) for line in text.splitlines()
            if "$\\pm$" in line and "&" in line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    counts = {
        "data_files": verify_sums(ROOT / "datasets/SHA256SUMS", ROOT),
        "stored_output_files": verify_sums(ROOT / "outputs/SHA256SUMS", ROOT / "outputs"),
    }
    if args.results:
        expected = json.loads((ROOT / "docs/reference_table1.json").read_text())
        actual = table_rows((args.results / "tables/table1.tex").read_text())
        if actual != expected["rows"]:
            raise SystemExit("Generated Table 1 differs from the published table.")
        counts["published_table_values"] = sum(map(len, actual))
        numbers = json.loads((args.results / "paper_numbers.json").read_text())
        for key, value in expected["key_results"].items():
            if numbers.get(key) != value:
                raise SystemExit(f"Reported result differs: {key}")
        counts["key_results"] = len(expected["key_results"])
    print(json.dumps({"passed": True, **counts}, indent=2))


if __name__ == "__main__":
    main()
