#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/processed.zip}"
URL="https://ndownloader.figshare.com/files/24095870"
SHA256="02562756b9428aa24fbd27105f91687b38541ced8d5432766c14176573d0b4c9"
if [ ! -f "$OUT" ]; then
  curl -L --fail -o "$OUT.part" "$URL"
  mv "$OUT.part" "$OUT"
fi
if command -v sha256sum >/dev/null 2>&1; then
  GOT="$(sha256sum "$OUT" | cut -d' ' -f1)"
else
  GOT="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
fi
if [ "$GOT" != "$SHA256" ]; then
  echo "sha256 mismatch for $OUT: $GOT" >&2
  exit 1
fi
echo "OK $OUT"
