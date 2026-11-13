#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=ab971c0ec43df7664c4b727ca69d5b8dd0cab5da
PY="${PYTHON:-python3}"
if [ "$#" -ge 1 ]; then
  "$PY" -I "$HERE/apply_patch.py" "$1"
  exit 0
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
if command -v git >/dev/null 2>&1; then
  git clone --quiet https://github.com/ALRhub/MTS3.git "$WORK/MTS3"
  git -C "$WORK/MTS3" -c advice.detachedHead=false checkout --quiet "$COMMIT"
else
  mkdir -p "$WORK/MTS3"
  curl -fsSL "https://codeload.github.com/ALRhub/MTS3/tar.gz/$COMMIT" | tar xz -C "$WORK/MTS3" --strip-components 1
fi
"$PY" -I "$HERE/apply_patch.py" "$WORK/MTS3"
