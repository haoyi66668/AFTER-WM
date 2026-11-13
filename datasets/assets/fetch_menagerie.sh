#!/usr/bin/env bash
set -euo pipefail
DS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="6da8b971844845de6ab27315aacc3a3fea4ddb8e"
SRC="${1:-}"
if [ -z "$SRC" ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git -C "$TMP" init -q menagerie
  SRC="$TMP/menagerie"
  git -C "$SRC" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
  git -C "$SRC" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
  git -C "$SRC" sparse-checkout set --no-cone /franka_emika_panda/ /universal_robots_ur5e/ /LICENSE
  git -C "$SRC" checkout -q FETCH_HEAD
fi
check() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c --quiet SHA256SUMS.txt
  else
    shasum -a 256 -c --quiet SHA256SUMS.txt
  fi
}
install_robot() {
  local robot="$1" target="$2" name="$3"
  local dest="$DS/$target/mujoco_menagerie_6da8b971"
  if [ -e "$dest" ]; then
    echo "$dest exists" >&2
    exit 1
  fi
  mkdir -p "$dest"
  cp -R "$SRC/$robot" "$dest/$robot"
  cp "$SRC/LICENSE" "$dest/LICENSE_repo"
  cp "$DS/assets/GIT_LS_TREE_$robot.txt" "$dest/GIT_LS_TREE_$robot.txt"
  cp "$DS/assets/menagerie_$name.sha256" "$dest/SHA256SUMS.txt"
  (cd "$dest" && check)
  echo "OK $dest"
}
install_robot franka_emika_panda panda_model panda
install_robot universal_robots_ur5e ur5e_model ur5e
