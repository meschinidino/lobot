#!/bin/sh
# Build the Space folder for the read-only public demo.
# Usage: deploy/space/build.sh <out-dir> [commit]   (run from the repo root)
# The jobs come from LOBOT_JOBS_DIR, else ./jobs. Local paths in them are
# rewritten to /app, where the image keeps the code.
set -eu
out=${1:?usage: deploy/space/build.sh <out-dir> [commit]}
commit=${2:-HEAD}
jobs=${LOBOT_JOBS_DIR:-jobs}
root=$(cd "$(dirname "$jobs")" && pwd)

rm -rf "$out" && mkdir -p "$out/src"
cp deploy/space/Dockerfile deploy/space/README.md "$out/"
git archive "$commit" | tar -x -C "$out/src"
cp -R "$jobs" "$out/jobs"
grep -rlI "$root" "$out/jobs" | while read -r f; do
  sed -i.bak "s#$root#/app#g" "$f" && rm "$f.bak"
done
if grep -rI "/Users/\|/home/" "$out/jobs" >/dev/null; then
  echo "build.sh: a local path is still in $out/jobs" >&2; exit 1
fi
# git archive takes tracked files only, so .env never enters src/.
if [ -e "$out/src/.env" ]; then
  echo "build.sh: .env is in $out/src" >&2; exit 1
fi
key=$(grep -E '^META_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d "\"'" || true)
if [ -n "$key" ] && grep -rIqF "$key" "$out"; then
  echo "build.sh: the META_API_KEY value is in $out" >&2; exit 1
fi
echo "Space folder ready: $out (code from $(git rev-parse --short "$commit"))"
