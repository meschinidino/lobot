#!/bin/sh
# Build the static Space: web/ plus static-api.js plus exported answers.
# Usage (repo root): deploy/static/build.sh <out-dir>
# The jobs come from LOBOT_JOBS_DIR, else ./jobs.
set -eu
out=${1:?usage: deploy/static/build.sh <out-dir>}
rm -rf "$out" && mkdir -p "$out"
git archive HEAD web | tar -x -C "$out" --strip-components 1
cp deploy/static/static-api.js "$out/"
cp deploy/static/README.md "$out/README.md"
# static-api.js must patch fetch before js/app.js runs.
sed -i.bak 's#<script type="module" src="js/app.js"></script>#<script src="static-api.js"></script>\n<script type="module" src="js/app.js"></script>#' "$out/index.html"
rm "$out/index.html.bak"
grep -q 'src="static-api.js"' "$out/index.html" || { echo "build.sh: script tag not added" >&2; exit 1; }
env -u META_API_KEY uv run python deploy/static/export.py "$out"
if grep -rIq "/Users/\|/home/" "$out/static-api"; then
  echo "build.sh: a local path is in the exported answers" >&2; exit 1
fi
key=$(grep -E '^META_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d "\"'" || true)
if [ -n "$key" ] && grep -rIqF "$key" "$out"; then
  echo "build.sh: the META_API_KEY value is in $out" >&2; exit 1
fi
echo "Static Space ready: $out"
