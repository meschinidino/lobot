#!/bin/sh
# Build the pinned llama.cpp toolchain + lobot-llamad (docs/plans/llama-backend.md).
# Clones ggml-org/llama.cpp at scripts/llamacpp.lock into third_party/llama.cpp,
# builds llama-quantize there, then builds native/lobot-llamad against it.
# Idempotent: re-runs check out the pinned sha and rebuild.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOCK="$ROOT/scripts/llamacpp.lock"
SRC="$ROOT/third_party/llama.cpp"
BUILD="$SRC/build"
SHA=$(tr -d '[:space:]' < "$LOCK")
case "$SHA" in
  ''|*[!0-9a-f]*)
    echo "bad sha in $LOCK" >&2; exit 1 ;;
esac
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  mkdir -p "$ROOT/third_party"
  attempt=0
  until git clone https://github.com/ggml-org/llama.cpp.git "$SRC"; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 3 ]; then
      echo "clone failed 3 times; aborting" >&2; exit 1
    fi
    echo "[build-llamacpp] clone failed, retry $attempt/3" >&2
    rm -rf "$SRC"
    sleep 10
  done
fi
git -C "$SRC" fetch --quiet origin "$SHA"
git -C "$SRC" checkout --quiet "$SHA"
echo "[build-llamacpp] $SHA checked out" >&2
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build "$BUILD" -j --target llama-quantize
echo "[build-llamacpp] llama-quantize built" >&2
DBUILD="$ROOT/third_party/lobot-llamad-build"
cmake -S "$ROOT/native/lobot-llamad" -B "$DBUILD" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMACPP_SRC="$SRC" \
  -DLLAMACPP_COMMIT="$SHA"
cmake --build "$DBUILD" -j
echo "[build-llamacpp] $DBUILD/lobot-llamad built" >&2
