#!/usr/bin/env bash
# Self-host @imgly/background-removal model + ORT runtime under /public/imgly
# so bg removal doesn't depend on staticimgly.com at runtime.
#
# Idempotent: skips the download when public/imgly/.version matches the
# installed package version. Prunes everything except the SDK's default
# isnet_fp16 weights — the runtime call site does not pass a `model`
# option, so any other variant would fetch-fail at runtime.

set -euo pipefail

cd "$(dirname "$0")/.."

PUBLIC_DIR="public/imgly"
MARKER="$PUBLIC_DIR/.version"
KEEP_MODEL="isnet_fp16"

PKG_VERSION="$(node -p "require('./node_modules/@imgly/background-removal/package.json').version")"
TGZ_URL="https://staticimgly.com/@imgly/background-removal-data/${PKG_VERSION}/package.tgz"

if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "${PKG_VERSION}:${KEEP_MODEL}" ]; then
  echo "imgly bg-removal assets already at v${PKG_VERSION} (${KEEP_MODEL})"
  exit 0
fi

echo "Fetching imgly bg-removal v${PKG_VERSION} (model: ${KEEP_MODEL})..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/pkg.tgz" "$TGZ_URL"
tar -xzf "$TMP/pkg.tgz" -C "$TMP"

# Hashes the SDK will request: the ORT runtime files and one model variant.
# We keep resources.json intact (the SDK reads it and picks the configured
# model) but delete chunks for the model variants we're not shipping.
KEEP_HASHES="$(node -e "
  const r = require('$TMP/package/dist/resources.json');
  const keep = new Set();
  for (const [path, info] of Object.entries(r)) {
    const isRuntime = path.startsWith('/onnxruntime-web/');
    const isKeptModel = path === '/models/' + '$KEEP_MODEL';
    if (!isRuntime && !isKeptModel) continue;
    for (const c of info.chunks) keep.add(c.hash);
  }
  process.stdout.write([...keep].join('\n'));
")"

rm -rf "$PUBLIC_DIR"
mkdir -p "$PUBLIC_DIR"
cp "$TMP/package/dist/resources.json" "$PUBLIC_DIR/"
while IFS= read -r hash; do
  cp "$TMP/package/dist/$hash" "$PUBLIC_DIR/$hash"
done <<< "$KEEP_HASHES"

echo "${PKG_VERSION}:${KEEP_MODEL}" > "$MARKER"
echo "Wrote $(find "$PUBLIC_DIR" -type f | wc -l | tr -d ' ') files to $PUBLIC_DIR"
