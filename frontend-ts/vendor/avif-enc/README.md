# Vendored AVIF encoder

Forked from [jSquash](https://github.com/jamsinclair/jSquash) `packages/avif/codec/`
to expose libavif's CICP fields (color primaries / transfer characteristics /
matrix coefficients / range) so we can write nclx-tagged HDR AVIFs (BT.2020 + PQ).

The patches live in `enc/avif_enc.cpp` — search for `cicp` to find them.

## Built artifacts

`enc/avif_enc.{js,wasm}` are checked into git. Vite picks them up via the import
in `frontend-ts/src/lib/avif-hdr.ts`. CI does **not** rebuild them — only commit
new artifacts when the C++ binding changes.

## Rebuilding (only when patching avif_enc.cpp)

Requires [emsdk](https://github.com/emscripten-core/emsdk) installed at
`~/emsdk` and CMake on `$PATH`. From this directory:

```sh
source ~/emsdk/emsdk_env.sh
export PATH="$EMSDK/upstream/emscripten:$PATH"
export CFLAGS="-Oz -flto"
export CXXFLAGS="$CFLAGS -std=c++17"
export LDFLAGS="$CFLAGS -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1"
touch enc/avif_enc.cpp   # force rebuild
emmake make enc/avif_enc.js
```

The first build downloads + compiles libaom and libavif from source (~5 min).
Subsequent builds reuse the cached libs and only relink (~10 sec).

The multithreaded variant (`avif_enc_mt.{js,wasm}`) is not currently used —
multi-thread emscripten needs SharedArrayBuffer which requires COOP/COEP
response headers we don't set.
