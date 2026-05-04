# Pictolab Development Notes

## Building the Rust WASM Module

To build the frontend-rs WASM module:

```bash
cd frontend-rs
wasm-pack build
```

Note: Don't use `--target web`, just `wasm-pack build` is sufficient.

## Package Manager

Use pnpm for installing packages in frontend-ts, not npm.

## Background Removal Assets

`@imgly/background-removal` model + ORT runtime are self-hosted under
`frontend-ts/public/imgly/` instead of fetched from staticimgly.com.
The folder is gitignored; `pnpm predev` and `pnpm prebuild` invoke
`scripts/fetch-bg-removal-assets.sh`, which is idempotent and prunes
to the SDK's default `isnet_fp16` model.
