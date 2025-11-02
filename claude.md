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
