import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

import wasm from "vite-plugin-wasm";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    tailwindcss(),
    wasm(),
  ],
  worker: {
    format: 'es',
    plugins: () => [
      wasm(),
    ],
  },
  // Vendored emscripten encoder ships its own .wasm next to the JS glue;
  // we need to keep it out of Vite's depopt and let it serve the wasm asset.
  assetsInclude: ['**/avif_enc.wasm'],
  build: {
    // We require browsers with WebGPU, Float16Array, and top-level await
    // — es2022 is well within that envelope and lets vite skip the
    // destructuring lowering that tripped up the vendored emscripten glue.
    target: 'es2022',
  },
  resolve: {
    alias: {
      'frontend-rs': path.resolve(__dirname, '../frontend-rs/pkg'),
      '@': path.resolve(__dirname, './src'),
    },
  },
})
