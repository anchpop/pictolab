import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

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
    topLevelAwait(),
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
  resolve: {
    alias: {
      'frontend-rs': path.resolve(__dirname, '../frontend-rs/pkg'),
      '@': path.resolve(__dirname, './src'),
    },
  },
})
