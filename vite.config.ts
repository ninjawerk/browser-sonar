import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the build works on GitHub Pages project sites.
  base: './',
  worker: {
    // IIFE keeps the AudioWorklet bundle self-contained (no runtime imports),
    // which is what AudioWorkletGlobalScope needs across browsers.
    format: 'iife',
  },
  build: {
    target: 'es2020',
  },
});
