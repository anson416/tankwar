import { defineConfig } from "vite";
import obfuscator from "vite-plugin-javascript-obfuscator";

// Relative base so the built site works under any GitHub Pages sub-path
// (https://<user>.github.io/<repo>/) without hard-coding the repo name.
export default defineConfig({
  base: "./",

  // ES-module workers: required for the lazy dynamic backend imports
  // (webgpu/webgl/wasm) inside train.worker.ts.
  worker: { format: "es" },

  build: {
    target: "es2022",
    sourcemap: false,
  },

  plugins: [
    // Moderate obfuscation: string-array encoding + identifier renaming —
    // deliberately WITHOUT control-flow flattening, numbers-to-expressions or
    // self-defending, which would slow the training/inference hot loops.
    // The plugin's default exclude already skips ALL node_modules module ids,
    // so the entire @tensorflow/* graph is never obfuscated (verified against
    // vite-plugin-javascript-obfuscator@3.1.0 source) — only our src/ is.
    // Build-only: the dev server and vitest stay readable and fast.
    obfuscator({
      apply: "build",
      options: {
        compact: true,
        simplify: true,
        identifierNamesGenerator: "hexadecimal",
        renameGlobals: false,
        stringArray: true,
        stringArrayEncoding: ["base64"],
        stringArrayThreshold: 0.75,
        stringArrayRotate: true,
        stringArrayWrappersCount: 2,
        unicodeEscapeSequence: true,
        // Vite emits build-time placeholders like __VITE_WORKER_ASSET__<hash>__
        // (and __VITE_ASSET__ / __VITE_PUBLIC_ASSET__) as plain string literals,
        // then does a literal text replace AFTER this plugin runs. Without
        // reserving them, stringArray/base64 encoding moves them into the
        // decoder array, the literal replace finds nothing, and the raw
        // placeholder ships — so `new Worker(new URL(...))` 404s in prod.
        // Dev is unaffected (obfuscator is build-only).
        reservedStrings: [
          "__VITE_WORKER_ASSET__",
          "__VITE_ASSET__",
          "__VITE_PUBLIC_ASSET__",
        ],
        transformObjectKeys: true,
        deadCodeInjection: false,
        // Explicitly off — these are the transforms that hurt hot-loop perf.
        controlFlowFlattening: false,
        numbersToExpressions: false,
        splitStrings: false,
        selfDefending: false,
        debugProtection: false,
      },
    }),
  ],
});
