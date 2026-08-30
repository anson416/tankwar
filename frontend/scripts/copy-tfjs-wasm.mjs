// copy-tfjs-wasm.mjs — stage the tfjs WASM backend binaries + JS glue into
// public/tfjs-wasm/ before `vite build` so they're served self-hosted from
// the Pages origin (no CDN dependency, works offline once cached). The worker
// calls setWasmPaths() with a path relative to its own chunk, so the wasm
// tier keeps working under the relative GitHub Pages base.
//
// All variants are copied (~1.1 MB total, fetched ONLY if the wasm fallback
// tier is actually reached): plain, simd, and simd-threaded (the threaded
// variant needs COOP/COEP headers GitHub Pages can't set, so in practice the
// simd variant is the ceiling there — shipping it costs nothing if unfetched).
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";

const SRC = "node_modules/@tensorflow/tfjs-backend-wasm/dist";
const DEST = "public/tfjs-wasm";

async function main() {
  if (!existsSync(SRC)) {
    console.error(`FAIL: ${SRC} not found — run "npm install" first.`);
    process.exit(1);
  }
  await mkdir(DEST, { recursive: true });
  const files = (await readdir(SRC)).filter((f) => /^tfjs-backend-wasm.*\.(js|wasm)$/.test(f));
  for (const f of files) await copyFile(join(SRC, f), join(DEST, f));
  console.log(`OK: copied ${files.length} wasm-backend files to ${DEST}/`);
}

main();
