// check-build-artifacts.mjs — fast, browser-free post-build guard that runs
// after `vite build`. Two assertions; either failing exits non-zero so the
// GitHub Pages deploy is gated. This exists because obfuscation can mangle the
// __VITE_WORKER_ASSET__ string-replacement Vite relies on to rewrite
// `new Worker(new URL(...))` into the real chunk filename, causing a 404.
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";

const distDir = "dist";
const PLACEHOLDERS = [
  "__VITE_WORKER_ASSET__",
  "__VITE_ASSET__",
  "__VITE_PUBLIC_ASSET__",
];
const TEXT_EXT = new Set([".js", ".mjs", ".cjs", ".html", ".css"]);
const WORKER_RE = /^train\.worker-.*\.js$/;
const INDEX_RE = /^index-.*\.js$/;

async function listTextBundles(dir) {
  const all = await listFiles(dir);
  return all.filter((p) => {
    const name = p.split("/").pop() ?? "";
    return TEXT_EXT.has(name.slice(name.lastIndexOf(".")));
  });
}

/** Recursively list every file (full path) under `dir`. */
async function listFiles(dir) {
  const out = [];
  for (const name of await readdir(dir)) {
    const path = join(dir, name);
    const s = await stat(path);
    if (s.isDirectory()) {
      out.push(...(await listFiles(path)));
    } else {
      out.push(path);
    }
  }
  return out;
}

async function main() {
  try {
    await stat(distDir);
  } catch {
    console.error(`FAIL: ${distDir}/ does not exist — run "vite build" first.`);
    process.exit(1);
  }

  const bundles = await listTextBundles(distDir);
  const srcs = await Promise.all(bundles.map((p) => readFile(p, "utf8")));
  const allText = srcs.join("\n");

  // (a) No leaked Vite placeholders.
  for (const ph of PLACEHOLDERS) {
    if (allText.includes(ph)) {
      console.error(`FAIL: leaked Vite placeholder "${ph}" found in dist text bundles.`);
      process.exit(1);
    }
  }

  // (b) Exactly one train.worker chunk AND an index chunk literally references it.
  // Vite emits chunks under dist/assets/ (recursively scan the whole dist tree
  // so this works whether or not the assets subdir is used).
  const allFiles = await listFiles(distDir);
  const workerChunks = allFiles
    .map((p) => p.split("/").pop())
    .filter((f) => !!f && WORKER_RE.test(f));
  const indexChunks = allFiles
    .map((p) => p.split("/").pop())
    .filter((f) => !!f && INDEX_RE.test(f));
  if (workerChunks.length !== 1) {
    console.error(`FAIL: expected exactly one train.worker-*.js chunk, got ${workerChunks.length}.`);
    process.exit(1);
  }
  if (indexChunks.length < 1) {
    console.error("FAIL: no index-*.js chunk found.");
    process.exit(1);
  }
  const workerName = workerChunks[0];
  // Confirm some index chunk text references the worker filename. Gather the
  // text of every index chunk (search by content, not just name).
  const indexTexts = await Promise.all(
    allFiles
      .filter((p) => INDEX_RE.test(p.split("/").pop() ?? ""))
      .map((p) => readFile(p, "utf8")),
  );
  const indexOk = indexTexts.some((t) => t.includes(workerName));

  if (!indexOk) {
    console.error(`FAIL: index chunk does not reference worker chunk "${workerName}".`);
    process.exit(1);
  }

  console.log(
    `OK: no leaked placeholders; worker chunk "${workerName}" referenced by index chunk.`,
  );
}

main();
