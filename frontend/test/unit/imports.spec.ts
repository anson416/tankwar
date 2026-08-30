// imports.spec.ts — the bundle-boundary gate (plan §Architecture: "main bundle
// imports zero tfjs — enforced by a unit test grep"). One worker owns every
// tensor; the main thread renders snapshots and ships input messages, so the
// UI bundle must not pull a single @tensorflow/* module (weights never cross
// postMessage, and the main bundle stays two orders of magnitude smaller than
// the kernel stack).
//
// The scan is a WHITELIST, not a list of scanned directories: every .ts file
// under src/** is walked and tfjs is allowed in EXACTLY workers/train.worker.ts
// and src/nn/**. A per-directory blocklist rots — a new src/train-protocol.ts,
// a new top-level module, a "harmless" helper import from src/nn in main.ts
// would all be invisible to it, and any of them drags the whole @tensorflow
// graph (or the net builders) into index-*.js with every gate still green.
// src/nn/** itself must NOT import tfjs from anywhere but the pinned packages
// either way; the worker is additionally asserted to STILL reference tfjs: a
// build that accidentally dropped the worker's tfjs imports would silently
// unlearn the whole app (every act/train would throw at runtime).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = new URL("../../src", import.meta.url).pathname;
/** The only module outside src/nn allowed to import tfjs (the worker chunk). */
const WORKER = join(SRC, "workers", "train.worker.ts");
/** Path-separator string (join() output is absolute and platform-separated). */
const SEP = "/";
/**
 * tfjs in an IMPORT position — static, side-effect or dynamic — not the bare
 * substring: the boundary docs ("imports no @tensorflow/*") live in exactly
 * the comments a naive grep would trip over.
 */
const TFJS_IMPORT = /(?:from\s+|import\s*\(\s*|import\s+)["']@tensorflow\//;
/**
 * A value import from src/nn in MAIN-side code is boundary-breaking too, even
 * though nn/** is itself tfjs-gated: importing the module graph pulls the
 * whole tfjs dependency tree into the main chunk (plus couples the UI to the
 * learner's internals). Matched in import position only — the word "nn/" in a
 * comment or a path string must not trip it.
 */
const NN_IMPORT = /(?:from\s+|import\s*\(\s*|import\s+)["'](?:\.\.?\/)*nn\//;

/** Every .ts file under `dir`, recursively. */
function tsFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...tsFiles(p));
    else if (name.endsWith(".ts")) out.push(p);
  }
  return out;
}

describe("tfjs bundle boundary (plan §Architecture)", () => {
  it("@tensorflow/* appears ONLY in workers/train.worker.ts and src/nn/**", () => {
    const offenders = tsFiles(SRC).filter((f) => {
      if (f === WORKER) return false; // the one tfjs-owning chunk by design
      if (f.startsWith(join(SRC, "nn") + SEP)) return false; // the learner (worker-side only)
      return TFJS_IMPORT.test(readFileSync(f, "utf8"));
    });
    expect(offenders.map((f) => f.slice(SRC.length + 1))).toEqual([]);
  });

  it("main.ts, src/app/** and src/train-protocol.ts import nothing from src/nn", () => {
    const mainSide = [join(SRC, "main.ts"), ...tsFiles(join(SRC, "app")), join(SRC, "train-protocol.ts")];
    const offenders = mainSide.filter((f) => NN_IMPORT.test(readFileSync(f, "utf8")));
    expect(offenders.map((f) => f.slice(SRC.length + 1))).toEqual([]);
  });

  it("the train worker is the tfjs side of the boundary", () => {
    expect(TFJS_IMPORT.test(readFileSync(WORKER, "utf8"))).toBe(true);
  });
});
