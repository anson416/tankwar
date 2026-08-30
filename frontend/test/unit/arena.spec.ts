// arena.spec.ts — the 500-seed property battery for generated arenas. These
// are the SPEC-level invariants every consumer leans on: the shared network is
// only fair if boards are 180°-symmetric, the renderer only draws honest boxes
// if every obstacle is one CELL block, training only terminates if pockets
// cannot seal, and replays only reproduce if generation is a pure function of
// its seed. Checks here are deliberately re-implemented from the layout's
// public fields (not via arena.ts internals) so a broken invariant cannot hide
// behind the same helper that enforces it.
import { describe, expect, it } from "vitest";
import {
  CELL,
  GRID_COLS,
  GRID_ROWS,
  SAFE_LAYOUT,
  floodFill,
  generateArena,
  type ArenaLayout,
} from "../../src/game/arena";
import { ARENA_H, ARENA_W, CRATE_HP } from "../../src/game/constants";
import { mulberry32 } from "./util/rng";

/** The plan's four quarter points, in tank-id order — spawns must be exactly
 *  these, not "somewhere clear": mirrored pairs (A↔D, B↔C) are what makes any
 *  tankCount slice of the list fair. */
const QUARTER: [number, number][] = [
  [75, 87.5],
  [825, 87.5],
  [75, 612.5],
  [825, 612.5],
];
const CENTER = { x: ARENA_W / 2, y: ARENA_H / 2 };

// --- independent reimplementations of the invariants -------------------------

function cellIndex(x: number, y: number): number {
  return Math.floor(y / CELL) * GRID_COLS + Math.floor(x / CELL);
}

/** Occupancy grid rebuilt from the layout's public AABBs. */
function occupancy(layout: ArenaLayout): Uint8Array {
  const occ = new Uint8Array(GRID_COLS * GRID_ROWS);
  for (const o of [...layout.walls, ...layout.crates]) {
    occ[cellIndex(o.x, o.y)] = 1;
  }
  return occ;
}

/** True iff every spawn's cell is in spawn 0's 4-connected free component. */
function allSpawnsReachable(layout: ArenaLayout): boolean {
  const occ = occupancy(layout);
  const start = cellIndex(layout.spawns[0].x, layout.spawns[0].y);
  if (occ[start] === 1) return false;
  const seen = new Uint8Array(GRID_COLS * GRID_ROWS);
  const queue = [start];
  seen[start] = 1;
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    const c = i % GRID_COLS;
    const r = (i / GRID_COLS) | 0;
    const push = (j: number): void => {
      if (seen[j] === 0 && occ[j] === 0) {
        seen[j] = 1;
        queue.push(j);
      }
    };
    if (c > 0) push(i - 1);
    if (c < GRID_COLS - 1) push(i + 1);
    if (r > 0) push(i - GRID_COLS);
    if (r < GRID_ROWS - 1) push(i + GRID_COLS);
  }
  return layout.spawns.every((sp) => seen[cellIndex(sp.x, sp.y)] === 1);
}

/** Min distance from any spawn to any obstacle AABB — the ≥100 u clearance. */
function minClearance(layout: ArenaLayout): number {
  let min = Number.POSITIVE_INFINITY;
  for (const sp of layout.spawns) {
    for (const o of [...layout.walls, ...layout.crates]) {
      const dx = Math.max(o.x - sp.x, 0, sp.x - (o.x + o.w));
      const dy = Math.max(o.y - sp.y, 0, sp.y - (o.y + o.h));
      min = Math.min(min, Math.hypot(dx, dy));
    }
  }
  return min;
}

/** True iff no obstacle sits on a spawn cell, its 4-neighbours included. */
function spawnCellsClear(layout: ArenaLayout): boolean {
  for (const sp of layout.spawns) {
    const sc = Math.floor(sp.x / CELL);
    const sr = Math.floor(sp.y / CELL);
    for (const o of [...layout.walls, ...layout.crates]) {
      const c = Math.round(o.x / CELL);
      const r = Math.round(o.y / CELL);
      if (Math.abs(c - sc) + Math.abs(r - sr) <= 1) return false;
    }
  }
  return true;
}

/** Worst (lowest) free-sample fraction over the straight spawn→centre lines. */
function worstCorridorFreeFrac(layout: ArenaLayout): number {
  const occ = occupancy(layout);
  let worst = 1;
  for (const sp of layout.spawns) {
    const len = Math.hypot(CENTER.x - sp.x, CENTER.y - sp.y);
    const n = Math.max(1, Math.ceil(len / (CELL / 4)));
    let free = 0;
    let total = 0;
    for (let k = 0; k <= n; k++) {
      const t = k / n;
      const x = sp.x + (CENTER.x - sp.x) * t;
      const y = sp.y + (CENTER.y - sp.y) * t;
      const c = Math.floor(x / CELL);
      const r = Math.floor(y / CELL);
      if (c < 0 || c >= GRID_COLS || r < 0 || r >= GRID_ROWS) continue;
      total++;
      if (occ[r * GRID_COLS + c] === 0) free++;
    }
    worst = Math.min(worst, free / total);
  }
  return worst;
}

/** Rotation identity: rotating every obstacle by 180° about the grid centre
 *  must return the same multiset. Keyed strings, so kind/hp must match too — a
 *  wall and a crate at mirrored positions is NOT symmetric. Spawns are held to
 *  the slice invariant instead of set closure: a 3-tank match takes the FIRST
 *  three quarter points, whose mirror set is the fourth, so the ACTIVE spawn
 *  subset cannot be 180°-closed (3 is odd over 4 points); what must hold is
 *  that every spawn's rotation is one of the four quarter points, i.e. cover
 *  is fair and each tank faces its exact rotational counterpart position. */
function symmetryProblems(layout: ArenaLayout): string[] {
  const problems: string[] = [];
  const boxes = new Set(
    [...layout.walls, ...layout.crates].map(
      (o) => `${o.x},${o.y},${o.w},${o.h},${o.kind},${o.hp}`,
    ),
  );
  for (const o of [...layout.walls, ...layout.crates]) {
    const rotated = `${ARENA_W - o.x - o.w},${ARENA_H - o.y - o.h},${o.w},${o.h},${o.kind},${o.hp}`;
    if (!boxes.has(rotated)) problems.push(`no 180° partner for ${o.kind}@${o.x},${o.y}`);
  }
  const quarter = new Set(QUARTER.map(([x, y]) => `${x},${y}`));
  for (const p of layout.spawns) {
    if (!quarter.has(`${ARENA_W - p.x},${ARENA_H - p.y}`)) {
      problems.push(`spawn ${p.x},${p.y} rotates off the quarter-point set`);
    }
  }
  return problems;
}

/** Structure contract: CELL-sized aligned boxes inside the arena, hp per kind,
 *  spawns exactly at the quarter points sliced in order. */
function structureProblems(layout: ArenaLayout, tankCount: number): string[] {
  const problems: string[] = [];
  if (layout.spawns.length !== tankCount) {
    problems.push(`spawns ${layout.spawns.length} != tankCount ${tankCount}`);
  }
  for (const o of [...layout.walls, ...layout.crates]) {
    if (o.w !== CELL || o.h !== CELL) problems.push(`obstacle not ${CELL}² at ${o.x},${o.y}`);
    if (o.x % CELL !== 0 || o.y % CELL !== 0) problems.push(`unaligned at ${o.x},${o.y}`);
    if (o.x < 0 || o.y < 0 || o.x + o.w > ARENA_W || o.y + o.h > ARENA_H) {
      problems.push(`out of arena at ${o.x},${o.y}`);
    }
    if (o.kind === "wall" && o.hp !== 0) problems.push(`wall hp ${o.hp} != 0`);
    if (o.kind === "crate" && o.hp !== CRATE_HP) problems.push(`crate hp ${o.hp} != ${CRATE_HP}`);
    if (o.kind !== "wall" && o.kind !== "crate") problems.push(`bad kind ${String(o.kind)}`);
  }
  layout.spawns.forEach((sp, i) => {
    if (sp.x !== QUARTER[i][0] || sp.y !== QUARTER[i][1]) {
      problems.push(`spawn ${i} at ${sp.x},${sp.y} != quarter point ${QUARTER[i]}`);
    }
  });
  return problems;
}

/** The whole battery for one layout — returns human-readable violations. */
function violations(layout: ArenaLayout, tankCount: number): string[] {
  const problems = [...structureProblems(layout, tankCount), ...symmetryProblems(layout)];
  const fill = floodFill(layout);
  if (fill.total === 0 || fill.largest / fill.total < 0.85) {
    problems.push(`flood fill ${fill.largest}/${fill.total} < 85%`);
  }
  if (!allSpawnsReachable(layout)) problems.push("a spawn is not reachable from spawn 0");
  if (minClearance(layout) < 100) problems.push(`clearance ${minClearance(layout)} < 100`);
  if (!spawnCellsClear(layout)) problems.push("obstacle on a spawn cell or its neighbour");
  const corridor = worstCorridorFreeFrac(layout);
  if (corridor < 0.6) problems.push(`corridor free ${corridor} < 60%`);
  return problems;
}

describe("grid geometry anchors", () => {
  it("pins GRID 18×14 CELL 50 against the world constants", () => {
    // These four numbers ARE the arena: every consumer (renderer background,
    // encoder raycasts, physics borders) derives from ARENA_W/ARENA_H, while
    // generation derives from the grid — the product must never drift apart.
    expect(GRID_COLS).toBe(18);
    expect(GRID_ROWS).toBe(14);
    expect(CELL).toBe(50);
    expect(GRID_COLS * CELL).toBe(ARENA_W);
    expect(GRID_ROWS * CELL).toBe(ARENA_H);
  });
});

describe("500-seed invariant battery", () => {
  it("every seed × seeded tankCount: symmetry, fill, clearance, corridor, structure", () => {
    const bad: string[] = [];
    // mulberry32 (test util, not the engine rng) picks the tank count per seed:
    // each seed then exercises a different slice of spawn pairs, and the run
    // itself is reproducible.
    const rng = mulberry32(20260830);
    for (let seed = 0; seed < 500; seed++) {
      const tankCount = [2, 3, 4][Math.floor(rng() * 3)] as 2 | 3 | 4;
      const layout = generateArena(seed, tankCount);
      for (const problem of violations(layout, tankCount)) bad.push(`seed ${seed}/${tankCount}: ${problem}`);
    }
    expect(bad).toEqual([]);
  });

  it("every seed: ≥85% of free cells in spawn 0's component", () => {
    const bad: string[] = [];
    for (let seed = 0; seed < 500; seed++) {
      const layout = generateArena(seed, 4);
      const { largest, total } = floodFill(layout);
      // total > 0 is part of the assertion: an all-wall board would pass 0/0
      // vacuously and strand every tank at tick 0.
      if (total <= 0 || largest / total < 0.85) {
        bad.push(`seed ${seed}: ${largest}/${total}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("every seed: all spawns reachable and ≥100 u clear of every obstacle", () => {
    const bad: string[] = [];
    for (let seed = 100; seed < 600; seed++) {
      const layout = generateArena(seed, 3);
      if (!allSpawnsReachable(layout)) bad.push(`seed ${seed}: unreachable spawn`);
      if (minClearance(layout) < 100) bad.push(`seed ${seed}: clearance ${minClearance(layout)}`);
      if (!spawnCellsClear(layout)) bad.push(`seed ${seed}: spawn cell blocked`);
    }
    expect(bad).toEqual([]);
  });

  it("every seed: 180° rotation maps the obstacle+spawn sets onto themselves", () => {
    const bad: string[] = [];
    for (let seed = 0; seed < 500; seed++) {
      const layout = generateArena(seed, 4);
      for (const problem of symmetryProblems(layout)) bad.push(`seed ${seed}: ${problem}`);
    }
    expect(bad).toEqual([]);
  });

  it("every seed: straight spawn→centre corridor ≥60% free", () => {
    const bad: string[] = [];
    for (let seed = 250; seed < 750; seed++) {
      const layout = generateArena(seed, 4);
      const worst = worstCorridorFreeFrac(layout);
      if (worst < 0.6) bad.push(`seed ${seed}: corridor ${worst.toFixed(3)}`);
    }
    expect(bad).toEqual([]);
  });
});

describe("determinism", () => {
  it("generateArena(s) is byte-identical to generateArena(s) across 500 seeds", () => {
    const bad: string[] = [];
    for (let seed = 0; seed < 500; seed++) {
      const a = generateArena(seed, 4);
      const b = generateArena(seed, 4);
      if (JSON.stringify(a) !== JSON.stringify(b)) bad.push(`seed ${seed}`);
    }
    expect(bad).toEqual([]);
  });

  it("adjacent seeds differ for ≥95% of (s, s+1) — the seed really varies boards", () => {
    // A generator that ignored its seed would pass every invariant above and
    // still be useless for training diversity; this catches it.
    let same = 0;
    const total = 499;
    for (let seed = 0; seed < total; seed++) {
      const a = JSON.stringify(generateArena(seed, 4));
      const b = JSON.stringify(generateArena(seed + 1, 4));
      if (a === b) same++;
    }
    expect(same / total).toBeLessThanOrEqual(0.05);
  });
});

describe("tankCount slices", () => {
  it("2, 3 and 4 tanks all yield valid boards with the matching spawn count", () => {
    const bad: string[] = [];
    for (const tankCount of [2, 3, 4] as const) {
      for (let seed = 0; seed < 60; seed++) {
        const layout = generateArena(seed * 7 + 3, tankCount);
        for (const problem of violations(layout, tankCount)) {
          bad.push(`seed ${seed * 7 + 3}/${tankCount}: ${problem}`);
        }
      }
    }
    expect(bad).toEqual([]);
  });
});

describe("SAFE_LAYOUT fallback", () => {
  it("passes the full battery with all four spawns", () => {
    expect(violations(SAFE_LAYOUT, 4)).toEqual([]);
  });

  it("is the hardcoded corner-L + centre-cluster board, not a generated one", () => {
    // Pins the fallback's shape so a refactor cannot quietly replace it with a
    // generated board (which would make the retry ladder's terminator a tautology).
    expect(SAFE_LAYOUT.spawns).toHaveLength(4);
    expect(SAFE_LAYOUT.walls.length).toBe(20); // four 5-cell Ls
    expect(SAFE_LAYOUT.crates.length).toBe(4); // centre 2×2 cluster
    expect(SAFE_LAYOUT.crates.every((c) => c.hp === CRATE_HP)).toBe(true);
  });

  it("keeps ≥85% fill, reachable spawns and open corridors", () => {
    const { largest, total } = floodFill(SAFE_LAYOUT);
    expect(largest / total).toBeGreaterThanOrEqual(0.85);
    expect(allSpawnsReachable(SAFE_LAYOUT)).toBe(true);
    expect(worstCorridorFreeFrac(SAFE_LAYOUT)).toBeGreaterThanOrEqual(0.6);
    expect(minClearance(SAFE_LAYOUT)).toBeGreaterThanOrEqual(100);
  });
});
