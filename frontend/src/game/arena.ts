// arena.ts — seeded procedural arena generation with hard, test-checked
// invariants. DOM-free, tfjs-free: runs inside the train worker AND in node
// vitest (no Math.random anywhere — every draw threads a mulberry32 state).
//
// WHY generate per episode instead of hand-placing one map: a training fleet
// playing thousands of episodes overfits to a fixed layout within minutes (the
// original gym env had exactly this problem — one static 450×350 board), and
// the plan pins "obstacles generated per episode" with four invariants:
// 180°-rotational symmetry (so no spawn side is advantaged — critical when all
// agents share ONE network), obstacle spacing (no unshootable nooks), spawn
// clearance (nobody opens inside a wall or under instant fire), and
// connectivity (≥85% of the floor reachable from spawn 0 — no sealed pockets a
// timeout-staller can camp).
//
// Representation: everything is grid-first. An obstacle is always exactly one
// CELL×CELL axis-aligned block at (col·CELL, row·CELL); wall/crate geometry is
// tracked in flat Uint8Arrays (index = row·GRID_COLS + col) because the flood
// fill and the spacing checks are grid neighbourhood operations, and only the
// final ArenaLayout is materialised as Obstacle AABBs. Crates are stored full
// CELL-sized too — the visual inset is a RENDER concern (renderer.ts), never a
// physics one, so the engine and the encoder raycasts see one honest box.
//
// Symmetry is maintained by CONSTRUCTION, not checked after the fact: every
// random placement happens in "generator space" and its 180° rotation about the
// grid centre — cell (c,r) ↦ (GRID_COLS−1−c, GRID_ROWS−1−r) — is applied in the
// same step, and every deletion removes a cell and its rotation together. So
// arena.spec's rotate-everything test is a real invariant probe, not a tautology.

import { ARENA_H, ARENA_W, CRATE_HP } from "./constants";
// randInt only: every generator decision is an integer draw (blob counts, cell
// picks, walk directions) so the threaded state stays exact — rand()'s float
// path would be just as deterministic but nothing here needs it.
import { randInt } from "./rng";
import type { Obstacle, Vec } from "./types";

/** Arena width in grid cells — 18·50 = ARENA_W (constants.ts, 2× the 2020 board). */
export const GRID_COLS = 18;
/** Arena height in grid cells — 14·50 = ARENA_H. */
export const GRID_ROWS = 14;
/** Edge length u of one obstacle block (walls AND crates — see header). */
export const CELL = 50;

/** A ready-to-play arena: static geometry plus the spawn points, in the
 *  fixed quarter-point order (top-left, top-right, bottom-left, bottom-right)
 *  sliced to the tank count. Consumers treat it as read-only — the engine
 *  copies walls/crates into GameState and never mutates the layout. */
export interface ArenaLayout {
  walls: Obstacle[];
  crates: Obstacle[];
  spawns: Vec[];
}

// --- generation tuning (module-private: these are generator knobs, not world
// --- physics; nothing outside this file may key off them) -------------------

const CELLS = GRID_COLS * GRID_ROWS;
/** Rows 0..GRID_ROWS/2−1 — the north half where wall blobs are born before
 *  mirroring. Even GRID_ROWS ⇒ no cell is its own mirror, so blobs and their
 *  rotations are always disjoint sets (no special-casing the centre line). */
const NORTH_ROWS = GRID_ROWS / 2;
const WALL_SEED_MIN = 5;
const WALL_SEED_MAX = 8;
const WALL_GROW_MIN = 1;
const WALL_GROW_MAX = 3;
/** Blob-seed rejections before giving up on that seed (spacing is easy to
 *  satisfy on 18×7, so this only bounds a pathologically unlucky run). */
const SEED_TRIES = 40;
/** ≥1.5 cells between blob CENTRES (Euclidean, in cells): forbids seeds on the
 *  same or diagonally adjacent cells so 5–8 blobs stay 5–8 distinct structures
 *  instead of merging into one giant wall by random walk. */
const SEED_SPACING = 1.5;
const CRATE_MIN = 8;
const CRATE_MAX = 14;
/** Crate rejections per mirrored pair (per the plan) — a pair that cannot fit
 *  is skipped, and the final 8..14 count check rejects starved boards. */
const CRATE_REJECT_TRIES = 50;
/** Crates keep a full empty cell (Chebyshev ≥ 2) from walls and each other: a
 *  crate flush against a wall creates a dead corner bullets can never reach,
 *  and adjacent crates turn three hits of cover into an unbreakable rampart. */
const CRATE_CHEBYSHEV = 2;
/** No obstacle AABB within this distance u of any ACTIVE spawn, and nothing
 *  on a spawn cell or its 4-neighbours. 100 u = 2 cells ≈ 2.5 tank diameters:
 *  enough that a tank spawns pointing at open floor, not at a wall face. */
const SPAWN_CLEARANCE = 100;
/** Straight spawn→centre line must be this fraction free (sampled every
 *  quarter cell) — guarantees every spawn has an immediate route into the
 *  middle, so early engagements never need a maze solve. */
const CORRIDOR_FREE_FRAC = 0.6;
const CORRIDOR_STEP = CELL / 4;
/** Floor fraction that must be reachable from spawn 0 (plan invariant). */
const FILL_FRAC = 0.85;
/** Retries with seed+1 before falling back to SAFE_LAYOUT. 20 is plenty: a
 *  failing seed fails for a *reason* (over-dense roll), not by a hair, and the
 *  fallback must stay rare enough that its sameness is never felt in training. */
const MAX_ATTEMPTS = 20;

/** Arena centre — the far end of every corridor and the 180° rotation pivot. */
const CENTER: Vec = { x: ARENA_W / 2, y: ARENA_H / 2 };

/** The four quarter points, in tank-id order. Fixed (not rolled): the clear-
 *  ance/corridor guarantees are proven for exactly these points, and mirrored
 *  pairs A↔D, B↔C keep every layout fair for any tankCount slice of them. */
const QUARTER_SPAWNS: Vec[] = [
  { x: 75, y: 87.5 },
  { x: 825, y: 87.5 },
  { x: 75, y: 612.5 },
  { x: 825, y: 612.5 },
];

/** 180° rotation of a cell about the grid centre. */
function mirrorI(i: number): number {
  const c = i % GRID_COLS;
  const r = (i / GRID_COLS) | 0;
  return (GRID_ROWS - 1 - r) * GRID_COLS + (GRID_COLS - 1 - c);
}

/** Precomputed mirror of every cell index — the deletion pass pairs cells with
 *  their rotations hundreds of times per attempt, so table-look it up once. */
const MIRROR = new Int32Array(CELLS);
for (let i = 0; i < CELLS; i++) MIRROR[i] = mirrorI(i);

/** AABB of grid cell (c,r) — the geometry every distance/sampling check uses. */
function boxOf(c: number, r: number): { x: number; y: number; w: number; h: number } {
  return { x: c * CELL, y: r * CELL, w: CELL, h: CELL };
}

/** Materialise a grid cell as an engine Obstacle. Walls carry hp 0 (they cannot
 *  take damage — `kind` discriminates, hp only matters for crates); crates carry
 *  CRATE_HP so three hits break one and two do not (constants.ts). */
function cellObstacle(c: number, r: number, kind: "wall" | "crate"): Obstacle {
  const box = boxOf(c, r);
  return { x: box.x, y: box.y, w: box.w, h: box.h, kind, hp: kind === "crate" ? CRATE_HP : 0 };
}

/** Shortest distance from a point to an axis-aligned box (0 when inside). */
function distPointBox(px: number, py: number, b: { x: number; y: number; w: number; h: number }): number {
  const dx = Math.max(b.x - px, 0, px - (b.x + b.w));
  const dy = Math.max(b.y - py, 0, py - (b.y + b.h));
  return Math.hypot(dx, dy);
}

/** Claim a grid cell for wall blob `blob`, appending it to that blob's cell
 *  list. Refuses already-owned cells so a random walk that revisits or crosses
 *  another blob never double-claims — ownership stays unique, which is what
 *  makes "delete blocking blob" (corridor repair) a clean operation. */
function claimCell(
  walls: Uint8Array,
  owner: Int16Array,
  cells: number[],
  blob: number,
  c: number,
  r: number,
): void {
  const i = r * GRID_COLS + c;
  if (owner[i] !== -1) return;
  owner[i] = blob;
  walls[i] = 1;
  cells.push(i);
}

/** Crate placement test: no wall or crate within Chebyshev CRATE_CHEBYSHEV−1
 *  (the square ring around the candidate — "no occupant at distance ≤1" is
 *  exactly "distance ≥ 2"). */
function crateFits(walls: Uint8Array, crates: Uint8Array, c: number, r: number): boolean {
  const ring = CRATE_CHEBYSHEV - 1;
  for (let dr = -ring; dr <= ring; dr++) {
    const nr = r + dr;
    if (nr < 0 || nr >= GRID_ROWS) continue;
    for (let dc = -ring; dc <= ring; dc++) {
      const nc = c + dc;
      if (nc < 0 || nc >= GRID_COLS) continue;
      const i = nr * GRID_COLS + nc;
      if (walls[i] || crates[i]) return false;
    }
  }
  return true;
}

/** True iff cell (c,r) is at least SPAWN_CLEARANCE from ALL FOUR quarter
 *  spawns — folded into crate rejection so a placed crate is never one that
 *  clearance pruning would immediately delete. Checked against all four (not
 *  just the active tankCount slice) for two reasons: the deletion pass clears
 *  mirrors anyway, so a crate near an INACTIVE spawn would vanish too and
 *  starve the 8..14 count on exactly the 2-tank boards; and it keeps the
 *  generated geometry a pure function of the seed, identical across tank
 *  counts, so cross-count comparisons in training stay meaningful. */
function clearOfSpawns(c: number, r: number): boolean {
  const box = boxOf(c, r);
  for (const sp of QUARTER_SPAWNS) {
    if (distPointBox(sp.x, sp.y, box) < SPAWN_CLEARANCE) return false;
  }
  return true;
}

/** BFS over free cells (4-connectivity) from `start`. Returns the component
 *  size, the total free-cell count and a membership mask — the mask is what the
 *  spawn-reachability check reads, so the walk runs once per attempt. */
function component(
  walls: Uint8Array,
  crates: Uint8Array,
  start: number,
): { largest: number; total: number; inComponent: Uint8Array } {
  const free = new Uint8Array(CELLS);
  let total = 0;
  for (let i = 0; i < CELLS; i++) {
    if (!walls[i] && !crates[i]) {
      free[i] = 1;
      total++;
    }
  }
  const inComponent = new Uint8Array(CELLS);
  let largest = 0;
  if (start >= 0 && start < CELLS && free[start] === 1) {
    // Array + head pointer, not shift(): O(1) dequeue, no realloc per pop.
    const queue = new Int32Array(CELLS);
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    inComponent[start] = 1;
    while (head < tail) {
      const i = queue[head++];
      largest++;
      const c = i % GRID_COLS;
      const r = (i / GRID_COLS) | 0;
      // Neighbours in fixed order — the visit ORDER never matters (it is a set
      // fill), only that all four are scanned, so no rng is consumed here.
      if (c > 0 && free[i - 1] === 1 && inComponent[i - 1] === 0) {
        inComponent[i - 1] = 1;
        queue[tail++] = i - 1;
      }
      if (c < GRID_COLS - 1 && free[i + 1] === 1 && inComponent[i + 1] === 0) {
        inComponent[i + 1] = 1;
        queue[tail++] = i + 1;
      }
      if (r > 0 && free[i - GRID_COLS] === 1 && inComponent[i - GRID_COLS] === 0) {
        inComponent[i - GRID_COLS] = 1;
        queue[tail++] = i - GRID_COLS;
      }
      if (r < GRID_ROWS - 1 && free[i + GRID_COLS] === 1 && inComponent[i + GRID_COLS] === 0) {
        inComponent[i + GRID_COLS] = 1;
        queue[tail++] = i + GRID_COLS;
      }
    }
  }
  return { largest, total, inComponent };
}

/** Grid occupancy arrays for an existing layout (exported-path reuse: floodFill
 *  must measure real layouts, not just freshly built grids). */
function gridsFromLayout(layout: ArenaLayout): { walls: Uint8Array; crates: Uint8Array } {
  const walls = new Uint8Array(CELLS);
  const crates = new Uint8Array(CELLS);
  const place = (o: Obstacle, grid: Uint8Array): void => {
    // round(), not floor(): obstacle coords are exact multiples of CELL by
    // contract; rounding absorbs float noise if a layout ever round-trips
    // through JSON (the {arena} protocol message) before being measured.
    const c = Math.round(o.x / CELL);
    const r = Math.round(o.y / CELL);
    if (c >= 0 && c < GRID_COLS && r >= 0 && r < GRID_ROWS) grid[r * GRID_COLS + c] = 1;
  };
  for (const o of layout.walls) place(o, walls);
  for (const o of layout.crates) place(o, crates);
  return { walls, crates };
}

/**
 * One generation attempt from a single seed. Returns null on ANY violated
 * invariant — generateArena retries with seed+1, so null is control flow, not
 * an error. Every rng draw threads `s` (rand/randInt are pure), so an attempt
 * is a pure function of its seed and tankCount.
 */
function buildAttempt(seed: number, tankCount: 2 | 3 | 4): ArenaLayout | null {
  let s = seed | 0;
  const walls = new Uint8Array(CELLS);
  const crates = new Uint8Array(CELLS);
  const owner = new Int16Array(CELLS).fill(-1);
  /** Per-blob cell indices, north half + mirrors — the unit of corridor repair. */
  const blobCells: number[][] = [];

  // --- 1. wall blobs: 5–8 seeds in the north half, grown, then mirrored -----

  let draw: number;
  [draw, s] = randInt(s, WALL_SEED_MAX - WALL_SEED_MIN + 1);
  const seedCount = WALL_SEED_MIN + draw;
  const seeds: Vec[] = []; // grid coords smuggled through Vec: x = col, y = row
  for (let i = 0; i < seedCount; i++) {
    for (let tries = 0; tries < SEED_TRIES; tries++) {
      let c: number;
      let r: number;
      [c, s] = randInt(s, GRID_COLS);
      [r, s] = randInt(s, NORTH_ROWS);
      let spaced = true;
      for (const p of seeds) {
        if (Math.hypot(p.x - c, p.y - r) < SEED_SPACING) {
          spaced = false;
          break;
        }
      }
      if (!spaced) continue;
      seeds.push({ x: c, y: r });
      break;
    }
  }
  // A starved board means the spacing roll jammed — retry from a fresh seed
  // rather than shipping fewer structures than the plan's 5–8.
  if (seeds.length < WALL_SEED_MIN) return null;

  for (let b = 0; b < seeds.length; b++) {
    const cells: number[] = [];
    const c0 = seeds[b].x;
    const r0 = seeds[b].y;
    claimCell(walls, owner, cells, b, c0, r0);
    // Grow 1–3 cells by random walk: blobs stay small (≤4 cells ⇒ ≤8 with the
    // mirror) so no single structure can ever cut the board in half — the
    // 85% connectivity bar is then a formality, not a coin flip.
    [draw, s] = randInt(s, WALL_GROW_MAX - WALL_GROW_MIN + 1);
    const steps = WALL_GROW_MIN + draw;
    let c = c0;
    let r = r0;
    for (let k = 0; k < steps; k++) {
      let d: number;
      [d, s] = randInt(s, 4);
      // d: 0 = north, 1 = east, 2 = south, 3 = west. The walk MOVES even off-
      // board (clamped to a no-op) — re-rolling on a wall would bias blob
      // shape toward the board interior and away from the border funnels.
      c += d === 1 ? 1 : d === 3 ? -1 : 0;
      r += d === 0 ? -1 : d === 2 ? 1 : 0;
      if (c < 0 || c >= GRID_COLS || r < 0 || r >= NORTH_ROWS) continue;
      claimCell(walls, owner, cells, b, c, r);
    }
    // Mirror the blob as a UNIT after growth: the rotation of a connected walk
    // is a connected walk, so each blob reads as one symmetric structure (an
    // L in the top-left implies the same L, upside down, in the bottom-right).
    for (const i of cells.slice()) {
      const m = MIRROR[i];
      claimCell(walls, owner, cells, b, m % GRID_COLS, (m / GRID_COLS) | 0);
    }
    blobCells.push(cells);
  }

  // --- 2. crates: 8–14 in mirrored pairs, one empty cell from everything ----

  // Even counts only — pairs are disjoint (no self-mirroring cell exists), so
  // odd totals are unreachable and randInt would otherwise waste its draw.
  [draw, s] = randInt(s, (CRATE_MAX - CRATE_MIN) / 2 + 1);
  const cratePairs = CRATE_MIN / 2 + draw;
  let crateCount = 0;
  for (let p = 0; p < cratePairs; p++) {
    for (let tries = 0; tries < CRATE_REJECT_TRIES; tries++) {
      let c: number;
      let r: number;
      [c, s] = randInt(s, GRID_COLS);
      [r, s] = randInt(s, GRID_ROWS);
      const mc = GRID_COLS - 1 - c;
      const mr = GRID_ROWS - 1 - r;
      // BOTH cells of the pair are checked before either is claimed — never
      // half-placed, which would break symmetry. Spawn clearance is part of
      // the rejection, symmetric by construction. The PAIRWISE Chebyshev test
      // is separate and explicit: crateFits only measures each cell against
      // ALREADY-PLACED occupancy, and neither cell is claimed yet, so a
      // candidate adjacent to its own mirror (only reachable in the centre
      // 2×2, where c/mc and r/mr differ by exactly 1) would pass both crateFits
      // calls and ship two corner-fused crates — the unbreakable-rampart
      // geometry CRATE_CHEBYSHEV exists to prevent.
      if (
        !clearOfSpawns(c, r) ||
        !crateFits(walls, crates, c, r) ||
        !crateFits(walls, crates, mc, mr) ||
        Math.max(Math.abs(c - mc), Math.abs(r - mr)) < CRATE_CHEBYSHEV
      ) {
        continue;
      }
      crates[r * GRID_COLS + c] = 1;
      crates[mr * GRID_COLS + mc] = 1;
      crateCount += 2;
      break;
    }
  }

  // --- 3. spawns + clearance pruning (delete with mirror) -------------------

  const spawns = QUARTER_SPAWNS.slice(0, tankCount);
  const kill = new Uint8Array(CELLS);
  for (const sp of spawns) {
    const sc = Math.floor(sp.x / CELL);
    const sr = Math.floor(sp.y / CELL);
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const i = r * GRID_COLS + c;
        if (!walls[i] && !crates[i]) continue;
        // Manhattan ≤1 = the spawn cell itself plus its 4-neighbours — belt to
        // the 100 u brace: it holds even if SPAWN_CLEARANCE is ever retuned.
        const near =
          distPointBox(sp.x, sp.y, boxOf(c, r)) < SPAWN_CLEARANCE ||
          Math.abs(c - sc) + Math.abs(r - sr) <= 1;
        if (near) {
          kill[i] = 1;
          kill[MIRROR[i]] = 1;
        }
      }
    }
  }
  for (let i = 0; i < CELLS; i++) {
    if (!kill[i]) continue;
    // Mirrors of cleared cells near spawn C/D are cleared even for tankCount 2
    // (A↔D and B↔C are rotation pairs), so the UNUSED spawns stay clear too —
    // a mid-episode respawn onto the far quarter points must not be a trap.
    if (walls[i] === 1) {
      walls[i] = 0;
      owner[i] = -1;
    }
    crates[i] = 0;
  }
  // Drop pruned cells from blob lists so corridor repair deletes only cells a
  // blob still owns.
  for (const cells of blobCells) {
    for (let k = cells.length - 1; k >= 0; k--) {
      if (owner[cells[k]] === -1) cells.splice(k, 1);
    }
  }

  // --- 4. corridor: spawn→centre line ≥60% free, else delete blocking blob --
  //
  // The 60% bar is per LINE and measured before any blob is blamed: a wall
  // face grazing the diagonal (a 95%-free line) is cover on the way in, not a
  // maze. Blaming blobs on any single blocked sample would gut the board and
  // collapse distinct seeds onto near-empty layouts — exactly the diversity
  // per-episode generation exists to provide.

  for (let pass = 0; pass <= blobCells.length; pass++) {
    const blockedBy = new Map<number, number>();
    let clear = true;
    for (const sp of spawns) {
      const len = Math.hypot(CENTER.x - sp.x, CENTER.y - sp.y);
      const n = Math.max(1, Math.ceil(len / CORRIDOR_STEP));
      let free = 0;
      let total = 0;
      // Blocked-sample blob ids for THIS line, merged into the tally only if
      // the line actually fails — a passing line never nominates a blob.
      const lineBlocked: number[] = [];
      for (let k = 0; k <= n; k++) {
        const t = k / n;
        const c = Math.floor((sp.x + (CENTER.x - sp.x) * t) / CELL);
        const r = Math.floor((sp.y + (CENTER.y - sp.y) * t) / CELL);
        if (c < 0 || c >= GRID_COLS || r < 0 || r >= GRID_ROWS) continue;
        total++;
        const i = r * GRID_COLS + c;
        if (!walls[i] && !crates[i]) {
          free++;
          continue;
        }
        // Only WALL blobs are repairable — crates are cover by design, and a
        // line blocked mostly by crates means this seed rolled too dense.
        if (walls[i] === 1) lineBlocked.push(owner[i]);
      }
      if (free / total < CORRIDOR_FREE_FRAC) {
        clear = false;
        for (const b of lineBlocked) {
          if (b >= 0) blockedBy.set(b, (blockedBy.get(b) ?? 0) + 1);
        }
      }
    }
    if (clear) break;
    let worst = -1;
    let worstHits = 0;
    for (const [b, hits] of blockedBy) {
      if (hits > worstHits) {
        worst = b;
        worstHits = hits;
      }
    }
    // No blob blocks any sampled point but the line still fails ⇒ crates did
    // it, or the free fraction lost to sheer bad luck — neither is fixable by
    // deleting walls. Give the attempt up.
    if (worst < 0) return null;
    for (const i of blobCells[worst]) {
      if (owner[i] === worst) {
        walls[i] = 0;
        owner[i] = -1;
      }
    }
    blobCells[worst] = [];
  }

  // --- 5. final invariants: crate density + connectivity -------------------

  crateCount = 0;
  for (let i = 0; i < CELLS; i++) {
    if (crates[i] === 1) crateCount++;
  }
  // Clearance pruning can starve a board below the plan's 8 crates — that is a
  // bad roll, not a layout to ship half-dressed.
  if (crateCount < CRATE_MIN || crateCount > CRATE_MAX) return null;

  const start = spawns.length > 0 ? Math.floor(spawns[0].y / CELL) * GRID_COLS + Math.floor(spawns[0].x / CELL) : -1;
  const comp = component(walls, crates, start);
  if (comp.total === 0 || comp.largest / comp.total < FILL_FRAC) return null;
  for (const sp of spawns) {
    const i = Math.floor(sp.y / CELL) * GRID_COLS + Math.floor(sp.x / CELL);
    if (comp.inComponent[i] === 0) return null;
  }

  // --- 6. materialise (row-major, so JSON serialisations are byte-stable) ---

  const wallObstacles: Obstacle[] = [];
  const crateObstacles: Obstacle[] = [];
  for (let i = 0; i < CELLS; i++) {
    if (walls[i] === 1) wallObstacles.push(cellObstacle(i % GRID_COLS, (i / GRID_COLS) | 0, "wall"));
    if (crates[i] === 1) crateObstacles.push(cellObstacle(i % GRID_COLS, (i / GRID_COLS) | 0, "crate"));
  }
  return { walls: wallObstacles, crates: crateObstacles, spawns };
}

/**
 * Generate the arena for one episode. Deterministic: the same (seed, tankCount)
 * always returns deep-equal output — the layout seeds GameState.arenaSeed and
 * the 500-seed property tests.
 *
 * Retry ladder: seed, seed+1, … seed+19, then SAFE_LAYOUT. The +1 ladder (not
 * a re-roll of the same seed) guarantees progress: each attempt is a pure
 * function of its seed, so retrying a failing seed would fail identically.
 */
export function generateArena(seed: number, tankCount: 2 | 3 | 4): ArenaLayout {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const layout = buildAttempt(seed + attempt, tankCount);
    if (layout) return layout;
  }
  // Walls/crates are shared by reference from SAFE_LAYOUT on purpose: the
  // fallback is immutable and every consumer copies into GameState anyway;
  // only spawns are sliced, to the match's tank count.
  return {
    walls: SAFE_LAYOUT.walls,
    crates: SAFE_LAYOUT.crates,
    spawns: SAFE_LAYOUT.spawns.slice(0, tankCount),
  };
}

/**
 * Free-cell 4-connectivity report for a layout, flooding from spawn 0's cell.
 * `largest` is the size of spawn 0's component, `total` the number of free
 * cells — callers assert largest/total ≥ 0.85 (plan invariant) after checking
 * every spawn lies in spawn 0's component separately (reachability needs the
 * membership mask, which this summary deliberately does not expose).
 */
export function floodFill(layout: ArenaLayout): { largest: number; total: number } {
  const { walls, crates } = gridsFromLayout(layout);
  const sp = layout.spawns[0];
  const start =
    sp !== undefined
      ? Math.floor(sp.y / CELL) * GRID_COLS + Math.floor(sp.x / CELL)
      : -1;
  const comp = component(walls, crates, start);
  return { largest: comp.largest, total: comp.total };
}

// --- SAFE_LAYOUT: the hardcoded fallback -------------------------------------
//
// Used only after 20 failed attempts (never observed across the 500-seed suite,
// but the ladder must terminate SOMEWHERE for engine.spec's termination proof).
// Hand-built to pass every check the generator enforces: 180°-symmetric by
// construction (the bottom two L-walls are the rotations of the top two), every
// L ≥ 100 u from its nearest spawn (first wall face sits 125 u out — see the
// cell maths below), the centre 2×2 crate cluster leaves both corridors >90%
// free, and the board is obviously one connected component.
//
// Top-left L, in cells (col,row): arm along the top border (4,0)(5,0)(6,0) plus
// arm down the left of it (4,1)(4,2). Spawn A sits in cell (1,1) at (75,87.5);
// the nearest wall face, cell (4,·) at x=200, is 125 u away — inside the
// 100 u bar with 25 u to spare. The top-right L is the x-mirror; the bottom two
// are the 180° rotations of both, computed rather than transcribed so the
// symmetry cannot drift out of the hardcoded list.

/** Corner L walls of the north half; rotations supply the south half. */
const SAFE_NORTH_WALL_CELLS: [col: number, row: number][] = [
  [4, 0],
  [5, 0],
  [6, 0],
  [4, 1],
  [4, 2],
  [13, 0],
  [12, 0],
  [11, 0],
  [13, 1],
  [13, 2],
];

/** Centre 2×2 crate cluster — already 180°-closed: (8,6)↔(9,7), (9,6)↔(8,7).
 *  It is the cover both corridors deliberately run PAST (each line clips at
 *  most 2 of ~38 samples), not through. */
const SAFE_CRATE_CELLS: [col: number, row: number][] = [
  [8, 6],
  [9, 6],
  [8, 7],
  [9, 7],
];

/** Dedupe-by-cell-index build so a hand-edited list cannot silently double-
 *  claim a cell; row-major sort keeps the serialisation byte-stable. */
export const SAFE_LAYOUT: ArenaLayout = (() => {
  const cells = new Map<number, "wall" | "crate">();
  const put = (c: number, r: number, kind: "wall" | "crate"): void => {
    cells.set(r * GRID_COLS + c, kind);
    const m = MIRROR[r * GRID_COLS + c];
    cells.set(m, kind);
  };
  for (const [c, r] of SAFE_NORTH_WALL_CELLS) put(c, r, "wall");
  for (const [c, r] of SAFE_CRATE_CELLS) put(c, r, "crate");
  const walls: Obstacle[] = [];
  const crates: Obstacle[] = [];
  for (const i of [...cells.keys()].sort((a, b) => a - b)) {
    const kind = cells.get(i) as "wall" | "crate";
    const o = cellObstacle(i % GRID_COLS, (i / GRID_COLS) | 0, kind);
    (kind === "wall" ? walls : crates).push(o);
  }
  return {
    walls,
    crates,
    // Fresh point objects: SAFE_LAYOUT is a shared singleton, and a caller that
    // nudged a spawn Vec in place would corrupt every future fallback episode.
    spawns: QUARTER_SPAWNS.map((p) => ({ x: p.x, y: p.y })),
  };
})();
