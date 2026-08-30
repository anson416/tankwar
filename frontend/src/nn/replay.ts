// replay.ts — the flat-ring experience buffer every training agent samples.
// DOM-free, tfjs-free: plain typed-array arithmetic, so node vitest can
// property-test the exact bytes the worker will train on, and the worker can
// build tfjs tensors straight from the batch's freshly copied arrays.
//
// WHY one preallocated Float32Array instead of an array of transition objects:
// the plan's default buffer is 40k rows × 567 floats (~91 MB). Per-row objects
// would mean 40k × 5 live allocations and GC pauses exactly when the scheduler
// is busiest; one flat array is a SINGLE allocation reused forever, and
// push() never allocates at all. Allocation happens once in the constructor —
// an absurd capacity throws RangeError there, which the worker catches and
// turns into an `{error}` message instead of a half-built trainer.
//
// ROW LAYOUT (row base = rowIndex · ROW_FLOATS, all offsets in floats):
//   [0         .. OBS_DIM    )  s     observation BEFORE the action
//   [OBS_DIM   .. OBS_DIM+1  )  a     action index 0..17, stored as a float
//   [OBS_DIM+1 .. OBS_DIM+2  )  r     per-tick reward
//   [OBS_DIM+2 .. 2·OBS_DIM+2)  s2    observation AFTER the action
//   [2·OBS_DIM+2 .. ROW_FLOATS)  done  1.0 = s2 ends the episode (no bootstrap)
// 282 + 1 + 1 + 282 + 1 = ROW_FLOATS = 567 — one DQN transition per row.
//
// WHY a ring, not shift-on-full (the app's chart RingBuffer style): shifting
// 40k rows of 567 floats per push is ~22 M element moves a tick; the ring
// overwrites one stale row and bumps a cursor. `head` is the next slot to
// overwrite and, once the buffer is full, doubles as the OLDEST surviving row
// — so the logical order (oldest → newest) is head, head+1, … wrapping, and
// uniform sampling can walk it without any bookkeeping.
//
// SAMPLING: uniform WITH replacement (the tic-tac-toe reference's precedent —
// at 40k rows the duplicate-row bias a 256-row batch introduces is negligible
// next to the bookkeeping without-replacement costs, and it keeps sample()
// O(n) with no allocation beyond the output batch). Indices come from
// game/rng's explicit-state mulberry32 threaded through the returned `state`,
// never Math.random: a training run is seed-reproducible bit-for-bit, and a
// paused/resumed lane continues the same stream. Output rows are memcpy'd row
// ranges into FRESH typed arrays — a later push may overwrite the ring, but a
// batch already handed to a gradient step can never change under it.

import { OBS_DIM } from "../game/constants";
import { randInt } from "../game/rng";

/** Floats per stored transition: s(OBS_DIM) + a + r + s2(OBS_DIM) + done. */
export const ROW_FLOATS = 2 * OBS_DIM + 3;

// Column offsets inside one row (derived from OBS_DIM, never hand-maintained:
// a stride/offset mismatch here would silently train on shifted observations).
const OFF_A = OBS_DIM;
const OFF_R = OBS_DIM + 1;
const OFF_S2 = OBS_DIM + 2;
const OFF_DONE = 2 * OBS_DIM + 2;

/** One sampled training batch. `size` may be smaller than the requested n
 *  (capped at the buffer's current size); every array holds exactly `size`
 *  rows and is owned by the caller. */
export interface ReplayBatch {
  s: Float32Array;
  a: Int32Array;
  r: Float32Array;
  s2: Float32Array;
  done: Uint8Array;
  size: number;
}

export class ReplayBuffer {
  private data: Float32Array;
  private count = 0;
  /** Next row index to overwrite; once full, also the oldest surviving row. */
  private head = 0;

  constructor(public capacity: number) {
    // Single allocation for the buffer's whole life. Let an absurd size throw
    // — the worker catches it and reports {error} rather than limping on.
    this.data = new Float32Array(capacity * ROW_FLOATS);
  }

  get size(): number {
    return this.count;
  }

  /** Store one transition, overwriting the oldest row once at capacity. */
  push(s: Float32Array, a: number, r: number, s2: Float32Array, done: number): void {
    const base = this.head * ROW_FLOATS;
    this.data.set(s, base);
    this.data[base + OFF_A] = a;
    this.data[base + OFF_R] = r;
    this.data.set(s2, base + OFF_S2);
    this.data[base + OFF_DONE] = done;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  /** Draw `n` rows uniformly WITH replacement; thread the rng counter. */
  sample(n: number, rngState: number): { batch: ReplayBatch; state: number } {
    const k = Math.min(n, this.count);
    const batch: ReplayBatch = {
      s: new Float32Array(k * OBS_DIM),
      a: new Int32Array(k),
      r: new Float32Array(k),
      s2: new Float32Array(k * OBS_DIM),
      done: new Uint8Array(k),
      size: k,
    };
    let state = rngState;
    for (let i = 0; i < k; i++) {
      const [pick, next] = randInt(state, this.count);
      state = next;
      // Logical → physical: while still filling, rows sit at 0..count-1 in
      // push order; once full, `head` IS the oldest row, so logical position
      // `pick` lives at (head + pick) mod capacity.
      const row =
        (this.count < this.capacity ? pick : (this.head + pick) % this.capacity) * ROW_FLOATS;
      batch.s.set(this.data.subarray(row, row + OBS_DIM), i * OBS_DIM);
      batch.a[i] = this.data[row + OFF_A];
      batch.r[i] = this.data[row + OFF_R];
      batch.s2.set(this.data.subarray(row + OFF_S2, row + OFF_S2 + OBS_DIM), i * OBS_DIM);
      batch.done[i] = this.data[row + OFF_DONE];
    }
    return { batch, state };
  }

  /** Fraction of rows in use — the train tab's "buffer %" tile. */
  fillFraction(): number {
    return this.count / this.capacity;
  }
}
