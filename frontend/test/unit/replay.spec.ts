// replay.spec.ts — the flat-ring replay contract: ROW_FLOATS arithmetic,
// bit-exact push/sample round-trip, ring wrap evicting the OLDEST rows without
// corrupting survivors, seeded determinism of the sample stream (and the
// explicit-state threading that makes it checkpointable), and the fill /
// oversample edges. Every test row is TAGGED — column c of row j holds a value
// derived from (j, c) — so a wrap bug that bleeds one row into the next names
// the exact row and column instead of failing as a vague loss spike.
import { describe, expect, it } from "vitest";
import { mulberry32 } from "./util/rng";
import { OBS_DIM } from "../../src/game/constants";
import { ROW_FLOATS, ReplayBuffer, type ReplayBatch } from "../../src/nn/replay";

// --- tagged fixture rows -----------------------------------------------------
//
// Tag arithmetic stays under 2^24 (the float32 exact-integer range) so every
// stored value round-trips bit-exactly and `toBe` equality is meaningful.

const TAG_STRIDE = 1000;

/** s of row j: j·1000 + [0..282); s2: j·1000 + [284..566) — the row's own
 *  offsets, so a wrongly-copied half of a row cannot masquerade as another. */
function tag(j: number): { s: Float32Array; s2: Float32Array } {
  const s = new Float32Array(OBS_DIM);
  const s2 = new Float32Array(OBS_DIM);
  for (let c = 0; c < OBS_DIM; c++) {
    s[c] = j * TAG_STRIDE + c;
    s2[c] = j * TAG_STRIDE + (OBS_DIM + 2) + c;
  }
  return { s, s2 };
}

/** Push tagged row j; a/r/done are valid transition values, also tagged by j. */
function pushRow(buf: ReplayBuffer, j: number): void {
  const { s, s2 } = tag(j);
  buf.push(s, j % 18, ((j % 21) - 10) * 0.25, s2, j % 7 === 0 ? 1 : 0);
}

/** True iff batch row `i` is bit-exactly tagged row `j` — all 567 floats. */
function matchesRow(b: ReplayBatch, i: number, j: number): boolean {
  const base = i * OBS_DIM;
  for (let c = 0; c < OBS_DIM; c++) {
    if (b.s[base + c] !== j * TAG_STRIDE + c) return false;
    if (b.s2[base + c] !== j * TAG_STRIDE + (OBS_DIM + 2) + c) return false;
  }
  return (
    b.a[i] === (j % 18) &&
    b.r[i] === ((j % 21) - 10) * 0.25 &&
    b.done[i] === (j % 7 === 0 ? 1 : 0)
  );
}

/** The row id a batch row claims to be, read off its own first float. */
function rowId(b: ReplayBatch, i: number): number {
  return Math.floor(b.s[i * OBS_DIM] / TAG_STRIDE);
}

/** Fill `buf` with tagged rows 0..rows-1. */
function fill(buf: ReplayBuffer, rows: number): ReplayBuffer {
  for (let j = 0; j < rows; j++) pushRow(buf, j);
  return buf;
}

/** Byte-equality of two whole batches (same size, same every element). */
function batchEq(x: ReplayBatch, y: ReplayBatch): boolean {
  if (x.size !== y.size) return false;
  for (let i = 0; i < x.size; i++) {
    if (x.a[i] !== y.a[i] || x.r[i] !== y.r[i] || x.done[i] !== y.done[i]) return false;
    for (let c = 0; c < OBS_DIM; c++) {
      if (x.s[i * OBS_DIM + c] !== y.s[i * OBS_DIM + c]) return false;
      if (x.s2[i * OBS_DIM + c] !== y.s2[i * OBS_DIM + c]) return false;
    }
  }
  return true;
}

/** Byte-equality of single rows of two batches. */
function rowsEq(x: ReplayBatch, xi: number, y: ReplayBatch, yi: number): boolean {
  for (let c = 0; c < OBS_DIM; c++) {
    if (x.s[xi * OBS_DIM + c] !== y.s[yi * OBS_DIM + c]) return false;
    if (x.s2[xi * OBS_DIM + c] !== y.s2[yi * OBS_DIM + c]) return false;
  }
  return x.a[xi] === y.a[yi] && x.r[xi] === y.r[yi] && x.done[xi] === y.done[yi];
}

describe("row layout", () => {
  it("pins ROW_FLOATS at 567 = s(282) + a(1) + r(1) + s′(282) + done(1)", () => {
    expect(ROW_FLOATS).toBe(2 * OBS_DIM + 3);
    expect(ROW_FLOATS).toBe(567); // the plan's literal — the worker quotes it
  });
});

describe("push/sample round-trip", () => {
  it("returns pushed rows bit-exactly, all 567 floats including a/r/done", () => {
    const cap = 8;
    const buf = fill(new ReplayBuffer(cap), cap);
    // n > size caps at size (8 draws over 8 rows, replacement allowed).
    const { batch } = buf.sample(cap * 4, 0x51ee);
    expect(batch.size).toBe(cap);
    for (let i = 0; i < batch.size; i++) {
      const j = rowId(batch, i);
      expect(j).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(cap);
      expect(matchesRow(batch, i, j)).toBe(true);
    }
  });

  it("round-trips arbitrary float32 payloads bit-exactly", () => {
    // Non-tagged, obs-scale random floats: the worst case for a copy bug that
    // shuffles or truncates columns. Each sampled row must byte-match SOME
    // pushed payload (with replacement, ids cannot be assumed unique).
    const cap = 6;
    const buf = new ReplayBuffer(cap);
    const rng = mulberry32(0x2a11);
    const payload: { s: Float32Array; a: number; r: number; s2: Float32Array; done: number }[] = [];
    // Scratch float32: rounds r through the SAME store the buffer uses, so
    // exact comparison is honest rather than accidentally forgiving.
    const f32 = new Float32Array(1);
    for (let j = 0; j < cap; j++) {
      const s = new Float32Array(OBS_DIM);
      const s2 = new Float32Array(OBS_DIM);
      for (let c = 0; c < OBS_DIM; c++) {
        s[c] = rng() * 2 - 1;
        s2[c] = rng() * 2 - 1;
      }
      f32[0] = (rng() * 2 - 1) * 60; // death-scale rewards
      const row = {
        s,
        a: (rng() * 18) | 0,
        r: f32[0],
        s2,
        done: rng() < 0.2 ? 1 : 0,
      };
      payload.push(row);
      buf.push(row.s, row.a, row.r, row.s2, row.done);
    }
    const { batch } = buf.sample(300, 7); // capped at the 6 pushed rows
    expect(batch.size).toBe(cap);
    for (let i = 0; i < batch.size; i++) {
      let hit = false;
      for (const w of payload) {
        let eq = true;
        for (let c = 0; c < OBS_DIM && eq; c++) {
          if (batch.s[i * OBS_DIM + c] !== w.s[c]) eq = false;
          if (batch.s2[i * OBS_DIM + c] !== w.s2[c]) eq = false;
        }
        if (eq && batch.a[i] === w.a && batch.done[i] === w.done) {
          f32[0] = w.r;
          eq = batch.r[i] === f32[0];
          hit = hit || eq;
        }
      }
      expect(hit).toBe(true); // no sampled row may be anything but a pushed row
    }
  });
});

describe("ring wrap", () => {
  it("pushing capacity+50 evicts the 50 oldest, caps size, and never corrupts a survivor", () => {
    const cap = 64;
    const extra = 50;
    const total = cap + extra;
    const buf = fill(new ReplayBuffer(cap), total);
    expect(buf.size).toBe(cap); // capped, not grown
    expect(buf.fillFraction()).toBe(1);
    // Every returned row must be an INTACT survivor — tag j ≥ 50 (the evicted
    // oldest are unreachable) and all 567 floats still exact, so an off-by-one
    // in the ring cursor or row stride names itself here. sample caps at size,
    // so four independent draws are merged to widen coverage.
    const seen = new Set<number>();
    for (let run = 0; run < 4; run++) {
      const { batch } = buf.sample(cap, 0x1a11 + run);
      expect(batch.size).toBe(cap);
      for (let i = 0; i < batch.size; i++) {
        const j = rowId(batch, i);
        expect(j).toBeGreaterThanOrEqual(extra);
        expect(j).toBeLessThan(total);
        expect(matchesRow(batch, i, j)).toBe(true);
        seen.add(j);
      }
    }
    // 256 draws over 64 rows leave a row unseen with p≈(63/64)^256≈0.018 per
    // row — most survivors must be revisited. A silently-collapsed ring
    // (sampling one physical row twice) would crater this.
    expect(seen.size).toBeGreaterThan(cap / 2);
  });

  it("wraps at capacity 1 and 2 without ever resurfacing an overwritten row", () => {
    const one = new ReplayBuffer(1);
    pushRow(one, 5);
    pushRow(one, 6); // evicts 5 immediately
    // sample caps at size, so each draw returns the single survivor — 50 runs
    // pin that the victim (5) can never resurface and the row stays intact.
    for (let run = 0; run < 50; run++) {
      const b1 = one.sample(8, 3 + run).batch;
      expect(b1.size).toBe(1);
      expect(rowId(b1, 0)).toBe(6);
      expect(matchesRow(b1, 0, 6)).toBe(true);
    }
    const two = fill(new ReplayBuffer(2), 3); // rows 1,2 survive; 0 is gone
    const seen = new Set<number>();
    for (let run = 0; run < 12; run++) {
      const b2 = two.sample(2, 3 + run).batch;
      expect(b2.size).toBe(2);
      for (let i = 0; i < b2.size; i++) {
        const j = rowId(b2, i);
        expect(j).toBeGreaterThan(0);
        expect(matchesRow(b2, i, j)).toBe(true);
        seen.add(j);
      }
    }
    // 24 draws over 2 rows misses one with p≈(1/2)^24≈6e-8 — both reachable.
    expect(seen.size).toBe(2);
  });
});

describe("determinism", () => {
  const buf = fill(new ReplayBuffer(40), 40);

  it("same rngState ⇒ identical batch and identical successor state", () => {
    const a = buf.sample(64, 0x1234);
    const b = buf.sample(64, 0x1234);
    expect(a.state).toBe(b.state);
    expect(batchEq(a.batch, b.batch)).toBe(true);
  });

  it("threads the state: sample(24,s0) then sample(16,s₁) equals sample(40,s0)", () => {
    // The contract the trainer relies on to checkpoint mid-epoch: the returned
    // state is exactly "the counter after the draws", so splitting a sample
    // across calls reproduces the unsplit stream byte-for-byte.
    const s0 = 0xbead;
    const whole = buf.sample(40, s0); // capped at the 40 pushed rows
    const first = buf.sample(24, s0);
    const rest = buf.sample(16, first.state);
    expect(rest.state).toBe(whole.state);
    for (let i = 0; i < 24; i++) expect(rowsEq(first.batch, i, whole.batch, i)).toBe(true);
    for (let i = 0; i < 16; i++) expect(rowsEq(rest.batch, i, whole.batch, 24 + i)).toBe(true);
  });

  it("different rngState ⇒ almost surely a different batch", () => {
    const x = buf.sample(32, 1);
    const y = buf.sample(32, 2);
    expect(batchEq(x.batch, y.batch)).toBe(false); // 32 draws agreeing by chance: ~(1/40)^32
  });
});

describe("fill and oversample", () => {
  it("fillFraction tracks pushes and saturates at 1", () => {
    const buf = new ReplayBuffer(4);
    expect(buf.fillFraction()).toBe(0);
    pushRow(buf, 0);
    expect(buf.fillFraction()).toBe(0.25);
    fill(buf, 3);
    expect(buf.fillFraction()).toBe(1);
    fill(buf, 9); // overfill: still full, never over 1
    expect(buf.fillFraction()).toBe(1);
    expect(buf.size).toBe(4);
  });

  it("sampling more than size returns exactly size rows, all from what was pushed", () => {
    const buf = fill(new ReplayBuffer(64), 3); // rows 0..2, buffer mostly empty
    const { batch } = buf.sample(100, 0x0dd);
    expect(batch.size).toBe(3);
    for (let i = 0; i < batch.size; i++) {
      const j = rowId(batch, i);
      expect(j).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(3);
      expect(matchesRow(batch, i, j)).toBe(true);
    }
  });

  it("sampling an empty buffer returns size 0, empty arrays, and an UNCHANGED state", () => {
    const buf = new ReplayBuffer(8);
    const { batch, state } = buf.sample(5, 0xfeed);
    expect(batch.size).toBe(0);
    expect(batch.s.length).toBe(0);
    expect(batch.a.length).toBe(0);
    expect(state).toBe(0xfeed); // no draws consumed — the counter is untouched
    expect(buf.fillFraction()).toBe(0);
  });
});
