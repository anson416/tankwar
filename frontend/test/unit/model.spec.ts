// model.spec.ts — the Q-network factory contract (plan §Network). The only
// tfjs-touching spec besides dqn.spec: both architectures must build, map
// [B, 282] → [B, 18] through a LINEAR head, snapshot/restore bit-exactly,
// train through optimizer.minimize over variables(), and dispose back to the
// tensor baseline (the leak assertion the worker's reconfigure path relies
// on). The attention path additionally gets an offset sanity check: a valid
// flag written at the TANKS block offset must move q differently than one
// written at the BULLETS offset — the one place a silently drifted layout
// table would still produce shapes that "look right".
//
// CPU backend (dmc.spec precedent): this spec must run under Node in CI with
// no GPU, and the numbers it asserts are exact float comparisons — the CPU
// kernel is the deterministic reference the GPU backends are held near.
import { beforeAll, describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-layers";
import { ACTION_COUNT } from "../../src/game/actions";
import { OBS_DIM } from "../../src/game/constants";
import { OFF } from "../../src/game/encoding";
import { buildQNetwork } from "../../src/nn/model";
import type { QNetwork } from "../../src/nn/model";
import { defaultConfig } from "../../src/train-protocol";
import type { NetConfig } from "../../src/train-protocol";
import { mulberry32 } from "./util/rng";

// Tiny variants exercise the same code paths as the defaults but keep a CPU
// run in seconds; the DEFAULT configs are each built once to prove the plan's
// shipped numbers (hidden [256,256]; d64/4 heads/2 blocks/ffn 128) construct
// and forward — a default that only ever exists in the config form is a
// default that rots.
const MLP_TINY: NetConfig = { ...defaultConfig(), architecture: "mlp", hidden: [32, 32] };
const MLP_DEFAULT: NetConfig = { ...defaultConfig(), architecture: "mlp" };
const ATTN_TINY: NetConfig = {
  ...defaultConfig(),
  architecture: "attention",
  dModel: 16,
  heads: 2,
  layers: 2,
  ffn: 24,
};
const ATTN_DEFAULT: NetConfig = { ...defaultConfig(), architecture: "attention" };

beforeAll(async () => {
  expect(await tf.setBackend("cpu")).toBe(true);
  await tf.ready();
});

/** Seeded obs batch, values ∈ [−1, 1] like the encoder emits. No Math.random:
 *  a flaky input must never be able to flip a bit-exact assertion. */
function obsBatch(rows: number, seed: number): tf.Tensor2D {
  const rnd = mulberry32(seed);
  const buf = new Float32Array(rows * OBS_DIM);
  for (let i = 0; i < buf.length; i++) buf[i] = rnd() * 2 - 1;
  return tf.tensor2d(buf, [rows, OBS_DIM]);
}

/** All-zero obs with one mutation applied — zero is the sentinel row, so a
 *  zero obs is "nothing in the world but me". */
function obsWith(mutate: (buf: Float32Array) => void): tf.Tensor2D {
  const buf = new Float32Array(OBS_DIM);
  mutate(buf);
  return tf.tensor2d(buf, [1, OBS_DIM]);
}

/** Forward, await, dispose — the only q() call pattern that cannot leak. */
async function qVec(net: QNetwork, obs: tf.Tensor2D): Promise<Float32Array> {
  const y = net.q(obs);
  // Every net here is float32, so .data() hands back a Float32Array; the d.ts
  // widens it across dtypes, hence the one assertion.
  const d = (await y.data()) as Float32Array;
  y.dispose();
  return d;
}

function expectSame(a: Float32Array, b: Float32Array): void {
  expect(a.length).toBe(b.length);
  // toBe = Object.is: exact float equality, the whole point of the round-trip
  // assertion (a near-equal would hide a partial copy or an aliasing bug).
  for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]);
}

function expectDifferent(a: Float32Array, b: Float32Array): void {
  expect(a.length).toBe(b.length);
  expect(Array.from(a).some((v, i) => v !== b[i])).toBe(true);
}

describe("QNetwork MLP path", () => {
  it("maps [B, 282] → [B, 18], deterministically across repeated calls", async () => {
    const net = buildQNetwork(MLP_TINY);
    const obs = obsBatch(3, 1);
    const y = net.q(obs);
    expect(y.shape).toEqual([3, ACTION_COUNT]);
    y.dispose();
    const a = await qVec(net, obs);
    const b = await qVec(net, obs);
    const c = await qVec(net, obs);
    expectSame(a, b);
    expectSame(a, c);
    obs.dispose();
    net.dispose();
  });

  it("qSync is the documented alias of q (same numbers, same shape)", async () => {
    const net = buildQNetwork(MLP_TINY);
    const obs = obsBatch(2, 2);
    expectSame(await qVec(net, obs), await qVec(net, obs));
    const ys = net.qSync(obs);
    expect(ys.shape).toEqual([2, ACTION_COUNT]);
    ys.dispose();
    obs.dispose();
    net.dispose();
  });

  it("default [256, 256] params ≈ 143k ±10% (plan §Network)", () => {
    // Hand total: 282·256+256 + 256·256+256 + 256·18+18 = 142,866.
    const net = buildQNetwork(MLP_DEFAULT);
    const snap = net.weights();
    let total = 0;
    for (const t of snap) {
      let n = 1;
      for (const d of t.shape) n *= d;
      total += n;
    }
    snap.forEach((t) => t.dispose());
    net.dispose();
    expect(Math.abs(total - 143_000)).toBeLessThanOrEqual(14_300);
  });

  it("weights() → setWeights() round-trips bit-exactly with no aliasing", async () => {
    const a = buildQNetwork(MLP_TINY);
    const b = buildQNetwork(MLP_TINY);
    const obs = obsBatch(2, 3);
    // Two fresh nets have independent random init: the assertion has teeth.
    expectDifferent(await qVec(a, obs), await qVec(b, obs));
    const snap = a.weights();
    b.setWeights(snap);
    snap.forEach((t) => t.dispose()); // disposed BEFORE the check: b must own its copy
    expectSame(await qVec(a, obs), await qVec(b, obs));
    obs.dispose();
    a.dispose();
    b.dispose();
  });

  it("variables() > 0 and optimizer.minimize returns a finite scalar", async () => {
    const net = buildQNetwork(MLP_TINY);
    const obs = obsBatch(4, 4);
    const vars = net.variables();
    expect(vars.length).toBeGreaterThan(0);
    const opt = tf.train.adam(1e-3);
    const l1 = opt.minimize(() => tf.mean(net.qSync(obs)), true, vars);
    expect(l1).not.toBeNull();
    expect(Number.isFinite((await l1!.data())[0])).toBe(true);
    const l2 = opt.minimize(() => tf.mean(net.qSync(obs)), true, vars);
    expect(l2).not.toBeNull();
    expect(Number.isFinite((await l2!.data())[0])).toBe(true);
    l1!.dispose();
    l2!.dispose();
    opt.dispose();
    obs.dispose();
    net.dispose();
  });
});

describe("QNetwork attention path", () => {
  it("tiny config builds and maps [B, 282] → [B, 18]; variables() > 0", async () => {
    const net = buildQNetwork(ATTN_TINY);
    const obs = obsBatch(2, 5);
    const y = net.q(obs);
    expect(y.shape).toEqual([2, ACTION_COUNT]);
    expect(net.variables().length).toBeGreaterThan(0);
    expectSame(await qVec(net, obs), await qVec(net, obs));
    y.dispose();
    obs.dispose();
    net.dispose();
  });

  it("plan-default attention config (d64, 4 heads, 2 blocks, ffn 128) builds", async () => {
    const net = buildQNetwork(ATTN_DEFAULT);
    const obs = obsBatch(1, 6);
    const y = net.q(obs);
    expect(y.shape).toEqual([1, ACTION_COUNT]);
    y.dispose();
    obs.dispose();
    net.dispose();
  });

  it("weights() → setWeights() round-trips bit-exactly with no aliasing", async () => {
    const a = buildQNetwork(ATTN_TINY);
    const b = buildQNetwork(ATTN_TINY);
    const obs = obsBatch(2, 7);
    expectDifferent(await qVec(a, obs), await qVec(b, obs));
    const snap = a.weights();
    b.setWeights(snap);
    snap.forEach((t) => t.dispose());
    expectSame(await qVec(a, obs), await qVec(b, obs));
    obs.dispose();
    a.dispose();
    b.dispose();
  });

  it("gradients flow through the token encoder: finite loss over two steps", async () => {
    const net = buildQNetwork(ATTN_TINY);
    const obs = obsBatch(4, 8);
    const vars = net.variables();
    const opt = tf.train.adam(1e-3);
    const l1 = opt.minimize(() => tf.mean(net.qSync(obs)), true, vars);
    const l2 = opt.minimize(() => tf.mean(net.qSync(obs)), true, vars);
    expect(l1).not.toBeNull();
    expect(l2).not.toBeNull();
    expect(Number.isFinite((await l1!.data())[0])).toBe(true);
    expect(Number.isFinite((await l2!.data())[0])).toBe(true);
    l1!.dispose();
    l2!.dispose();
    opt.dispose();
    obs.dispose();
    net.dispose();
  });

  it("token slicing honours the encoding layout table", async () => {
    // All-zero obs ⇒ every entity token reads the sentinel embedding row and
    // the pool weights it to 0; SELF (always valid) plus RAYS (0) drive q0.
    const net = buildQNetwork(ATTN_TINY);
    const q0 = await qVec(net, obsWith(() => undefined));
    // valid=1 written at the TANKS block's first float: slot 0 goes
    // sentinel → tank-typed token AND enters the pool.
    const qTank = await qVec(net, obsWith((buf) => { buf[OFF.TANKS] = 1; }));
    // The same flag at the BULLETS offset must route to the BULLET type row —
    // a layout drift that swapped the two blocks cannot pass this.
    const qBullet = await qVec(net, obsWith((buf) => { buf[OFF.BULLETS] = 1; }));
    // A plain SELF feature exercises the concat path (no token involved).
    const qSelf = await qVec(net, obsWith((buf) => { buf[OFF.SELF + 2] = 1; }));
    expectDifferent(q0, qTank);
    expectDifferent(qTank, qBullet);
    expectDifferent(q0, qSelf);
    net.dispose();
  });

  it("rejects incoherent configs and weight lists", () => {
    // hd = dModel / heads must be integral or the head split is nonsense.
    expect(() => buildQNetwork({ ...ATTN_TINY, dModel: 12, heads: 5 })).toThrow(RangeError);
    expect(() => buildQNetwork({ ...ATTN_TINY, layers: 0 })).toThrow(RangeError);
    const net = buildQNetwork(ATTN_TINY);
    const w = net.weights();
    // Validated BEFORE any assignment: a bad list must not half-update the net.
    expect(() => net.setWeights(w.slice(0, w.length - 1))).toThrow(RangeError);
    w.forEach((t) => t.dispose());
    net.dispose();
  });
});

describe("QNetwork dispose discipline", () => {
  it("full lifecycle returns numTensors to baseline (both architectures)", async () => {
    const baseline = tf.memory().numTensors;
    for (const cfg of [MLP_TINY, ATTN_TINY]) {
      const src = buildQNetwork(cfg);
      const obs = obsBatch(2, 9);
      expect((await qVec(src, obs)).length).toBe(2 * ACTION_COUNT);
      // Snapshot → restore into a SECOND net → drop everything. This is the
      // exact sequence the worker's target sync and eval snapshots perform.
      const snap = src.weights();
      const dst = buildQNetwork(cfg);
      dst.setWeights(snap);
      snap.forEach((t) => t.dispose());
      expect((await qVec(dst, obs)).length).toBe(2 * ACTION_COUNT);
      obs.dispose();
      src.dispose();
      dst.dispose();
    }
    expect(tf.memory().numTensors).toBe(baseline);
  });
});
