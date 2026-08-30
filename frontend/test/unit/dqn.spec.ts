// dqn.spec.ts — THE tfjs spec (CPU backend, dmc.spec/model.spec precedent):
// the Dqn learner driven through the REAL Trainer and the REAL engine on tiny
// weights. The centrepiece assertions:
//   finite losses   ~60 seeded mini games through Trainer (episodeSeconds 2 ⇒
//                   ≤60-tick timeouts) with ≥30 gradient steps — every Huber
//                   loss finite, the DDQN target never NaNs;
//   leak immunity   tf.memory().numTensors IDENTICAL across a 20-fleetTick
//                   block that includes act forwards AND train steps (the
//                   house no-dataSync / dispose-everything rule, verified);
//   target sync     syncTarget makes net.q and target.q BIT-equal;
//   snapshot round-trip  snapshot → train → restore returns q outputs
//                   bit-equal (eval-on-a-frozen-copy depends on this);
//   ring cap        the shared replay ring caps at bufferSize;
//   DDQN smoke      doubleQ true and false both train finitely;
//   act contract    fire masking lands BEFORE the argmax, exploration is
//                   deterministic under a threaded rngState, temperature
//                   sampling stays in range, and the empty-fleet edge is a
//                   no-op.
//
// CPU on purpose: the numbers asserted are exact float comparisons, and CI has
// no GPU. Tiny config ([32] hidden, batch 16) keeps the whole file in seconds.
import { beforeAll, describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-layers";
import { ACTION_COUNT } from "../../src/game/actions";
import { OBS_DIM } from "../../src/game/constants";
import { Dqn } from "../../src/nn/dqn";
import { ReplayBuffer } from "../../src/nn/replay";
import type { ReplayBatch } from "../../src/nn/replay";
import { Trainer } from "../../src/nn/scheduler";
import type { QNetwork } from "../../src/nn/model";
import { defaultConfig } from "../../src/train-protocol";
import type { NetConfig } from "../../src/train-protocol";
import { mulberry32 } from "./util/rng";

// Tiny but real: lanes 3 keeps three episodes in flight (fewer fleetTicks for
// the 60-episode target), trainEvery 12 with 4–8 decisions/tick interleaves a
// gradient step every ~2 ticks, targetSync 40 exercises syncTarget mid-training
// exactly the way the worker will drive it.
const CFG: NetConfig = {
  ...defaultConfig(),
  architecture: "mlp",
  hidden: [32],
  lanes: 3,
  tankCount: 2,
  episodeSeconds: 2,
  batchSize: 16,
  bufferSize: 512,
  learningRate: 1e-3,
  trainEvery: 12,
  targetSync: 40,
  epsStart: 1,
  epsDecay: 0.9,
  epsMin: 0.05,
  evalEvery: 1e9, // eval is exercised explicitly in scheduler.spec; not here
  evalEpisodes: 2,
  doubleQ: true,
};

beforeAll(async () => {
  expect(await tf.setBackend("cpu")).toBe(true);
  await tf.ready();
});

/** Seeded obs batch, values ∈ [−1, 1] like encodeObs emits. Never Math.random:
 *  a bit-exact assertion must not be flippable by an unseeded input. */
function obsTensor(rows: number, seed: number): tf.Tensor2D {
  const rnd = mulberry32(seed);
  const buf = new Float32Array(rows * OBS_DIM);
  for (let i = 0; i < buf.length; i++) buf[i] = rnd() * 2 - 1;
  return tf.tensor2d(buf, [rows, OBS_DIM]);
}

/** A hand-built ReplayBatch with plausible reward/done marginals. */
function syntheticBatch(n: number, seed: number): ReplayBatch {
  const rnd = mulberry32(seed);
  const s = new Float32Array(n * OBS_DIM);
  const s2 = new Float32Array(n * OBS_DIM);
  for (let i = 0; i < n * OBS_DIM; i++) {
    s[i] = rnd() * 2 - 1;
    s2[i] = rnd() * 2 - 1;
  }
  const a = new Int32Array(n);
  const r = new Float32Array(n);
  const done = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.floor(rnd() * ACTION_COUNT);
    r[i] = rnd() * 20 - 10;
    done[i] = rnd() < 0.3 ? 1 : 0;
  }
  return { s, a, r, s2, done, size: n };
}

/** Forward, await, dispose — the only q() call pattern that cannot leak. */
async function qOf(net: QNetwork, obs: tf.Tensor2D): Promise<Float32Array> {
  const y = net.q(obs);
  const d = (await y.data()) as Float32Array;
  y.dispose();
  return d;
}

/** toBe = Object.is: bit equality, the whole point of the sync/restore checks. */
function expectSame(a: Float32Array, b: Float32Array): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]);
}

function expectDifferent(a: Float32Array, b: Float32Array): void {
  expect(a.length).toBe(b.length);
  expect(Array.from(a).some((v, i) => v !== b[i])).toBe(true);
}

describe("Dqn through the real Trainer (plan §Training)", () => {
  it("plays ~60 seeded games, ≥30 gradient steps, every loss finite, ring capped", async () => {
    const dqn = new Dqn(CFG);
    const buffer = new ReplayBuffer(CFG.bufferSize);
    const trainer = new Trainer(CFG, dqn, buffer);
    const losses: number[] = [];
    trainer.onTrainStep = (l) => losses.push(l);
    await trainer.reset(0xbeef);
    // The worker's targetSync duty, driven here the same way.
    let lastSync = 0;
    let ticks = 0;
    while (trainer.episodes < 60 && ticks++ < 6000) {
      await trainer.fleetTick();
      if (dqn.steps - lastSync >= CFG.targetSync) {
        dqn.syncTarget();
        lastSync = dqn.steps;
      }
    }
    expect(trainer.episodes).toBe(60); // the tick guard must never be what ends this
    expect(dqn.steps).toBeGreaterThanOrEqual(30);
    expect(losses).toHaveLength(dqn.steps); // the hook saw exactly the applied steps
    for (const l of losses) expect(Number.isFinite(l)).toBe(true);
    // Ring wrapped and capped: ~decisions of pushes far exceed 512 rows.
    expect(buffer.size).toBe(CFG.bufferSize);
    expect(buffer.fillFraction()).toBe(1);
    dqn.dispose();
  }, 180_000);

  it("20 fleetTicks of act + train leave tf.memory().numTensors IDENTICAL", async () => {
    const dqn = new Dqn(CFG);
    const buffer = new ReplayBuffer(CFG.bufferSize);
    const trainer = new Trainer(CFG, dqn, buffer);
    await trainer.reset(0x51ee);
    // Warm-up: fill the ring AND apply at least one Adam step before the
    // baseline — the first minimize lazily allocates Adam's m/v slots, and a
    // baseline taken before that would mask a one-off, not a per-tick, leak.
    let guard = 0;
    while (dqn.steps < 1 && guard++ < 100) await trainer.fleetTick();
    expect(dqn.steps).toBeGreaterThanOrEqual(1);
    const baseline = tf.memory().numTensors;
    for (let k = 0; k < 20; k++) {
      await trainer.fleetTick();
      // Inline check: a leak here accumulates, so fail at the first tick that
      // grew rather than only at the end (names the offending iteration).
      expect(tf.memory().numTensors).toBe(baseline);
    }
    dqn.dispose();
    // Full teardown lands below the baseline (the baseline still held the net).
    expect(tf.memory().numTensors).toBeLessThan(baseline);
  }, 120_000);
});

describe("target sync + snapshot round-trip", () => {
  it("syncTarget makes net.q and target.q bit-equal on a fixed obs", async () => {
    const dqn = new Dqn(CFG);
    const obs = obsTensor(3, 5);
    // The constructor's sync means even a FRESH pair is already equal.
    expectSame(await qOf(dqn.net, obs), await qOf(dqn.target, obs));
    // Diverge the online net, prove the copy is not aliasing, re-sync.
    for (let i = 0; i < 5; i++) await dqn.trainStep(syntheticBatch(CFG.batchSize, 11 + i));
    expectDifferent(await qOf(dqn.net, obs), await qOf(dqn.target, obs));
    dqn.syncTarget();
    expectSame(await qOf(dqn.net, obs), await qOf(dqn.target, obs));
    obs.dispose();
    dqn.dispose();
  });

  it("snapshot → train → restore returns q outputs bit-equal", async () => {
    const dqn = new Dqn(CFG);
    const obs = obsTensor(2, 7);
    const snap = dqn.snapshot();
    const before = await qOf(dqn.net, obs);
    for (let i = 0; i < 4; i++) await dqn.trainStep(syntheticBatch(CFG.batchSize, 21 + i));
    const after = await qOf(dqn.net, obs);
    expectDifferent(after, before); // training genuinely moved the net, or this proves nothing
    dqn.restore(snap);
    expectSame(await qOf(dqn.net, obs), before); // restored exactly, not approximately
    dqn.disposeSnapshot(snap);
    obs.dispose();
    dqn.dispose();
  });
});

describe("DDQN smoke (both arms train finitely)", () => {
  it("doubleQ true and false both produce finite losses", async () => {
    for (const doubleQ of [true, false]) {
      const dqn = new Dqn({ ...CFG, doubleQ });
      for (let i = 0; i < 4; i++) {
        const loss = await dqn.trainStep(syntheticBatch(CFG.batchSize, 31 + i));
        expect(loss).not.toBeNull();
        expect(Number.isFinite(loss as number)).toBe(true);
      }
      expect(dqn.steps).toBe(4);
      dqn.dispose();
    }
  });

  it("an empty batch declines to train without touching the step counter", async () => {
    const dqn = new Dqn(CFG);
    const empty = syntheticBatch(4, 1);
    expect(await dqn.trainStep({ ...empty, size: 0 })).toBeNull();
    expect(dqn.steps).toBe(0);
    dqn.dispose();
  });
});

describe("act contract", () => {
  it("masks unavailable fire before argmax, threads rngState, samples in range", async () => {
    const dqn = new Dqn(CFG);
    const obs = [new Float32Array(OBS_DIM), new Float32Array(OBS_DIM)];
    const noFire = [false, false];

    // Greedy (ε=0) with the mask on: no ODD action index can win — the fire
    // axis is stride 2, so every masked slot is −Inf before the argmax.
    const greedy = await dqn.act(obs, noFire, { kind: "explore", epsilon: 0 }, 12345);
    expect(greedy.actions).toHaveLength(2);
    for (const a of greedy.actions) expect(a % 2).toBe(0);
    // Same seed ⇒ same draws ⇒ the identical greedy answer.
    const replay = await dqn.act(obs, noFire, { kind: "explore", epsilon: 0 }, 12345);
    expect([...replay.actions]).toEqual([...greedy.actions]);
    expect(replay.state).toBe(greedy.state);

    // ε=1 explores uniformly over ALL 18 — in range (and NOT fire-masked by
    // design; the engine no-ops an unavailable shot).
    const explore = await dqn.act(obs, noFire, { kind: "explore", epsilon: 1 }, 777);
    for (const a of explore.actions) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(ACTION_COUNT);
    }
    const again = await dqn.act(obs, noFire, { kind: "explore", epsilon: 1 }, 777);
    expect([...again.actions]).toEqual([...explore.actions]);
    expect(again.state).toBe(explore.state);

    // Temperature: near-greedy categorical, still in table range.
    const soft = await dqn.act(obs, noFire, { kind: "temperature", tau: 0.05 }, 31);
    for (const a of soft.actions) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(ACTION_COUNT);
    }

    // Empty fleet: a no-op that neither draws from nor advances the stream.
    const none = await dqn.act([], [], { kind: "explore", epsilon: 1 }, 99);
    expect(none.actions).toHaveLength(0);
    expect(none.state).toBe(99);
    dqn.dispose();
  });

  it("fire-available rows are NOT masked: greedy can pick a fire action", async () => {
    // A zero obs makes Q equal the head bias exactly (zero-init biases, ReLU
    // hidden), so biasing action 1 — a fire action — gives the greedy answer a
    // deterministic target regardless of glorot init entropy. The same
    // machinery that masked fire above must leave a LEGAL fire pick alone, and
    // still mask it the moment fire becomes unavailable.
    const cfg: NetConfig = { ...CFG, hidden: [4] };
    const dqn = new Dqn(cfg);
    const snap = dqn.snapshot();
    const head = snap[snap.length - 1]!; // MLP head bias, shape [18]
    const bias = new Float32Array(await head.data());
    bias[1] += 10;
    const edited = tf.tensor1d(bias);
    snap[snap.length - 1] = edited;
    dqn.restore(snap);
    edited.dispose();
    dqn.disposeSnapshot(snap);

    const obs = [new Float32Array(OBS_DIM)];
    const free = await dqn.act(obs, [true], { kind: "explore", epsilon: 0 }, 1);
    expect(free.actions[0]).toBe(1); // legal fire, chosen greedily
    const masked = await dqn.act(obs, [false], { kind: "explore", epsilon: 0 }, 1);
    expect(masked.actions[0] % 2).toBe(0); // mask still applies when it must
    dqn.dispose();
  });
});

describe("structural fit", () => {
  it("buildQNetwork still fronts both architectures through Dqn", async () => {
    // Guard against a future Dqn that bypasses the model factory: the MLP and
    // attention paths must both come up behind the learner's constructor.
    for (const architecture of ["mlp", "attention"] as const) {
      const cfg: NetConfig = {
        ...CFG,
        architecture,
        dModel: 16,
        heads: 2,
        layers: 1,
        ffn: 16,
      };
      const dqn = new Dqn(cfg);
      const obs = obsTensor(2, 13);
      expectSame(await qOf(dqn.net, obs), await qOf(dqn.target, obs));
      obs.dispose();
      dqn.dispose();
    }
  });
});
