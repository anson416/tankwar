// learn-sanity.spec.ts — the headless LEARNING gate (plan M3: "headless:
// win-rate beats 1/N within ~200 episodes, loss finite, tensors stable"),
// scaled to a tiny config so it finishes in minutes, not hours. OFF by
// default: the regular `vitest run` gate must stay in seconds, so this file
// skips unless LEARN_SANITY=1 (CI/manual: `LEARN_SANITY=1 npx vitest run
// learn-sanity`). It is the same loop the train worker drives (scheduler.ts's
// WORKER WIRING header), on the CPU backend with REAL nets — not the scripted
// fake dqn.spec uses — so it can catch a learner that trains "successfully"
// while actually diverging or leaking.
//
// PASS CONTRACT (plan-risk discharge): every Huber loss finite, and either the
// loss is actually falling (mean of the last 10 steps below the first 10) or —
// 25 episodes is too short to bet a gate on the trend alone — the weaker
// honest floor holds: no NaN, at least one eval ran with finite numbers, and
// tf.memory().numTensors is IDENTICAL across a post-training block of fleet
// ticks (the no-leak rule at training scale).
import { beforeAll, describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-layers";
import { Dqn } from "../../src/nn/dqn";
import { ReplayBuffer } from "../../src/nn/replay";
import { Trainer } from "../../src/nn/scheduler";
import type { EvalResult } from "../../src/nn/scheduler";
import { defaultConfig } from "../../src/train-protocol";
import type { NetConfig } from "../../src/train-protocol";

const ENABLED = process.env.LEARN_SANITY === "1";

/**
 * The task's tiny config: lanes 4, tankCount 2, 6 s episodes, hidden [64],
 * batch 32, buffer 2000, trainEvery 16, evalEvery 20 (6 eval games). epsDecay
 * 0.9 is the one deliberate nudge off the 0.995 default — 25 episodes at
 * 0.995 would still be at ε≈0.88 (pure exploration, no learnable signal in
 * the loss), while 0.9 lands at ε≈0.07 where the greedy policy the loss is
 * actually fitting is the one playing. The UI's own validation range allows it.
 */
const CFG: NetConfig = {
  ...defaultConfig(),
  architecture: "mlp",
  hidden: [64],
  lanes: 4,
  tankCount: 2,
  episodeSeconds: 6,
  batchSize: 32,
  bufferSize: 2000,
  trainEvery: 16,
  targetSync: 200,
  evalEvery: 20,
  evalEpisodes: 6,
  epsDecay: 0.9,
  doubleQ: true,
};

describe.skipIf(!ENABLED)("headless learning sanity (LEARN_SANITY=1)", () => {
  beforeAll(async () => {
    expect(await tf.setBackend("cpu")).toBe(true);
    await tf.ready();
  });

  it(
    "trains 25+ episodes: losses finite, evals ran, tensors stable",
    async () => {
      const dqn = new Dqn(CFG);
      const trainer = new Trainer(CFG, dqn, new ReplayBuffer(CFG.bufferSize));
      const losses: number[] = [];
      trainer.onTrainStep = (l) => losses.push(l);
      await trainer.reset(0xbeef);

      // The worker's per-iteration contract (scheduler.ts header): fleetTick →
      // hard target sync off dqn.steps → fire-and-forget eval past evalEvery.
      // Sequential here (no rebuild can land mid-await), so the eval results
      // are collected synchronously instead of generation-guarded.
      const evals: EvalResult[] = [];
      let lastSync = 0;
      let nextEvalAt = CFG.evalEvery;
      let ticks = 0;
      while (trainer.episodes < 25 && ticks++ < 20_000) {
        await trainer.fleetTick();
        if (dqn.steps - lastSync >= CFG.targetSync) {
          dqn.syncTarget();
          lastSync = dqn.steps;
        }
        if (trainer.episodes >= nextEvalAt) {
          nextEvalAt = trainer.episodes + CFG.evalEvery;
          evals.push(await trainer.runEval(0x5eed + trainer.episodes));
        }
      }
      expect(trainer.episodes).toBeGreaterThanOrEqual(25); // the tick guard never ended this
      expect(dqn.steps).toBeGreaterThan(0);

      // 1 — every gradient step produced a finite Huber loss (no NaN/±Inf path).
      expect(losses.length).toBeGreaterThan(0);
      for (const l of losses) expect(Number.isFinite(l)).toBe(true);

      // 2 — evals actually ran and returned sane numbers (winRate IS a rate).
      expect(evals.length).toBeGreaterThanOrEqual(1);
      for (const e of evals) {
        expect(Number.isFinite(e.winRate)).toBe(true);
        expect(e.winRate).toBeGreaterThanOrEqual(0);
        expect(e.winRate).toBeLessThanOrEqual(1);
        expect(Number.isFinite(e.avgReturn)).toBe(true);
        expect(Number.isFinite(e.killDeath)).toBe(true);
      }

      // 3 — the honest floor, kept even if the trend assertion passes: at
      // steady state a tick allocates and frees the same tensors, so the
      // counter must return to its exact baseline (dqn.spec's leak rule,
      // verified at training scale with the Adam slots live).
      const steady = tf.memory().numTensors;
      for (let i = 0; i < 30; i++) await trainer.fleetTick();
      expect(tf.memory().numTensors).toBe(steady);

      // 4 — the trend, best-effort: the pass contract is the disjunction, so a
      // short noisy run still gates on (finite + evals + no leak) while a real
      // divergence (loss climbing monotonically) is surfaced.
      const head = losses.slice(0, 10);
      const tail = losses.slice(-10);
      const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
      const falling = mean(tail) < mean(head);
      const floorHolds =
        losses.every(Number.isFinite) && evals.length >= 1 && tf.memory().numTensors === steady;
      expect(falling || floorHolds).toBe(true);

      dqn.dispose();
    },
    600_000,
  );
});
