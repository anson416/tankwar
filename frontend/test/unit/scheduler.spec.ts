// scheduler.spec.ts — the Trainer's loop contract, driven by a SCRIPTABLE FAKE
// net: zero @tensorflow anywhere in this module's import graph (Trainer must
// stay runnable in pure node — that is the point of the TrainingNet seam).
// Asserts the properties the learner's data quality depends on: lanes advance
// in lockstep (one engine tick per fleetTick, always), transitions complete
// exactly one tick after their action with done=1 landing on precisely the
// tick the tank dies (or the episode ends), ε decays once per completed
// episode and floors at epsMin, the gradient drain waits for a full batch and
// then pays trainEvery debt with full-size batches, eval drives only the
// protagonist and only greedily on a snapshot it restores, and metricsSnapshot
// carries exactly the fields the trainer owns.
//
// The one non-trivial fixture is the aim policy: reading the nearest-tank
// bearing straight out of the encoded obs (the same rows a real net would
// consume) and steering/firing on it. Rationale: a straight-line scripted bot
// almost never lands a hit in this engine (probe: ~1 hit per 1350-tick game),
// and the done=1-on-death convention is the thing under test — the aim policy
// produces a mid-episode elimination deterministically on every seed tried.
import { describe, expect, it } from "vitest";
import { TICK_HZ, NORM_DIST } from "../../src/game/constants";
import { ACTION_COUNT } from "../../src/game/actions";
import { OFF } from "../../src/game/encoding";
import { ReplayBuffer } from "../../src/nn/replay";
import type { ReplayBatch } from "../../src/nn/replay";
import { Trainer } from "../../src/nn/scheduler";
import type { TrainingNet } from "../../src/nn/scheduler";
import type { ActMode } from "../../src/nn/dqn";
import { defaultConfig } from "../../src/train-protocol";
import type { Metrics, NetConfig } from "../../src/train-protocol";

// --- fixtures -----------------------------------------------------------------

/** Steer at the nearest tank's bearing (TANKS slot 0), close in, always pull
 *  the trigger — the deterministic elimination generator described above. */
function aimPolicy(obs: Float32Array): number {
  const bearing = obs[OFF.TANKS + 2];
  const dist = obs[OFF.TANKS + 1] * NORM_DIST;
  const steer = Math.abs(bearing) < 0.04 ? 1 : bearing > 0 ? 2 : 0;
  const throttle = dist < 100 ? 0 : 1;
  return (throttle * 3 + steer) * 2 + 1;
}

interface ModeLog {
  kind: ActMode["kind"];
  epsilon: number;
  tau: number;
  rows: number;
}

class FakeNet implements TrainingNet {
  calls = 0;
  modes: ModeLog[] = [];
  trainCalls = 0;
  losses: number[] = [];
  /** First few batches only — retention is capped so a long run cannot bloat. */
  batches: ReplayBatch[] = [];
  snapshots = 0;
  restores = 0;
  disposes = 0;
  private readonly policy: (obs: Float32Array, row: number) => number;

  constructor(policy: (obs: Float32Array, row: number) => number = () => 2) {
    this.policy = policy;
  }

  async act(
    obs: Float32Array[],
    _fireAvail: boolean[],
    mode: ActMode,
    rngState: number,
  ): Promise<{ actions: Int32Array; state: number }> {
    this.calls++;
    this.modes.push({
      kind: mode.kind,
      epsilon: mode.kind === "explore" ? mode.epsilon : Number.NaN,
      tau: mode.kind === "temperature" ? mode.tau : Number.NaN,
      rows: obs.length,
    });
    const actions = new Int32Array(obs.length);
    for (let i = 0; i < obs.length; i++) actions[i] = this.policy(obs[i], i);
    return { actions, state: rngState };
  }

  async train(b: ReplayBatch): Promise<number | null> {
    this.trainCalls++;
    if (this.batches.length < 16) this.batches.push(b);
    this.losses.push(0.5);
    return 0.5;
  }

  get steps(): number {
    return this.trainCalls;
  }

  syncTarget(): void {
    /* nothing to sync on a fake */
  }

  snapshot(): unknown {
    this.snapshots++;
    return { tag: "fake-snap" };
  }

  restore(_s: unknown): void {
    this.restores++;
  }

  disposeSnapshot(_s: unknown): void {
    this.disposes++;
  }
}

/** Recording ring: the transition tests read back the (r, done) columns the
 *  Trainer pushed, without reaching into ReplayBuffer's private bytes. */
class RecordingBuffer extends ReplayBuffer {
  pushed: { r: number; done: number }[] = [];
  push(s: Float32Array, a: number, r: number, s2: Float32Array, done: number): void {
    this.pushed.push({ r, done });
    super.push(s, a, r, s2, done);
  }
}

function tinyCfg(over: Partial<NetConfig> = {}): NetConfig {
  return {
    ...defaultConfig(),
    architecture: "mlp",
    hidden: [8],
    lanes: 2,
    tankCount: 2,
    episodeSeconds: 2, // 60 ticks — short enough to march through by hand
    batchSize: 8,
    trainEvery: 4,
    bufferSize: 256,
    epsStart: 0, // the fake net owns behaviour; ε stays 0 in most tests
    epsDecay: 0.5, // exact binary powers → toBe (not closeTo) is meaningful
    epsMin: 0.05,
    evalEvery: 1e9, // eval is exercised explicitly, never by the counter
    evalEpisodes: 2,
    targetSync: 1000,
    ...over,
  };
}

describe("lockstep + transition convention", () => {
  it("advances every lane exactly one tick per fleetTick", async () => {
    const trainer = new Trainer(tinyCfg(), new FakeNet(), new ReplayBuffer(256));
    await trainer.reset(11);
    for (let k = 0; k < 10; k++) {
      const before = trainer.lanes.map((l) => l.game.tick);
      await trainer.fleetTick();
      trainer.lanes.forEach((l, i) => expect(l.game.tick).toBe(before[i]! + 1));
    }
    expect(trainer.lanes).toHaveLength(2);
  });

  it("completes each transition one tick after its action; nothing is terminal without a death or a timeout", async () => {
    const buf = new RecordingBuffer(256);
    const trainer = new Trainer(tinyCfg(), new FakeNet(), buf);
    await trainer.reset(7);
    // Coast-only: no movement, no shells, no deaths — the pure plumbing case.
    for (let k = 0; k < 10; k++) await trainer.fleetTick();
    expect(trainer.decisions).toBe(10 * 2 * 2); // ticks × lanes × tanks
    // Every decision is flushed except the four pendings still staged at the
    // end (one per lane-tank) — i.e. each completed exactly one tick later.
    expect(buf.pushed).toHaveLength(trainer.decisions - 4);
    for (const p of buf.pushed) {
      expect(p.done).toBe(0);
      expect(p.r).toBeGreaterThan(0); // survival +0.02 dominates the (1−γ)Ψ drift
    }
  });

  it("flushes done=1 on exactly the tick a tank dies or the episode ends", async () => {
    const buf = new RecordingBuffer(4096);
    // One lane, three tanks, 10-second episodes: the aim policy kills tank 0
    // mid-episode on every consecutive seed (probed 100..115) while the
    // survivors keep playing — the mid-episode death case the coast fixtures
    // cannot reach — and the survivors still flush at the timeout.
    const cfg = tinyCfg({
      lanes: 1,
      tankCount: 3,
      episodeSeconds: 10,
      bufferSize: 4096,
      epsStart: 0,
    });
    const trainer = new Trainer(cfg, new FakeNet(aimPolicy), buf);
    await trainer.reset(100);
    const EPISODES = 4;
    let guard = 0;
    let done1 = 0;
    let midDeathTicks = 0;
    let eliminated = 0;
    while (trainer.episodes < EPISODES && guard++ < 4000) {
      // Per-lane view BEFORE the tick: alive flags and the lane object itself
      // (a replaced Lane object means that lane's episode just finished, and
      // the OLD lane still holds the final game state).
      const beforeLanes = trainer.lanes.slice();
      const beforeAlive = beforeLanes.map((l) => l.game.tanks.map((t) => t.alive));
      const start = buf.pushed.length;
      await trainer.fleetTick();
      let expectDone1 = 0;
      for (let i = 0; i < beforeLanes.length; i++) {
        const oldLane = beforeLanes[i]!;
        const ended = oldLane !== trainer.lanes[i];
        const tanks = ended ? oldLane.game.tanks : trainer.lanes[i]!.game.tanks;
        let newlyDead = 0;
        let aliveAtEnd = 0;
        tanks.forEach((t, j) => {
          if (beforeAlive[i]![j] && !t.alive) newlyDead++;
          if (t.alive) aliveAtEnd++;
        });
        if (ended) {
          expectDone1 += newlyDead + aliveAtEnd;
          if (aliveAtEnd >= 2 && tanks.length - aliveAtEnd >= 1) midDeathTicks++;
          if (aliveAtEnd <= 1) eliminated++;
        } else {
          expectDone1 += newlyDead;
        }
      }
      const flushed = buf.pushed.slice(start);
      done1 += flushed.filter((p) => p.done === 1).length;
      // THE convention, tick by tick: exactly the deaths (plus the survivors
      // of a lane that ended) flush terminal on this tick — no earlier, no later.
      expect(flushed.filter((p) => p.done === 1)).toHaveLength(expectDone1);
    }
    expect(trainer.episodes).toBe(EPISODES);
    expect(eliminated).toBeGreaterThan(0); // eliminations really happened
    expect(midDeathTicks).toBeGreaterThan(0); // deaths while others played on
    // Every tank's LAST transition of an episode is done=1 — dead ones on
    // their death tick, survivors on the terminal tick — so the count is
    // exactly tankCount per episode across eliminations AND timeouts alike.
    expect(done1).toBe(EPISODES * cfg.tankCount);
    // Conservation of decisions: EVERY decision flushes exactly one replay
    // row — the plan's (s, a, r, s′, done) transition is per decision, so a
    // tank's FINAL action of an episode (taken on the tick it dies, or on the
    // timeout) pushes its OWN done=1 row rather than folding its payoff into
    // the previous decision's row. The only rows not yet pushed are the
    // pendings still staged in the lanes (zero here: the loop exited on a
    // re-deal tick), which is why the done=1 count above is exactly
    // tankCount per episode.
    const outstanding = trainer.lanes.reduce(
      (n, l) => n + l.pending.filter((p) => p !== null).length,
      0,
    );
    expect(buf.pushed).toHaveLength(trainer.decisions - outstanding);
    // A death penalty (−60, plus −1.5/HP of damage taken) reached the buffer.
    expect(buf.pushed.some((p) => p.r < -20)).toBe(true);
  });
});

describe("epsilon schedule", () => {
  it("decays once per completed episode and floors at epsMin", async () => {
    const cfg = tinyCfg({ epsStart: 1, epsDecay: 0.5, epsMin: 0.125 });
    const trainer = new Trainer(cfg, new FakeNet(), new ReplayBuffer(256));
    await trainer.reset(3);
    expect(trainer.epsilon).toBe(1);
    // Coast-only episodes end together at the 60-tick timeout.
    for (let k = 0; k < TICK_HZ * cfg.episodeSeconds - 1; k++) await trainer.fleetTick();
    expect(trainer.episodes).toBe(0);
    expect(trainer.epsilon).toBe(1); // no completed episode ⇒ no decay
    await trainer.fleetTick();
    expect(trainer.episodes).toBe(2); // both lanes timed out on this tick
    expect(trainer.epsilon).toBe(0.25); // 1 · 0.5² — once PER completed episode
    for (let e = 2; e < 8; e++) {
      for (let k = 0; k < TICK_HZ * cfg.episodeSeconds; k++) await trainer.fleetTick();
    }
    expect(trainer.episodes).toBe(2 + 6 * cfg.lanes);
    // 0.5^12 « 0.125: floored exactly, never below.
    expect(trainer.epsilon).toBe(cfg.epsMin);
    // …and the schedule formula is the whole story.
    expect(trainer.epsilon).toBe(
      Math.max(cfg.epsMin, cfg.epsStart * Math.pow(cfg.epsDecay, trainer.episodes)),
    );
  });
});

describe("gradient drain", () => {
  it("waits for a full batch, then pays trainEvery debt with full-size batches", async () => {
    const net = new FakeNet();
    const trainer = new Trainer(tinyCfg(), net, new ReplayBuffer(64));
    const hooked: number[] = [];
    trainer.onTrainStep = (l) => hooked.push(l);
    await trainer.reset(5);
    await trainer.fleetTick(); // 4 decisions, 0 pushes yet (pendings just staged)
    expect(net.batches).toHaveLength(0); // no rows at all, let alone a batch
    await trainer.fleetTick(); // 8 decisions, 4 pushes — debt 8 ≥ 4, buffer 4 < 8
    expect(net.batches).toHaveLength(0); // still no full batch: the debt waits
    await trainer.fleetTick(); // 8 rows in the ring, debt 12 → drain, tick cap 2
    expect(net.batches).toHaveLength(2);
    for (const b of net.batches) expect(b.size).toBe(8);
    await trainer.fleetTick(); // debt 16 − 8 = 8 → two more full-batch steps
    expect(net.batches).toHaveLength(4);
    expect(net.batches[3]!.size).toBe(8);
    // The worker hook fired once per gradient step, with the fake's loss.
    expect(hooked).toEqual([0.5, 0.5, 0.5, 0.5]);
  });
});

describe("runEval", () => {
  it("scores the protagonist greedy on a snapshot, touches no training state", async () => {
    const net = new FakeNet();
    const cfg = tinyCfg({ evalEpisodes: 2 });
    const trainer = new Trainer(cfg, net, new ReplayBuffer(64));
    await trainer.reset(9);
    for (let k = 0; k < 5; k++) await trainer.fleetTick();
    const seen = net.modes.length;
    const episodes = trainer.episodes;
    const decisions = trainer.decisions;
    const epsilon = trainer.epsilon;
    const res = await trainer.runEval(1234);
    // Every eval act call is the protagonist alone (one row), greedy: explore
    // with ε = 0. The uniform-random opponents never reach the net.
    const evalModes = net.modes.slice(seen);
    expect(evalModes.length).toBeGreaterThan(0);
    for (const m of evalModes) {
      expect(m.kind).toBe("explore");
      expect(m.epsilon).toBe(0);
      expect(m.rows).toBe(1);
    }
    // Snapshot weights around the whole run; the clone is disposed exactly
    // once. A CLEAN eval never restores: nothing writes "eval weights" into
    // the live net (act is forward-only), so a restore would roll back every
    // gradient step the tick chain applied during the eval window while
    // Adam's moments keep their post-eval values. Restore is the ERROR path's
    // rollback only (see runEval's header).
    expect(net.snapshots).toBe(1);
    expect(net.restores).toBe(0);
    expect(net.disposes).toBe(1);
    // Eval is read-only: no episode, decision or ε movement, no buffer growth.
    expect(trainer.episodes).toBe(episodes);
    expect(trainer.decisions).toBe(decisions);
    expect(trainer.epsilon).toBe(epsilon);
    expect(res.winRate).toBeGreaterThanOrEqual(0);
    expect(res.winRate).toBeLessThanOrEqual(1);
    expect(res.survivalRate).toBeGreaterThanOrEqual(0);
    expect(res.survivalRate).toBeLessThanOrEqual(1);
    expect(Number.isFinite(res.avgReturn)).toBe(true);
    expect(res.killDeath).toBeGreaterThanOrEqual(0);
  });
});

describe("metricsSnapshot", () => {
  it("fills the trainer-owned fields, echoes the extras, leaves the worker's alone", async () => {
    const trainer = new Trainer(tinyCfg(), new FakeNet(), new ReplayBuffer(64));
    await trainer.reset(21);
    for (let k = 0; k < 5; k++) await trainer.fleetTick();
    const m: Metrics = trainer.metricsSnapshot({ backend: "cpu", gamesPerSec: 12.5 });
    expect(m.episodes).toBe(0);
    expect(m.decisions).toBe(20);
    expect(m.epsilon).toBe(trainer.epsilon);
    expect(m.bufferFill).toBeCloseTo(16 / 64, 12); // 20 decisions − 4 staged
    expect(m.aliveLanes).toBe(2);
    expect(m.backend).toBe("cpu");
    expect(m.gamesPerSec).toBe(12.5);
    // Nothing has finished yet: the rolling window is empty and reads as zeros.
    expect(m.avgReturn).toBe(0);
    expect(m.survivalRate).toBe(0);
    expect(m.killDeathRatio).toBe(0);
    // Worker-owned fields stay at their zeroMetrics defaults on purpose.
    expect(m.trainSteps).toBe(0);
    expect(m.smoothedLoss).toBe(0);
    expect(m.winRateVsRandom).toBe(0);
    expect(m.ts).toBe(0);
  });

  it("rolling averages cover the last episodes: survival 1.0 on coast-only timeouts", async () => {
    const cfg = tinyCfg({ epsStart: 0 });
    const trainer = new Trainer(cfg, new FakeNet(), new ReplayBuffer(4096));
    await trainer.reset(33);
    const total = TICK_HZ * cfg.episodeSeconds;
    for (let k = 0; k < total; k++) await trainer.fleetTick();
    expect(trainer.episodes).toBe(cfg.lanes); // both lanes timed out together
    const m = trainer.metricsSnapshot({ backend: "wasm", gamesPerSec: 0 });
    expect(m.survivalRate).toBe(1); // coast-only: everyone survives to the bell
    expect(m.killDeathRatio).toBe(0); // no shells, no kills, no deaths
    // The hp-tie placement bonus alone is worth +20 per tank (40 / 0); the
    // survival and shaping terms move the total by well under a point either
    // way, so the mean must land just above 20 — proving the timeout bonus
    // reaches the rolling window rather than a vague magnitude.
    expect(m.avgReturn).toBeGreaterThan(20);
    expect(m.avgReturn).toBeLessThan(21);
    expect(m.aliveLanes).toBe(cfg.lanes); // re-dealt in the same tick
  });
});

// The action indices the fixtures ride on are real table rows, and
// ACTION_COUNT is the uniform-draw support the eval opponents sample.
describe("constants the loop leans on", () => {
  it("scripted actions are in range and the action count is 18", () => {
    for (const a of [2, 7, 9, 11]) expect(a).toBeLessThan(ACTION_COUNT);
    expect(ACTION_COUNT).toBe(18);
  });
});
