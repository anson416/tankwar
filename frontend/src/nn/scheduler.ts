// scheduler.ts — the self-play fleet driver: `lanes` lockstep arenas ticked in
// lockstep, one fleet-wide behaviour forward per tick, per-tank per-tick
// transitions into the shared replay ring, ε decay per completed episode, the
// bounded gradient drain, and the greedy-vs-random eval. tfjs-FREE BY
// CONTRACT: the learner is reached through the TrainingNet interface below, so
// node vitest can run this whole loop against a scriptable fake net with zero
// @tensorflow weight in the module graph. The one import from nn/dqn is
// type-only (erased at compile time — verbatimModuleSyntax guarantees it).
//
// TRANSITION CONVENTION (per-tick, documented because the replay's `done`
// semantics rest on it): a transition completes ONE TICK after its action, and
// carries ITS OWN reward. Tick t encodes every ALIVE tank, acts, steps, and
// stores {s: obs(t), a, r} as the tank's pending — r is read at STAGING time,
// immediately after step(t) refilled lastRewards, so it is the payoff of
// (obs(t), a_t) itself (lastRewards is a fresh array every step; re-reading it
// a tick later would pair a_{t−1}'s row with a_t's reward — one-tick-shifted
// credit for every kill, death, pickup and damage event in the run). Tick t+1
// completes the pending with s′ = obs(t+1), done = 0. A TERMINAL decision —
// the tank died on tick t or the episode ended there — never pends: it flushes
// IMMEDIATELY as (obs(t), a_t, r, obs(t), 1), so the −60 death, the +50 kill
// and the timeout placement bonus land on exactly the action that earned them
// (done stops the bootstrap; s′ = s satisfies the ring's full-row requirement).
// A dead tank holds no pending and is never encoded again.
//
// WHY double-buffered obs slots (two Float32Array per lane-tank, chosen by
// game.tick parity): the pending transition's `s` must stay valid for one full
// tick, but encodeObs zero-fills and rewrites its target. A single reused
// buffer would be overwritten by the very encode that completes the pending;
// copying on push would instead allocate 282 floats × 4 tanks × 16 lanes ×
// 30 Hz ≈ 2 MB/s of nursery garbage. Parity gives each observation one tick
// of grace: tick t writes slot t&1 while tick t−1's pending reads slot
// (t−1)&1, and by the time slot t&1 comes round again the row is already
// memcpy'd into the ring.
//
// WORKER WIRING (what train.worker.ts does with a Trainer — one home for the
// contract so the two files cannot drift):
//   const dqn = new Dqn(cfg);
//   const trainer = new Trainer(cfg, dqn, new ReplayBuffer(cfg.bufferSize));
//   trainer.onTrainStep = (l) => { trainSteps++; smoothedLoss = smoothedLoss*0.95 + l*0.05; };
//   per iteration: await trainer.fleetTick();
//     if (dqn.steps - lastSync >= cfg.targetSync) { dqn.syncTarget(); lastSync = dqn.steps; }
//     if (trainer.episodes >= nextEvalAt) nextEvalAt = trainer.episodes + cfg.evalEvery;
//       → await trainer.runEval(seed) and read .winRate etc.
//     push trainer.metricsSnapshot({ backend, gamesPerSec }) at ≤5 Hz
//   reconfigure: build net+buffer+Trainer into LOCALS → dispose the old net →
//     await newTrainer.reset(seed) → trainer.rebuildGen++.
//   Generation guard is the CALLER's job: read trainer.rebuildGen before
//   `await trainer.runEval(...)` and drop the result if it moved mid-await —
//   runEval itself snapshots/ restores weights but never touches rebuildGen.

import { ACTION_COUNT } from "../game/actions";
import { createGame, step } from "../game/engine";
import { encodeObs, OBS_DIM } from "../game/encoding";
import { MAX_BULLETS_PER_TANK } from "../game/constants";
import { potentialApproach, potentialDanger } from "../game/rewards";
import { randInt } from "../game/rng";
import type { GameState, Tank } from "../game/types";
import type { ActMode } from "./dqn";
import type { ReplayBatch } from "./replay";
import { ReplayBuffer } from "./replay";
import type { Metrics, NetConfig } from "../train-protocol";
import { zeroMetrics } from "../train-protocol";

/**
 * The tfjs-free learner contract the Trainer drives. Dqn satisfies it
 * structurally: tf.Tensor[] snapshots are unknown-compatible, and the `train`
 * alias on Dqn fuses the method names. `steps` is a readonly number (Dqn
 * exposes it as a getter) so a caller reads the counter without reaching into
 * the learner's internals.
 */
export interface TrainingNet {
  act(
    obs: Float32Array[],
    fireAvail: boolean[],
    mode: ActMode,
    rngState: number,
  ): Promise<{ actions: Int32Array; state: number }>;
  /** One gradient step on a sampled batch; null when it declined to train. */
  train(batch: ReplayBatch): Promise<number | null>;
  readonly steps: number;
  syncTarget(): void;
  /** Opaque weight snapshot — opaque HERE so this module stays tfjs-free. */
  snapshot(): unknown;
  restore(s: unknown): void;
  disposeSnapshot(s: unknown): void;
}

/** An action decision awaiting its successor state: the pre-action obs, the
 *  action chosen, the reward captured at staging time (the payoff THIS action
 *  produced — see the TRANSITION CONVENTION above for why it cannot be
 *  re-read at flush time), and the tank (id == array index; the engine never
 *  reorders or removes tanks, corpses persist in place). */
export type Transition = { s: Float32Array; a: number; r: number; tank: number };

/** One self-play lane. `game` is public and mutable on purpose — the eval
 *  bookkeeping and tests read tank state straight off it. prevPhi/prevPsi are
 *  the per-tank potentials as of the LAST completed tick, live telemetry for a
 *  lane visualizer/metrics read; they are NOT re-added to rewards (the
 *  engine's shapedStep already folded the same potentials into lastRewards —
 *  re-adding them would double-count the shaping). */
export interface Lane {
  id: number;
  game: GameState;
  pending: (Transition | null)[];
  prevPhi: number[];
  prevPsi: number[];
}

/** Rolling-window size for the trainer-owned rate metrics (plan: last 200). */
const ROLL_WINDOW = 200;
/**
 * Gradient steps per fleetTick — the plan's "≤2 gradient steps" scheduler
 * priority. Bounding the drain keeps a tick bounded: the fleet forward, the
 * lane steps and the message pump never wait behind a deep replay backlog.
 */
const TRAIN_STEPS_PER_TICK = 2;
/**
 * Eval yields a real macrotask at least this often (one sim second) — see
 * runEval's BOUNDED SLICES note: on wasm/cpu the awaited tensor read inside
 * act resolves in a microtask, so only an explicit setTimeout lets the tick
 * chain, the lanes and the message queue run during an eval.
 */
const EVAL_YIELD_TICKS = 30;

/** One real macrotask turn (setTimeout, not a microtask await). */
function yieldMacrotask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
/** Action index for dead tanks' slots: decode(2) = coast/straight/hold. */
const COAST = 2;
/** Lane seed stride: coprime-ish odd prime so lane i's arena layout and the
 *  episode-to-episode sequence never fall into lockstep with each other. */
const LANE_SEED_STRIDE = 7919;

/** Protagonist-side eval result — the numbers the train tab's tiles chart. */
export interface EvalResult {
  winRate: number;
  avgReturn: number;
  survivalRate: number;
  killDeath: number;
}

/** True when the engine would accept this tank's fire request (per-tank half
 *  of tryFire's gate; the global bullet cap is deliberately NOT part of the
 *  behaviour mask — it is transient arena state, not the tank's own readiness). */
function canFire(t: Tank): boolean {
  return t.reload <= 0 && t.bulletsInFlight < MAX_BULLETS_PER_TANK;
}

/**
 * Winner of a finished game, or null for a mutual-destruction draw. Mirrors
 * rewards.placementBonus's ranking (alive first, then hp desc, ties to the
 * lowest id via the strict >) so eval's win-rate and the training reward pay
 * the SAME agent for the same finish.
 */
function winnerOf(game: GameState): number | null {
  let best: Tank | null = null;
  for (const t of game.tanks) {
    if (!t.alive) continue;
    if (best === null || t.hp > best.hp) best = t;
  }
  return best ? best.id : null;
}

interface RollRow {
  /** Mean per-tank undiscounted episode return. */
  ret: number;
  kills: number;
  deaths: number;
  /** Tanks alive at the end (survivors of an elimination OR a timeout). */
  alive: number;
  tanks: number;
}

export class Trainer {
  lanes: Lane[] = [];
  /** Completed episodes across all lanes (rebuilds reset it). */
  episodes = 0;
  /** Action selections seen fleet-wide — the trainEvery debt counter. */
  decisions = 0;
  /** Current exploration rate; decays once per COMPLETED episode. */
  epsilon = 1;
  /**
   * Rebuild generation. Owned by the CALLER (the worker bumps it on
   * transactional rebuild and compares around awaited evals); the Trainer
   * never reads or writes it.
   */
  rebuildGen = 0;
  /**
   * Worker hook, fired after every gradient step with its Huber loss — the
   * worker's channel for trainSteps/smoothedLoss (fields metricsSnapshot
   * deliberately leaves alone). Null (default) = no-op.
   */
  onTrainStep: ((loss: number) => void) | null = null;

  private readonly cfg: NetConfig;
  private readonly net: TrainingNet;
  private readonly buffer: ReplayBuffer;
  /** Episode-seed base from reset(); lanes re-deal off baseSeed + stride. */
  private baseSeed = 0;
  /** Threaded mulberry32 counter: act draws, replay sampling. No Math.random. */
  private rng = 0;
  /** Decisions already retired by gradient steps (the paid-down debt mark). */
  private trainMark = 0;
  /** Per-lane episode return accumulator (reset on lane re-deal). */
  private laneReturn: number[] = [];
  /** Double-buffered obs scratch: [lane][tank][tick parity]. */
  private obsBufs: Float32Array[][][] = [];
  /** Last ROLL_WINDOW episodes — the trainer-owned rate metrics. */
  private roll: RollRow[] = [];

  constructor(cfg: NetConfig, net: TrainingNet, buffer: ReplayBuffer) {
    this.cfg = cfg;
    this.net = net;
    this.buffer = buffer;
  }

  /**
   * (Re)deal every lane from `seed` and zero the counters: lanes, pendings,
   * episodes, decisions, ε → epsStart, the gradient debt, the rolling window
   * and the per-lane return accumulators. The replay buffer is NOT cleared —
   * it is the caller's object and a rebuild hands in a fresh one. The lane
   * seed formula (seed + id·7919 + episodes) is the same at reset and at every
   * re-deal, so lane k's episode sequence is reproducible from (seed, k).
   */
  async reset(seed: number): Promise<void> {
    this.baseSeed = seed | 0;
    // Mix before first use (engine.createGame's precedent): the act/sampling
    // stream must not share its whole draw sequence with the lane-seed ladder.
    this.rng = (this.baseSeed ^ 0x9e3779b9) | 0;
    this.episodes = 0;
    this.decisions = 0;
    this.epsilon = this.cfg.epsStart;
    this.trainMark = 0;
    this.roll.length = 0;
    this.lanes = [];
    this.laneReturn = [];
    this.obsBufs = [];
    for (let id = 0; id < this.cfg.lanes; id++) {
      this.lanes.push(this.dealLane(id));
      this.laneReturn.push(0);
      const perTank: Float32Array[][] = [];
      for (let t = 0; t < this.tankCount(); t++) {
        perTank.push([new Float32Array(OBS_DIM), new Float32Array(OBS_DIM)]);
      }
      this.obsBufs.push(perTank);
    }
  }

  /**
   * ONE lockstep tick across the fleet: encode every ALIVE tank (dead tanks
   * hold no opinion — a corpse cannot act), ONE batched act over all rows,
   * one engine step per lane, then complete last tick's pendings and stage the
   * next ones. A lane whose episode just ended decays ε, banks its rolling
   * stats and is re-dealt in the SAME call, so every lane is always live at
   * the next entry. Closes with the bounded gradient drain (≤
   * TRAIN_STEPS_PER_TICK steps, only while trainEvery debt is due AND the
   * buffer can fill a full batch — before that the debt simply accrues).
   */
  async fleetTick(): Promise<void> {
    // 1 — encode: rows are grouped lane-by-lane, tank order within a lane.
    const rows: Float32Array[] = [];
    const fire: boolean[] = [];
    const rowLane: number[] = [];
    const rowTank: number[] = [];
    for (const lane of this.lanes) {
      const slot = lane.game.tick & 1;
      lane.game.tanks.forEach((tank, i) => {
        if (!tank.alive) return;
        const buf = this.obsBufs[lane.id][i][slot];
        encodeObs(lane.game, tank.id, buf);
        rows.push(buf);
        fire.push(canFire(tank));
        rowLane.push(lane.id);
        rowTank.push(i);
      });
    }

    // 2 — ONE fleet forward for the whole tick (≤ lanes·tankCount rows).
    let actions: Int32Array = new Int32Array(0);
    if (rows.length > 0) {
      const res = await this.net.act(rows, fire, { kind: "explore", epsilon: this.epsilon }, this.rng);
      actions = res.actions;
      this.rng = res.state;
      this.decisions += rows.length;
    }

    // 3 — step every lane exactly once: the lockstep property. The per-lane
    // action vectors are kept for stage 4, which stages them into pendings.
    const preTick: number[] = [];
    const laneActs: number[][] = [];
    for (const lane of this.lanes) {
      const acts = lane.game.tanks.map(() => COAST); // dead tanks coast
      for (let r = 0; r < rowLane.length; r++) {
        if (rowLane[r] === lane.id) acts[rowTank[r]] = actions[r];
      }
      laneActs.push(acts);
      preTick.push(lane.game.tick);
      step(lane.game, acts);
    }

    // 4 — complete the pendings staged last tick, stage the next ones.
    for (let l = 0; l < this.lanes.length; l++) {
      const lane = this.lanes[l];
      const over = lane.game.over;
      const slot = preTick[l] & 1; // where THIS tick's obs was written
      lane.game.tanks.forEach((tank, i) => {
        // r is THIS tick's own payoff, read right after step(t) refilled
        // lastRewards: carrying it in the pending is what keeps the stored row
        // (s_t, a_t, r_t, s_{t+1}) — re-reading lastRewards at flush time
        // would hand a_{t−1}'s row the reward of a_t (engine.ts replaces the
        // array every step).
        const r = lane.game.lastRewards[i];
        this.laneReturn[lane.id] += r;
        const obs = this.obsBufs[lane.id][i][slot];
        const pending = lane.pending[i];
        if (pending) {
          // s′ = the just-encoded obs — the state the pending's action produced.
          this.buffer.push(pending.s, pending.a, pending.r, obs, 0);
        }
        // `acted` = this tank was encoded (alive) at stage 1 and drove this
        // tick's action: either it is still alive, or it died mid-tick
        // (diedThisTick). A tank dead BEFORE the tick holds no pending and
        // must not emit a phantom terminal row on the timeout tick.
        const acted = tank.alive || tank.diedThisTick;
        if (acted && tank.alive && !over) {
          lane.pending[i] = { s: obs, a: laneActs[l][i], r, tank: i };
        } else {
          if (acted) {
            // Terminal decision — flush NOW with its own reward and done=1
            // (s′ = s; done is what stops the bootstrap). The fatal action,
            // and the survivor's final action on a timeout, get their own
            // row instead of folding the −60/+50/placement payoff into the
            // PREVIOUS decision's transition.
            this.buffer.push(obs, laneActs[l][i], r, obs, 1);
          }
          lane.pending[i] = null;
        }
        // Potentials as-of-now (telemetry only — see Lane's header).
        lane.prevPhi[i] = potentialDanger(lane.game, tank.id);
        lane.prevPsi[i] = potentialApproach(lane.game, tank.id);
      });
      if (over) this.finishEpisode(lane);
    }

    // 5 — bounded gradient drain. Each step retires exactly trainEvery
    // decisions of debt and only runs while the buffer can fill a whole
    // batch; the mark advances on a step that RAN, so an unfillable batch
    // retries next tick instead of swallowing the trigger.
    let steps = 0;
    while (
      steps < TRAIN_STEPS_PER_TICK &&
      this.decisions - this.trainMark >= this.cfg.trainEvery &&
      this.buffer.size >= this.cfg.batchSize
    ) {
      this.trainMark += this.cfg.trainEvery;
      const { batch, state } = this.buffer.sample(this.cfg.batchSize, this.rng);
      this.rng = state;
      const loss = await this.net.train(batch);
      if (loss === null) {
        // Unreachable behind the size guard (a non-empty batch is trainable);
        // restoring the mark keeps the accounting exact if that ever changes.
        this.trainMark -= this.cfg.trainEvery;
        break;
      }
      steps++;
      this.onTrainStep?.(loss);
    }
  }

  /**
   * Score the CURRENT weights over evalEpisodes greedy-vs-random games: tank
   * 0 (the protagonist) acts ε=0-greedy through the net; every other tank
   * draws uniform over all 18 (the same distribution act's ε=1 explore path
   * produces, without burning a wasted forward per opponent per tick).
   *
   * Snapshot discipline — cloned before the first game, DISPOSED always,
   * restored only on the error path: nothing ever writes "eval weights" into
   * the live net (act is forward-only), so a restore after a CLEAN run would
   * be a pure rollback — it would discard every gradient step the tick chain
   * applied while the eval awaited its tensor reads, while stepCount,
   * trainMark, the replay and Adam's m/v all keep their post-eval values
   * (optimizer second moments describing weights that no longer exist). A
   * crash is the one state a rollback heals, and even that is skipped when
   * the generation moved (a rebuild disposed the live net under us — the
   * restore would throw before the dispose ever ran), so the dispose sits in
   * its own finally: the snapshot clone is released on EVERY path. Eval
   * touches NO training state: no buffer pushes, no ε decay, no episode
   * counters. Episodes are the config's own length (episodeSeconds,
   * hard-capped by the engine), and every game is a pure function of
   * (evalSeed, e) — reproducible, no Math.random.
   *
   * BOUNDED SLICES: the loop yields a real macrotask between games and every
   * EVAL_YIELD_TICKS ticks inside a game. On the wasm/cpu tiers act's awaited
   * read resolves in a MICROTASK (the backend's read() is synchronous), so
   * without an explicit setTimeout the whole eval — 24 × 1350 batch-1
   * forwards — would run as ONE macrotask and freeze the tick chain, the
   * interactive lanes and the message queue for seconds (the worker header's
   * "the eval yields between tensor reads" is a contract this loop keeps).
   */
  async runEval(evalSeed: number): Promise<EvalResult> {
    const snap = this.net.snapshot();
    const gen = this.rebuildGen;
    let failed = false;
    try {
      let wins = 0;
      let survived = 0;
      let ret = 0;
      let kills = 0;
      let deaths = 0;
      for (let e = 0; e < this.cfg.evalEpisodes; e++) {
        await yieldMacrotask();
        const game = createGame({
          tankCount: this.tankCount(),
          teamMode: "ffa",
          episodeSeconds: this.cfg.episodeSeconds,
          gamma: this.cfg.gamma, // eval rewards score the SAME shaping the trainer sees
          seed: (evalSeed + e * LANE_SEED_STRIDE) | 0,
        });
        const obs = new Float32Array(OBS_DIM);
        // Opponent stream seeded per game (fresh layout AND fresh draws); the
        // protagonist's act state is threaded back so even the ε=0 threshold
        // draws stay on the record.
        let rng = (evalSeed + e * 104729) | 0;
        let sinceYield = 0;
        while (!game.over) {
          const me = game.tanks[0];
          const res = me.alive
            ? await this.net.act(
                [encodeObs(game, me.id, obs)],
                [canFire(me)],
                { kind: "explore", epsilon: 0 },
                rng,
              )
            : null;
          if (res) rng = res.state;
          const acts = game.tanks.map((_t, i) => {
            if (i === 0) return res ? res.actions[0] : COAST; // a dead hero coasts
            const [a, next] = randInt(rng, ACTION_COUNT);
            rng = next;
            return a;
          });
          step(game, acts);
          ret += game.lastRewards[0];
          if (++sinceYield >= EVAL_YIELD_TICKS) {
            sinceYield = 0;
            await yieldMacrotask();
          }
        }
        if (winnerOf(game) === 0) wins++;
        if (game.tanks[0].alive) survived++;
        kills += game.tanks[0].kills;
        deaths += game.tanks[0].deaths;
      }
      const games = Math.max(1, this.cfg.evalEpisodes);
      return {
        winRate: wins / games,
        avgReturn: ret / games,
        survivalRate: survived / games,
        killDeath: kills / Math.max(1, deaths),
      };
    } catch (err) {
      failed = true;
      throw err;
    } finally {
      try {
        if (failed && gen === this.rebuildGen) this.net.restore(snap);
      } finally {
        this.net.disposeSnapshot(snap); // guaranteed, even when restore throws
      }
    }
  }

  /**
   * The trainer-owned slice of the metrics push. zeroMetrics() is the base:
   * trainSteps, smoothedLoss, winRateVsRandom and ts stay at their defaults
   * because the WORKER owns them (losses arrive through onTrainStep, win-rate
   * through runEval, ts is the push moment) — the worker overwrites them on
   * the returned object before posting.
   */
  metricsSnapshot(extra: { backend: string; gamesPerSec: number }): Metrics {
    const m = zeroMetrics();
    m.episodes = this.episodes;
    m.decisions = this.decisions;
    m.epsilon = this.epsilon;
    m.bufferFill = this.buffer.fillFraction();
    let alive = 0;
    for (const lane of this.lanes) if (!lane.game.over) alive++;
    m.aliveLanes = alive;
    let ret = 0;
    let kills = 0;
    let deaths = 0;
    let surv = 0;
    for (const r of this.roll) {
      ret += r.ret;
      kills += r.kills;
      deaths += r.deaths;
      surv += r.alive / r.tanks;
    }
    const n = this.roll.length;
    m.avgReturn = n > 0 ? ret / n : 0;
    // 0 deaths reads as 1: a spotless episode is a finite K/D, not infinity.
    m.killDeathRatio = kills / Math.max(1, deaths);
    m.survivalRate = n > 0 ? surv / n : 0;
    m.backend = extra.backend;
    m.gamesPerSec = extra.gamesPerSec;
    return m;
  }

  // --- internals ---------------------------------------------------------------

  /** Fresh episode for lane `id`, seeded by the shared formula (see reset). */
  private dealLane(id: number): Lane {
    const game = createGame({
      tankCount: this.tankCount(),
      teamMode: "ffa", // training is always FFA (plan, confirmed requirement)
      episodeSeconds: this.cfg.episodeSeconds,
      // The shaping γ rides with the episode so it can never drift from the
      // net that trains on these rewards (rewards.ts / GameConfig.gamma).
      gamma: this.cfg.gamma,
      seed: (this.baseSeed + id * LANE_SEED_STRIDE + this.episodes) | 0,
    });
    return {
      id,
      game,
      pending: new Array<Transition | null>(game.tanks.length).fill(null),
      prevPhi: new Array<number>(game.tanks.length).fill(0),
      prevPsi: new Array<number>(game.tanks.length).fill(0),
    };
  }

  /** NetConfig.tankCount is a free number; the arena contract is 2–4. */
  private tankCount(): 2 | 3 | 4 {
    const n = Math.round(this.cfg.tankCount);
    return n < 3 ? 2 : n > 3 ? 4 : 3;
  }

  /**
   * Bank one finished episode: rolling stats, ε decay, re-deal. Called once
   * PER over lane, so two lanes finishing in the same tick decay ε twice —
   * exactly "once per completed episode" (ε = max(epsMin, epsStart·epsDecay^episodes)).
   */
  private finishEpisode(lane: Lane): void {
    let kills = 0;
    let deaths = 0;
    let alive = 0;
    for (const t of lane.game.tanks) {
      kills += t.kills;
      deaths += t.deaths;
      if (t.alive) alive++;
    }
    this.roll.push({
      ret: this.laneReturn[lane.id] / lane.game.tanks.length,
      kills,
      deaths,
      alive,
      tanks: lane.game.tanks.length,
    });
    if (this.roll.length > ROLL_WINDOW) this.roll.shift();
    this.laneReturn[lane.id] = 0;
    this.episodes++;
    this.epsilon = Math.max(this.cfg.epsMin, this.epsilon * this.cfg.epsDecay);
    this.lanes[lane.id] = this.dealLane(lane.id);
  }
}
