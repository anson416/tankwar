// train.worker.ts — the ENTIRE tfjs backend on one thread: the
// WebGPU→WebGL→WASM→CPU ladder, the DQN learner + replay ring + 16-lane
// self-play scheduler (nn/*), and the inference lanes for watch/play. The main
// thread ships zero @tensorflow/* (a unit-test grep enforces this) and never
// holds weights — only metrics, arena layouts, render snapshots and episode
// results cross postMessage.
//
// NOT EXECUTED UNDER VITEST: this module only runs in a real browser worker
// (backend ladder, self.location, postMessage). The spec suite covers the
// machinery it wires — dqn.spec/model.spec (nets), scheduler.spec/replay.spec
// (the training loop), engine.spec (the sim) — while typecheck plus the
// Playwright smoke suite cover the wiring itself (big-two precedent).
//
// SCHEDULER (big-two pattern): exactly ONE tick chain — setTimeout(0),
// re-armed in the finally, single-chain invariant held by
// tickScheduled/tickRunning (schedule only when neither is set; the tick's
// finally is the single point of continuation). Each iteration does BOUNDED
// work in the plan's pinned order (§Architecture — ORDER IS LAW):
//   (1) INTERACTIVE LANES FIRST. Watch and play each tick on their own
//       wall-clock accumulator, ≤4 catch-up ticks per iteration (spiral-of-
//       death cap), so a slow backend can never starve the mode a human is
//       looking at in favour of training throughput.
//   (2) TRAINING: ONE trainer.fleetTick() — the lockstep fleet slice, which
//       ALSO owns the bounded gradient drain (scheduler.ts stage 5 is the
//       worker's only train path: the worker never samples the replay itself,
//       it receives each Huber loss through trainer.onTrainStep and keeps the
//       trainSteps/loss-EMA counters there). Then the hard target sync
//       (cfg.targetSync, read off dqn.steps) and the fire-and-forget eval.
//   (3) One throttled metrics push (200 ms — the plan's ≤5 Hz).
// Queued messages are served between iterations (drain below); the eval yields
// between tensor reads; neither can wedge the lanes.
//
// DETERMINISM: every draw threads the explicit mulberry32 counter (game/rng) —
// lane deals, eval seeds, act draws and replay sampling all hang off
// SESSION_SEED, so a session is reproducible end-to-end and Math.random never
// appears. Only the session seed itself comes from the wall clock: two reloads
// must not train the identical episode sequence, but everything AFTER that one
// derivation is a pure function of threaded state.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu"; // static: the guaranteed floor tier
import { Dqn } from "../nn/dqn";
import type { ActMode } from "../nn/dqn";
import { Trainer } from "../nn/scheduler";
import { ReplayBuffer } from "../nn/replay";
import { createGame, step } from "../game/engine";
import { encodeAction } from "../game/actions";
import { encodeObs, OBS_DIM } from "../game/encoding";
import {
  BULLET_SPEED,
  EPISODE_SECONDS_DEFAULT,
  MAX_BULLETS_PER_TANK,
  TICK_HZ,
} from "../game/constants";
import { randInt } from "../game/rng";
import { toArenaMsg, toSnapshot } from "../game/snapshot";
import type { GameEvent, GameState, Tank } from "../game/types";
import {
  defaultConfig,
  isInputReq,
  isPauseReq,
  isPlayStartReq,
  isReconfigureReq,
  isStartReq,
  isStepReq,
  isStopReq,
  isWatchStartReq,
  isWatchStopReq,
} from "../train-protocol";
import type {
  EpisodeEndMsg,
  Metrics,
  NetConfig,
  TeamMode,
  WorkerReq,
  WorkerResp,
} from "../train-protocol";

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerReq>) => void) | null;
  postMessage: (msg: WorkerResp) => void;
};

// --- backend ladder (big-two, research-verified for tfjs 4.22) ---------------
//
// Lazy dynamic imports keep webgpu/webgl/wasm in separate chunks fetched only
// when their tier is reachable. setWasmPaths MUST run BEFORE the wasm factory
// initializes; the binaries are self-hosted by the copy script at
// <dist>/tfjs-wasm/ while THIS chunk is emitted to <dist>/assets/, so the
// correct URL is ../tfjs-wasm/ resolved against import.meta.url at runtime —
// that stays right under the relative Pages base regardless of sub-path.
// Probing uses engine().backendNames() (registered factories, no side effect);
// findBackend would FORCE an init attempt as a side effect. setBackend
// resolves false on factory failure (e.g. webgl inside Safari workers, where
// OffscreenCanvas is broken) — that means "try the next tier".

async function initBackend(): Promise<string> {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      await import("@tensorflow/tfjs-backend-webgpu");
    } catch {
      /* WebGPU glue unavailable in this worker — ladder falls through */
    }
  }
  try {
    await import("@tensorflow/tfjs-backend-webgl");
  } catch {
    /* no WebGL in this worker context */
  }
  try {
    const { setWasmPaths } = await import("@tensorflow/tfjs-backend-wasm");
    // @vite-ignore: the directory only exists in the BUILD OUTPUT (dist/),
    // never next to this source file — Vite must leave the URL literal alone
    // so the browser resolves it against the emitted worker chunk at runtime.
    setWasmPaths(new URL(/* @vite-ignore */ "../tfjs-wasm/", import.meta.url).href);
  } catch {
    /* wasm glue unavailable */
  }

  const ladder = ["webgpu", "webgl", "wasm", "cpu"] as const;
  for (const name of ladder) {
    if (!tf.engine().backendNames().includes(name)) continue;
    if (await tf.setBackend(name)) {
      await tf.ready();
      return name;
    }
  }
  throw new Error("no tfjs backend available — cpu failed to register (static import broken?)");
}

// --- constants ---------------------------------------------------------------

/** Yield to the event loop between iterations — serves the message queue. */
const TICK_DELAY_MS = 0;
/** Metrics throttle: the plan's ≤5 Hz progress push. */
const REPORT_THROTTLE_MS = 200;
/** Max engine ticks one lane may bank per iteration. A background tab stalls
 *  the worker clock; without this cap the resume would "catch up" minutes of
 *  simulation in one burst (spiral of death) instead of dropping the debt. */
const MAX_CATCHUP_TICKS = 4;
/** Beat between a finished watch episode and the next deal (big-two's watch
 *  precedent): long enough to read the scoreboard, short enough to feel live. */
const WATCH_RESTART_MS = 1200;
/** Watch policy temperature: the CURRENT training policy viewed near-greedily
 *  — τ = 0.05 keeps a sliver of softness so ties do not look robotic. */
const WATCH_TAU = 0.05;
/** Play-opponent temperature floor added to (1 − difficulty) — τ = 0.05 at
 *  maximum difficulty still samples, so play never freezes on an exact tie. */
const PLAY_TAU_FLOOR = 0.05;
/** Aim assist budget per tick, in radians (plan: ≤6° intercept snap). Deliberately
 *  just under the tank's own TURN_RATE·DT (3.6/30 ≈ 6.9°): the assist may never
 *  out-turn a deliberate steer. */
const AIM_ASSIST_STEP = (6 * Math.PI) / 180;
/** Two fixed-point iterations converge the intercept time-of-flight far below
 *  a tick of bullet travel (BULLET_SPEED·DT = 28 u); more buys nothing. */
const AIM_ASSIST_ITERATIONS = 2;
/** decode(2) = coast/straight/hold — the identity action for dead seats and a
 *  play lane whose first input packet has not arrived yet. */
const COAST_ACTION = 2;
/** One draw of the session seed, mixed so the lane/eval/training streams do
 *  not share their draw sequence with each other. */
const SESSION_SEED = (Date.now() | 0) ^ 0x9e3779b9;
/**
 * Boot warmup: the FIRST fleet forward and the FIRST gradient steps make the
 * backend JIT-compile their kernels — seconds on a software GL (CI's swiftshader)
 * — and mid-session that cost lands on the tick chain, freezing the interactive
 * lanes behind it (measured: a ~3.5 s watch-lane stall exactly when training
 * crossed its first trainEvery debt). A bounded handful of REAL fleet slices
 * during init pays the compilation while no lane can exist yet (their start
 * requests queue behind initPromise) and before the boot handshake flips
 * __engineReady. These are honest training ticks — same lanes, same replay,
 * same debt accounting — merely executed a few seconds earlier than the chain
 * would have; eight is the minimum that guarantees ≥1 gradient step for both
 * the default config (16 lanes × 4 tanks → batchSize 256 filled at tick 4) and
 * the ?test=1 fixture (2 × 2 → batchSize 16 filled at tick 4).
 */
const WARMUP_FLEET_TICKS = 8;

// --- state -------------------------------------------------------------------

let cfg: NetConfig = defaultConfig();
let dqn!: Dqn;
let trainer!: Trainer;
let backend = "cpu";
let initError: string | null = null;

let training = false;
/** Bumped by every rebuild — a mid-await eval/lane/restart checks it and drops
 *  stale work. Mirrored into trainer.rebuildGen (the caller-owned field the
 *  scheduler contract reserves for exactly this guard). */
let rebuildGen = 0;
/** Gradient steps applied (from trainer.onTrainStep) — the targetSync counter. */
let trainSteps = 0;
/** Huber loss EMA 0.95/0.05, seeded by the first step (plan §Training). */
let smoothedLoss = 0;
/** Last greedy-vs-random eval win rate (chance = 1/N on the dashboard). */
let winRateVsRandom = 0;
let lastSync = 0; // dqn.steps already synced into the target net
let nextEvalAt = 0; // first eval AFTER evalEvery episodes, not at episode 0
let evalRunning = false;
let lastReport = 0;
/** Rolling games/s, re-anchored each report so the tile reflects the CURRENT
 *  regime (post-ε-decay, post-reconfigure), not the session average. */
let gamesPerSec = 0;
let gamesWindowStart = performance.now();
let gamesWindowCount = 0;
/** Threaded mulberry32 counter for every worker-level draw (lane deals, eval
 *  seeds). The engine and the trainer keep their own counters internally. */
let laneRng = SESSION_SEED;

// --- the interactive lanes ---------------------------------------------------

type LaneMode = "watch" | "play";

/**
 * One inference lane. Training lanes live inside the Trainer; these two stream
 * snapshots to the main thread instead of replay rows into the ring.
 * `ep` is the lane's episode counter AND its generation: it is bumped on every
 * re-deal, so a tick that awaited through a re-deal detects the swap by value.
 */
interface Lane {
  mode: LaneMode;
  tankCount: 2 | 3 | 4;
  teamMode: TeamMode;
  /** Episode length in seconds this lane deals with (watch follows the config;
   *  play is always the real-time 45 s — a human cannot fight a 3 s arena). */
  episodeSeconds: number;
  game: GameState;
  /** Wall-clock tick accumulator (float ticks) — pacing is per lane so an 8×
   *  watch lane never drags the 30 Hz play lane with it. */
  acc: number;
  /** performance.now() of the last pacing sample (kept fresh while paused so a
   *  resume never replays the pause as a burst of catch-up ticks). */
  lastAt: number;
  /** Real-time speed multiplier (1 = the engine's 30 ticks/s; watch knob
   *  0.5–8; play is pinned at 1). Feeds the accumulator and the snapshot
   *  coalescer — never a raw ticks/s figure. */
  speedHz: number;
  /** Ticks between snapshot pushes: 1 for play (every tick), ≥1 for watch so
   *  the stream coalesces to ≤60/s regardless of simulation rate. */
  streamEvery: number;
  paused: boolean;
  lastSnapTick: number;
  /** Events from ticks skipped since the last push — a coalesced snapshot
   *  ships the WHOLE interval's juice, not just its final tick's. */
  eventsSeen: GameEvent[];
  /** Episode counter (1-based) — doubles as the re-deal generation. */
  ep: number;
  // --- play lane only ---
  difficulty: number; // 0..1 → opponent τ = (1 − difficulty) + PLAY_TAU_FLOOR
  aimAssist: boolean;
  /** Highest input seq accepted; lower-or-equal seqs are dropped as stale. */
  inputSeq: number;
  /** Latest accepted input packed through the same 18-action table. */
  inputAction: number;
  /** Per-tank observation scratch (encodeObs zero-fills; act copies out before
   *  its first await, so reuse across ticks is safe). */
  scratch: Float32Array[];
}

let watch: Lane | null = null;
let play: Lane | null = null;

/** Worker-level uniform seed draw: lanes, evals and re-deals all thread this. */
function nextSeed(): number {
  const [seed, next] = randInt(laneRng, 0x40000000);
  laneRng = next;
  return seed;
}

/** The arena contract is 2–4 tanks; a free UI number rounds into it. */
function clampTankCount(n: number): 2 | 3 | 4 {
  const k = Math.round(n);
  return k < 3 ? 2 : k > 3 ? 4 : 3;
}

/** Watch speed knob range (protocol: 0.5×–8×). */
function clampSpeed(hz: number): number {
  return Number.isFinite(hz) ? Math.min(8, Math.max(0.5, hz)) : 1;
}

function clamp01(v: number): number {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5;
}

/** Snapshot coalescing: ticks-per-second is 30·speedHz, so dividing by 60 and
 *  rounding lands every watch rate on ≤60 pushes/s (1× stays at 30 — there is
 *  nothing to coalesce below the render rate). */
function streamEveryFor(speedHz: number): number {
  return Math.max(1, Math.round((TICK_HZ * speedHz) / 60));
}

function newLane(mode: LaneMode, tankCount: 2 | 3 | 4, teamMode: TeamMode): Lane {
  const game = createGame({
    tankCount,
    teamMode,
    // Watch spectates the CONFIGURED episode length (it may be shortened for
    // training throughput); play is always the plan's real-time 45 s.
    episodeSeconds: mode === "play" ? EPISODE_SECONDS_DEFAULT : cfg.episodeSeconds,
    // Lanes carry the configured shaping γ (rewards.ts): watch/play never
    // train, but their episodes are built by the same contract.
    gamma: cfg.gamma,
    seed: nextSeed(),
  });
  // Both lanes start at 1× real time (play is ALWAYS 1×; watch gets the knob
  // in startWatch). The accumulator scales this by TICK_HZ — see runLane.
  const speedHz = 1;
  return {
    mode,
    tankCount,
    teamMode,
    episodeSeconds: mode === "play" ? EPISODE_SECONDS_DEFAULT : cfg.episodeSeconds,
    game,
    acc: 0,
    lastAt: performance.now(),
    speedHz,
    streamEvery: mode === "play" ? 1 : streamEveryFor(speedHz),
    paused: false,
    lastSnapTick: 0,
    eventsSeen: [],
    ep: 1,
    difficulty: 0.5,
    aimAssist: false,
    inputSeq: -1,
    inputAction: COAST_ACTION,
    scratch: game.tanks.map(() => new Float32Array(OBS_DIM)),
  };
}

function pushArena(lane: Lane): void {
  // Unsolicited push (id 0): the static world, once per episode, BEFORE the
  // first snapshot — the renderer must have walls before anything moves. The
  // lane tag is what keeps each controller's live gate fed by ITS lane only.
  ctx.postMessage({ id: 0, lane: lane.mode, arena: toArenaMsg(lane.game) });
}

function pushSnapshot(lane: Lane): void {
  const snap = toSnapshot(lane.game, lane.ep);
  // Steal the accumulator: it already contains this tick's events (laneTick
  // appends before calling), so the coalesced interval ships whole and the
  // engine's own per-tick array is never held across ticks.
  snap.events = lane.eventsSeen;
  lane.eventsSeen = [];
  lane.lastSnapTick = lane.game.tick;
  ctx.postMessage({ id: 0, lane: lane.mode, snapshot: snap });
}

/** Re-deal a finished watch lane in place (same pacing/knobs, fresh world). */
function redeal(lane: Lane): void {
  lane.game = createGame({
    tankCount: lane.tankCount,
    teamMode: lane.teamMode,
    episodeSeconds: lane.episodeSeconds,
    gamma: cfg.gamma,
    seed: nextSeed(),
  });
  lane.ep++;
  lane.acc = 0;
  lane.lastAt = performance.now();
  lane.lastSnapTick = 0;
  lane.eventsSeen = [];
  pushArena(lane);
  scheduleTick(); // the chain may have slept while the finished lane idled
}

/** True when a tank may pull the trigger — the per-tank half of physics'
 *  tryFire gate (scheduler.canFire's rule: the transient global bullet cap is
 *  deliberately NOT part of the behaviour mask). */
function canFire(t: Tank): boolean {
  return t.reload <= 0 && t.bulletsInFlight < MAX_BULLETS_PER_TANK;
}

/** Wrap an angle into [−π, π) so a heading DIFFERENCE is always the short way
 *  round (encoding.wrapPi's rule; that copy is module-private by design). */
function wrapPi(a: number): number {
  const twoPi = 2 * Math.PI;
  let w = (a + Math.PI) % twoPi;
  if (w < 0) w += twoPi;
  return w - Math.PI;
}

/**
 * Aim assist for the human seat (play tank 0): rotate the hull toward the
 * intercept point of the NEAREST hostile by at most AIM_ASSIST_STEP. The
 * intercept time-of-flight is converged with two fixed-point iterations of
 * t = dist/BULLET_SPEED, aim = target.pos + target.vel·t (target velocity is
 * hull-axis speed — tanks cannot strafe), reading the LIVE state right before
 * the step so the assist aims at where the target will be, not where it is.
 *
 * The assist is a VISIBLE hull snap the player did not input — that is the
 * documented trade (plan: ≤6° intercept snap, a toggle): a snapped hull
 * telegraphs the assist honestly instead of silently bending shells.
 */
function applyAimAssist(game: GameState, enabled: boolean): void {
  if (!enabled) return;
  const me = game.tanks[0];
  if (!me.alive) return;
  let best: Tank | null = null;
  let bestD2 = Infinity;
  for (const t of game.tanks) {
    if (!t.alive || t.team === me.team) continue; // hostile only (allied mode respects teams)
    const dx = t.x - me.x;
    const dy = t.y - me.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = t;
    }
  }
  if (best === null) return;
  let tof = Math.sqrt(bestD2) / BULLET_SPEED;
  let ax = best.x;
  let ay = best.y;
  for (let i = 0; i < AIM_ASSIST_ITERATIONS; i++) {
    ax = best.x + best.speed * Math.cos(best.heading) * tof;
    ay = best.y + best.speed * Math.sin(best.heading) * tof;
    tof = Math.hypot(ax - me.x, ay - me.y) / BULLET_SPEED;
  }
  ax = best.x + best.speed * Math.cos(best.heading) * tof;
  ay = best.y + best.speed * Math.sin(best.heading) * tof;
  const err = wrapPi(Math.atan2(ay - me.y, ax - me.x) - me.heading);
  me.heading += Math.abs(err) <= AIM_ASSIST_STEP ? err : err > 0 ? AIM_ASSIST_STEP : -AIM_ASSIST_STEP;
}

/**
 * Assemble one tick's 18-action vector for a lane: the human seat (play tank
 * 0) takes the latest accepted input action, every AI seat is decided by ONE
 * batched net forward over that lane's alive tanks. Temperature sampling means
 * no fallback chain is needed: the categorical runs over the fire-masked
 * softmax (masked mass is exactly 0), decode/rawAction clamp anything the net
 * could emit, and an unavailable shot no-ops in the engine — play cannot
 * freeze (protocol's never-freeze rule, discharged here).
 */
async function laneActions(lane: Lane): Promise<number[]> {
  const rows: Float32Array[] = [];
  const fire: boolean[] = [];
  const at: number[] = [];
  lane.game.tanks.forEach((t, i) => {
    if (!t.alive || (lane.mode === "play" && i === 0)) return; // corpses coast; the human needs no forward
    encodeObs(lane.game, t.id, lane.scratch[i]);
    rows.push(lane.scratch[i]);
    fire.push(canFire(t));
    at.push(i);
  });
  const acts = lane.game.tanks.map(() => COAST_ACTION);
  if (lane.mode === "play" && lane.game.tanks[0].alive) acts[0] = lane.inputAction;
  if (rows.length > 0) {
    const mode: ActMode =
      lane.mode === "watch"
        ? { kind: "temperature", tau: WATCH_TAU }
        : { kind: "temperature", tau: (1 - lane.difficulty) * 1.0 + PLAY_TAU_FLOOR };
    const res = await dqn.act(rows, fire, mode, laneRng);
    laneRng = res.state;
    for (let r = 0; r < at.length; r++) acts[at[r]] = res.actions[r];
  }
  return acts;
}

/** Which module slot owns this lane (null after a rebuild dropped it). */
function slotOf(mode: LaneMode): Lane | null {
  return mode === "watch" ? watch : play;
}

/** A lane needs the tick chain only while it can actually step. Paused lanes
 *  wake on their resume message, finished watch lanes wake on their restart
 *  beat — both re-arm the chain explicitly, so an idle lane never spins it. */
function isLive(lane: Lane | null): boolean {
  return lane !== null && !lane.paused && !lane.game.over;
}

/**
 * ONE engine tick for one lane: act → step → events → snapshot/episode-end.
 * The guard after the await closes the only race: a re-deal (ep bumped) or a
 * rebuild (lane dropped from its slot) landing while the net forward was in
 * flight — the stale actions are discarded instead of stepping a world they
 * were never computed for.
 */
async function laneTick(lane: Lane): Promise<void> {
  const ep = lane.ep;
  const acts = await laneActions(lane);
  if (ep !== lane.ep || slotOf(lane.mode) !== lane) return;
  if (lane.mode === "play") applyAimAssist(lane.game, lane.aimAssist);
  step(lane.game, acts);
  for (const ev of lane.game.events) lane.eventsSeen.push(ev);
  if (lane.game.over) {
    pushSnapshot(lane); // the board must SHOW the terminal frame
    ctx.postMessage({ id: 0, lane: lane.mode, episodeEnd: episodeEndMsg(lane.game) });
    // Watch lanes always auto-restart (worker-side beat below); play lanes
    // stay dead until the next PlayStart — the human decides on a rematch.
    if (lane.mode === "watch") scheduleWatchRestart(lane);
  } else if (lane.game.tick - lane.lastSnapTick >= lane.streamEvery) {
    pushSnapshot(lane);
  }
}

/** big-two's WATCH_RESTART_MS beat, generation-guarded: a watchStop, a fresh
 *  watch request or a rebuild cancels the pending deal by failing the checks,
 *  and the ep guard cancels it if something else re-dealt the lane meanwhile
 *  (an explicit startWatch on the finished duel must not double-deal). */
function scheduleWatchRestart(lane: Lane): void {
  const gen = rebuildGen;
  const ep = lane.ep;
  setTimeout(() => {
    if (gen !== rebuildGen || watch !== lane || lane.ep !== ep) return;
    redeal(lane);
  }, WATCH_RESTART_MS);
}

/**
 * Advance one lane by its wall-clock budget. Ticks are BANKED (accumulator)
 * and then spent in one bounded burst: faster-than-real-time watch is pacing
 * (speedHz multiplies the accrual rate), never an unbounded loop.
 */
async function runLane(lane: Lane, now: number): Promise<void> {
  if (lane.paused) {
    lane.lastAt = now; // no debt accrues while paused (and resume cannot burst)
    return;
  }
  const elapsed = (now - lane.lastAt) / 1000;
  lane.lastAt = now;
  // speedHz is a real-time MULTIPLIER (protocol 0.5×–8×): 1 = the engine's own
  // 30 ticks/s, 8 = 240. elapsed·speedHz·TICK_HZ converts wall seconds into
  // banked ticks; skipping the TICK_HZ factor silently runs a 1× lane at
  // 1 tick/s (a 45 s episode would take 22 minutes).
  lane.acc += elapsed * lane.speedHz * TICK_HZ;
  let ticks = Math.floor(lane.acc);
  if (ticks > MAX_CATCHUP_TICKS) {
    ticks = MAX_CATCHUP_TICKS;
    lane.acc = 0; // drop the debt — catch-up is capped, not queued (spiral of death)
  } else {
    lane.acc -= ticks;
  }
  for (let i = 0; i < ticks; i++) {
    // A finished lane idles until its restart beat; a lane a rebuild just
    // dropped from its slot stops spending forwards on an orphaned world.
    if (lane.game.over || slotOf(lane.mode) !== lane) break;
    await laneTick(lane);
  }
}

/** Placements best-first: ALIVE tanks by hp desc, then the dead — the same
 *  ranking rewards.placementBonus pays (stable sort keeps id order on ties),
 *  so the scoreboard's table and the timeout reward agree on who beat whom.
 *  kills/deaths are index-aligned with `ranks`, not with tank id (protocol). */
function episodeEndMsg(game: GameState): EpisodeEndMsg {
  const order = game.tanks
    .map((_t, i) => i)
    .sort((a, b) => {
      const A = game.tanks[a];
      const B = game.tanks[b];
      if (A.alive !== B.alive) return A.alive ? -1 : 1;
      return B.hp - A.hp;
    });
  let alive = 0;
  for (const t of game.tanks) if (t.alive) alive++;
  const top = game.tanks[order[0]];
  return {
    // Alive winner = the hp-best survivor (matches the reward's placement
    // winner); null = mutual destruction (nobody left to place first).
    winner: top.alive ? top.id : null,
    ranks: order.map((i) => game.tanks[i].id),
    kills: order.map((i) => game.tanks[i].kills),
    deaths: order.map((i) => game.tanks[i].deaths),
    ticks: game.tick,
    // The engine sets over on ≤1 alive (elimination) or the clock with 2+
    // still standing (timeout) — mirroring its own tie-break exactly.
    reason: alive > 1 ? "timeout" : "elimination",
  };
}

// --- rebuild (transactional) -------------------------------------------------

/**
 * Swap the whole learning state: nets, optimizer, replay, trainer, counters,
 * lanes. TRANSACTIONAL (big-two pattern): every throwing allocation (the nets,
 * the buffer's bufferSize×567-float backing array — an absurd bufferSize
 * RangeErrors in the constructor) runs into LOCALS before the old state is
 * touched, so a failed Apply leaves the previous run fully intact and the
 * error propagates to drain's catch, which reports it on the request id.
 */
async function rebuild(next: NetConfig): Promise<void> {
  const newDqn = new Dqn(next);
  const newBuffer = new ReplayBuffer(next.bufferSize);
  const newTrainer = new Trainer(next, newDqn, newBuffer);
  newTrainer.onTrainStep = (loss) => {
    // The worker's channel for trainSteps/smoothedLoss (metricsSnapshot
    // deliberately leaves both at their defaults for exactly this hook).
    trainSteps++;
    smoothedLoss = smoothedLoss === 0 ? loss : smoothedLoss * 0.95 + loss * 0.05;
  };
  // Snapshot the LIVE lanes before dropping them: a rebuild must be OBSERVABLE
  // (see the tail of this function) — the watch lane is re-dealt from its own
  // knobs, and the play lane's human is told their match ended.
  const watchKnobs = watch !== null ? { tankCount: watch.tankCount, speedHz: watch.speedHz, paused: watch.paused } : null;
  const deadPlay = play !== null && !play.game.over ? play : null;
  dqn?.dispose(); // nets + optimizer slots; the old buffer is plain memory (GC)
  cfg = next;
  dqn = newDqn;
  trainer = newTrainer;
  training = false; // the caller re-arms if it was running
  watch = null; // lanes die with their generation — re-dealt/notified below
  play = null;
  rebuildGen++;
  trainer.rebuildGen = rebuildGen; // mirror: the scheduler's guard field is caller-owned
  trainSteps = 0;
  smoothedLoss = 0;
  winRateVsRandom = 0;
  lastSync = 0;
  nextEvalAt = cfg.evalEvery;
  // evalRunning is deliberately NOT reset: an in-flight eval's finally owns the
  // flag, and its stale result is dropped by the generation guard below.
  gamesPerSec = 0;
  gamesWindowStart = performance.now();
  gamesWindowCount = 0;
  await newTrainer.reset(nextSeed());
  // LANES ARE RE-REQUESTED HERE, not "by their tabs": no controller learns
  // about a rebuild, so a silent `watch = null; play = null` used to leave a
  // mid-frame frozen arena, a keyboard forwarding into a null lane and no
  // banner on either tab until the user touched a toolbar. (a) Re-deal the
  // spectator from its own knobs (paused state carried, so a paused duel does
  // not secretly resume); (b) post a synthetic episode end for a live play
  // match so the existing banner/rematch path re-arms the human's seat — the
  // human, not the worker, decides to restart their match.
  if (watchKnobs !== null) {
    // startWatch hands back the lane it (re)dealt, so the paused flag can be
    // carried over without re-reading the module slot through a narrowing.
    startWatch(watchKnobs.tankCount, watchKnobs.speedHz).paused = watchKnobs.paused;
  }
  if (deadPlay !== null) {
    ctx.postMessage({
      id: 0,
      lane: "play",
      episodeEnd: {
        winner: null,
        ranks: deadPlay.game.tanks.map((t) => t.id),
        kills: deadPlay.game.tanks.map((t) => t.kills),
        deaths: deadPlay.game.tanks.map((t) => t.deaths),
        ticks: deadPlay.game.tick,
        reason: "reconfigured",
      },
    });
  }
}

// --- eval --------------------------------------------------------------------

/**
 * Greedy-vs-random eval on the trainer's weight snapshot (runEval snapshots,
 * restores and disposes around the WHOLE run — that contract is the trainer's;
 * the worker only guards the generation and never awaits the result, so eval's
 * tensor reads interleave with the tick chain instead of blocking it).
 */
function startEval(t: Trainer): void {
  evalRunning = true;
  const gen = rebuildGen;
  t.runEval(nextSeed())
    .then((res) => {
      if (gen !== rebuildGen) return; // reconfigured mid-eval — numbers are stale
      winRateVsRandom = res.winRate;
    })
    .catch(() => {
      // Telemetry only: a rebuild disposing this generation's net mid-await
      // lands here (runEval's own finally has already released its snapshot).
    })
    .finally(() => {
      evalRunning = false;
    });
}

// --- metrics -----------------------------------------------------------------

function snapshotMetrics(): Metrics {
  const now = performance.now();
  const elapsed = (now - gamesWindowStart) / 1000;
  if (trainer.episodes > gamesWindowCount && elapsed >= 0.5) {
    gamesPerSec = (trainer.episodes - gamesWindowCount) / elapsed;
    gamesWindowStart = now;
    gamesWindowCount = trainer.episodes;
  }
  const m = trainer.metricsSnapshot({ backend, gamesPerSec });
  // The worker-owned slice (see metricsSnapshot's header): losses arrive
  // through onTrainStep, win-rate through startEval, ts is this push moment.
  m.trainSteps = trainSteps;
  m.smoothedLoss = smoothedLoss;
  m.winRateVsRandom = winRateVsRandom;
  m.ts = Date.now();
  return m;
}

function maybeReport(now: number, force = false): void {
  if (!force && now - lastReport < REPORT_THROTTLE_MS) return;
  lastReport = now;
  ctx.postMessage({ id: 0, metrics: snapshotMetrics() });
}

/** Request ack: a metrics echo on the request id (the M1 stub's contract, kept
 *  — the client pairs by id, so watch/play/config starts surface a response). */
function ack(id: number): void {
  ctx.postMessage({ id, metrics: snapshotMetrics() });
}

// --- the tick chain ----------------------------------------------------------
//
// Exactly ONE chain may exist at a time (big-two's tickScheduled/tickRunning
// rule): a reconfigure landing while a tick is between awaits must not arm a
// SECOND chain, and a start landing while one is pending must not stall. The
// finally is the single point of continuation.

let tickScheduled = false;
let tickRunning = false;

function scheduleTick(): void {
  if (tickScheduled || tickRunning) return;
  if (!training && !isLive(watch) && !isLive(play)) return;
  tickScheduled = true;
  setTimeout(() => void tick(), TICK_DELAY_MS);
}

async function tick(): Promise<void> {
  tickScheduled = false;
  tickRunning = true;
  try {
    const now = performance.now();
    // (1) INTERACTIVE LANES FIRST (plan §Architecture). Each lane is its own
    // try: a rebuild can drop a lane whose forward is mid-await, and that
    // teardown must never kill the chain the other lanes depend on.
    if (watch !== null) {
      try {
        await runLane(watch, now);
      } catch {
        /* lane torn down mid-await (rebuild) — next iteration is clean */
      }
    }
    if (play !== null) {
      try {
        await runLane(play, now);
      } catch {
        /* same */
      }
    }
    // (2) TRAINING SLICE — pinned against a rebuild landing mid-await: the
    // generation guard discards the sync/eval decisions if the world they
    // were computed against is gone.
    if (training) {
      const t = trainer;
      const d = dqn;
      const gen = rebuildGen;
      try {
        await t.fleetTick();
        if (gen === rebuildGen) {
          if (d.steps - lastSync >= cfg.targetSync) {
            d.syncTarget();
            lastSync = d.steps;
          }
          if (t.episodes >= nextEvalAt && !evalRunning) {
            nextEvalAt = t.episodes + cfg.evalEvery;
            startEval(t);
          }
        }
      } catch {
        /* this generation's net was disposed mid-await by a rebuild */
      }
    }
    // (3) throttled progress push.
    maybeReport(performance.now());
  } finally {
    tickRunning = false;
    // Re-arm only while real work exists: training, or a lane that can step.
    // Paused/finished lanes sleep the chain — their resume/restart messages
    // call scheduleTick themselves, so nothing is ever stranded.
    if (training || isLive(watch) || isLive(play)) scheduleTick();
  }
}

// --- message dispatch --------------------------------------------------------

/** Start (or re-time) the watch lane; returns the lane it left in the slot
 *  (callers that only need the push/ack ignore the return value). */
function startWatch(tankCount: number, speedHz: number): Lane {
  const n = clampTankCount(tankCount);
  const hz = clampSpeed(speedHz);
  if (watch !== null && watch.tankCount === n) {
    // Speed is PACING, not identity: keep the duel, just re-time it. A request
    // landing on a finished duel deals fresh (and the ep guard retires any
    // restart beat already pending for the old episode). The arena is
    // RE-PUSHED either way: the controller's live gate only re-arms on an
    // {arena} message, and a same-settings restart mid-episode must not blank
    // the feed until the next auto-restart — the geometry is unchanged, so the
    // repaint is free.
    watch.speedHz = hz;
    watch.streamEvery = streamEveryFor(hz);
    if (watch.game.over) redeal(watch);
    else pushArena(watch);
    scheduleTick();
    return watch;
  }
  watch = newLane("watch", n, "ffa"); // spectator duels are always FFA
  watch.speedHz = hz;
  watch.streamEvery = streamEveryFor(hz);
  pushArena(watch);
  scheduleTick(); // a watch lane runs whether or not training is on
  return watch;
}

function startPlay(
  tankCount: number,
  teamMode: TeamMode,
  difficulty: number,
  aimAssist: boolean,
): void {
  // A PlayStart always restarts from a fresh arena (protocol) — the human's
  // rematch button and the config form share one honest rule.
  play = newLane("play", clampTankCount(tankCount), teamMode);
  play.difficulty = clamp01(difficulty);
  play.aimAssist = aimAssist;
  pushArena(play);
  scheduleTick();
}

async function handle(req: WorkerReq): Promise<void> {
  if (isStartReq(req)) {
    training = true;
    scheduleTick(); // no-op if a tick is already pending or running
    ack(req.id);
  } else if (isStopReq(req)) {
    training = false; // the active iteration exits without re-arming training
    ack(req.id);
  } else if (isReconfigureReq(req)) {
    // big-two's reconfigure: rebuild → ack with FRESH metrics (the UI mirrors
    // the reset) → re-arm if it was training. rebuild is transactional, so a
    // throw above leaves the previous run intact and drain's catch replies.
    const wasTraining = training;
    await rebuild(req.config);
    ack(req.id);
    if (wasTraining) {
      training = true;
      // A mid-flight iteration re-arms itself from its finally; if none is
      // pending, arm one now. The single-chain invariant folds both cases.
      scheduleTick();
    }
  } else if (isWatchStartReq(req)) {
    startWatch(req.watch.tankCount, req.watch.speedHz);
    ack(req.id);
  } else if (isWatchStopReq(req)) {
    watch = null; // snapshots stop after the in-flight tick; pending restart beat no-ops
    ack(req.id);
  } else if (isPauseReq(req)) {
    // Lane-SCOPED pause: the request addresses one lane (the controllers
    // always pass `lane`), or both when it carries none — one shared flag used
    // to freeze the Watch duel because the Play tab was hidden (and silently
    // resume a paused duel when Play came back). Resuming re-arms the
    // (possibly sleeping) chain and re-bases lastAt so the paused wall time is
    // not replayed as catch-up ticks.
    const targets = req.lane === undefined ? [watch, play] : [req.lane === "watch" ? watch : play];
    for (const lane of targets) {
      if (lane === null) continue;
      lane.paused = req.pause;
      lane.lastAt = performance.now();
    }
    if (!req.pause) scheduleTick();
    ack(req.id);
  } else if (isStepReq(req)) {
    // Frame-step the paused lane one tick and push REGARDLESS of pacing (the
    // pushSkip check below is what makes a coalesced watch lane still show it).
    const lane = watch ?? play;
    if (lane !== null && !lane.game.over) {
      await laneTick(lane);
      if (lane.game.tick !== lane.lastSnapTick) pushSnapshot(lane);
    }
    ack(req.id);
  } else if (isPlayStartReq(req)) {
    startPlay(req.play.tankCount, req.play.teamMode, req.play.difficulty, req.play.aimAssist);
    ack(req.id);
  } else if (isInputReq(req)) {
    // Fire-and-forget (never acked — 30/s would spam the metrics fan-out):
    // latest-wins by seq, so packets that overtake each other in the queue
    // collapse to the freshest control state.
    const lane = play;
    if (lane !== null && req.input.seq > lane.inputSeq) {
      lane.inputSeq = req.input.seq;
      lane.inputAction = encodeAction(req.input.throttle, req.input.steer, req.input.fire);
    }
  } else {
    // Unreachable while the union and the guards agree — reached on version
    // skew between main and worker bundles. Reply on 0, not a stale id, so
    // the client can never mis-pair the error.
    ctx.postMessage({ id: 0, error: `train worker: unrecognized request ${JSON.stringify(req)}` });
  }
}

// ?test=1 (smoke fixture): a tiny fast config so Playwright exercises the
// REAL loop — training lanes, replay, watch stream — in seconds. evalEvery is
// pushed beyond the session horizon (eval games would dominate a 3 s episode
// budget) and the watch lane starts moving before any controller asks.
function isTestMode(): boolean {
  // Workers read their own WorkerLocation; the DOM lib's `location` typings do
  // not cover it, hence the one structural cast.
  const loc = (self as unknown as { location?: { search?: string } }).location;
  return loc?.search !== undefined && new URLSearchParams(loc.search).has("test");
}

async function applyTestFixture(): Promise<void> {
  await rebuild({
    ...defaultConfig(),
    lanes: 2,
    tankCount: 2,
    episodeSeconds: 3,
    evalEvery: 1e9,
    batchSize: 16,
    bufferSize: 512,
    trainEvery: 8,
    hidden: [32],
  });
  training = true;
  startWatch(2, 8);
  scheduleTick();
}

// Safari/WebKit (≤26.x) re-evaluates a module-worker entry when a dynamically
// imported chunk imports it back (Vite's tfjs backend chunks do). The second
// evaluation must stay a GHOST: a second init would race tf.setBackend (the
// duplicate webgpu attempt silently loses, falls to webgl, and the mid-init
// backend swap crashes the trainer) and would boot a second trainer against
// the same realm-global tfjs registry. Module-scope state is per-instance;
// only the global scope spans both evaluations, so the sentinel lives there
// and is captured before any side effect below runs.
const bootScope = globalThis as typeof globalThis & { __trainWorkerBooted?: boolean };
const firstEvaluation = !bootScope.__trainWorkerBooted;
bootScope.__trainWorkerBooted = true;

// Promise-gated init: NOTHING may create a tensor before the backend resolves
// (a tensor without a backend throws). Any message arriving first is queued and
// drained after init completes.
const initPromise: Promise<void> = firstEvaluation ? (async () => {
  try {
    backend = await initBackend();
    if (isTestMode()) await applyTestFixture();
    else await rebuild(cfg);
    // Kernel warmup BEFORE the handshake (see WARMUP_FLEET_TICKS): the first
    // forward + gradient steps compile here, while the lanes cannot yet exist.
    for (let i = 0; i < WARMUP_FLEET_TICKS; i++) await trainer.fleetTick();
    // Boot handshake: one real metrics push so the shell lights the status
    // chip without a request/response round trip (the M1 stub's move).
    maybeReport(performance.now(), true);
  } catch (err) {
    initError = err instanceof Error ? err.message : String(err);
  }
})() : Promise.resolve();

const queue: WorkerReq[] = [];
let draining = false;

async function drain(): Promise<void> {
  if (draining) return; // one drainer; late arrivals wait in `queue`
  draining = true;
  try {
    await initPromise; // backend + first trainer exist past this point
    if (initError !== null) {
      ctx.postMessage({ id: 0, error: `worker init failed: ${initError}` });
      return;
    }
    while (queue.length > 0) {
      const req = queue.shift()!;
      try {
        await handle(req);
      } catch (err) {
        // Every handler failure (a rejected Apply, a bad lane config) is one
        // ErrorResp on the request id — the worker never wedges, the UI always
        // gets an answer to pair with its promise.
        ctx.postMessage({
          id: req.id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } finally {
    draining = false;
  }
}

// Ghost evaluation leaves instance 1's wiring in place (see bootScope above).
if (firstEvaluation) {
  ctx.onmessage = (e: MessageEvent<WorkerReq>): void => {
    queue.push(e.data);
    void drain();
  };
}
