// train-protocol.ts — the message contract between the main thread and the
// train worker. Discriminated-by-shape (field presence), id-keyed — no literal
// `type`/`kind` tag — mirroring the big-two / tic-tac-toe house pattern.
//
// ONE worker owns everything tfjs: self-play training across `lanes` lockstep
// arenas AND all inference for play/watch. The main thread never imports
// @tensorflow/* (a unit-test grep enforces this), so weights never cross the
// message boundary — only metrics, arena layouts, and render snapshots do.
//
// Requests carry a monotonically increasing `id`; responses echo it so the
// caller can pair them. Unsolicited pushes (metrics progress, the watch/play
// snapshot stream, episode results) use id 0 — they belong to no request.
//
// Snapshot/ArenaMsg live in game/snapshot.ts (they are the engine's own
// serialization, shared with the determinism tests); they are re-exported here
// so consumer code imports the whole wire contract from this one module.

import type { ArenaMsg, Snapshot } from "./game/snapshot";
import type { TeamMode } from "./game/types";

// The serialized world and the static arena layout — re-exported so the
// protocol stays the single import site for everything crossing the boundary.
export type { ArenaMsg, Snapshot } from "./game/snapshot";
export { toArenaMsg, toSnapshot } from "./game/snapshot";
export type { TeamMode } from "./game/types";

// ===== Config & metrics =====

/**
 * Everything the worker needs to (re)build nets, replay, and the scheduler.
 * Applied transactionally on reconfigure: build new → dispose old → reset, so
 * a rejected config leaves the previous run intact.
 */
export interface NetConfig {
  /** "mlp" = dense stack over the flat 282-float obs; "attention" = entity encoder. */
  architecture: "mlp" | "attention";
  /** MLP dense layer sizes (attention mode reads the four fields below instead). */
  hidden: number[];
  /** Attention encoder: token embedding dim, heads, pre-LN blocks, FFN width. */
  dModel: number;
  heads: number;
  layers: number;
  ffn: number;

  learningRate: number;
  batchSize: number;
  bufferSize: number; // replay rows (row = 567 floats)
  gamma: number;
  epsStart: number;
  epsDecay: number; // multiplied per completed episode
  epsMin: number;

  /** Concurrent self-play arenas ticked in lockstep (one fleet forward/tick). */
  lanes: number;
  /** One gradient step every N decisions seen across the fleet. */
  trainEvery: number;
  /** Hard target-net sync every N train steps. */
  targetSync: number;
  /** Greedy-vs-random eval every N episodes, over `evalEpisodes` games. */
  evalEvery: number;
  evalEpisodes: number;
  /** Episode length in seconds (hard cap 90 s regardless). */
  episodeSeconds: number;
  /** Tanks per arena (2–4; training always FFA). */
  tankCount: number;
  /** DDQN when true: target argmax comes from the online net. */
  doubleQ: boolean;
}

/**
 * Fleet-wide training telemetry, pushed at ≤5 Hz (id 0) and echoed per
 * request. Rates are per elapsed wall second; ratios are unit-interval.
 */
export interface Metrics {
  episodes: number; // completed training episodes (all lanes summed)
  decisions: number; // action selections seen (drives trainEvery)
  trainSteps: number; // gradient steps applied
  epsilon: number; // current exploration rate
  smoothedLoss: number; // Huber loss, EMA 0.95/0.05
  winRateVsRandom: number; // greedy eval vs random policy; chance = 1/N
  avgReturn: number; // mean discounted return per episode
  killDeathRatio: number; // kills / deaths across the fleet
  survivalRate: number; // fraction of episodes survived to timeout
  gamesPerSec: number; // fleet throughput (episodes / s)
  backend: string; // webgpu | webgl | wasm | cpu
  bufferFill: number; // 0..1 fraction of replay rows in use
  aliveLanes: number; // lanes currently mid-episode
  ts: number; // worker clock (ms) at push time
}

/** All-zero metrics — the worker's pre-training identity and reset baseline. */
export function zeroMetrics(): Metrics {
  return {
    episodes: 0,
    decisions: 0,
    trainSteps: 0,
    epsilon: 1,
    smoothedLoss: 0,
    winRateVsRandom: 0,
    avgReturn: 0,
    killDeathRatio: 0,
    survivalRate: 0,
    gamesPerSec: 0,
    // "cpu" until the M3 worker runs the WebGPU→WebGL→WASM→CPU ladder.
    backend: "cpu",
    bufferFill: 0,
    aliveLanes: 0,
    ts: 0,
  };
}

// ===== Requests (main → worker) =====

export interface StartTrainingReq {
  id: number;
  start: true;
}
export interface StopTrainingReq {
  id: number;
  stop: true;
}
export interface ReconfigureReq {
  id: number;
  config: NetConfig;
}

/** Start (or restart) the watch lane: `tankCount` greedy AI duels at `speedHz`
 *  wall-clock ticks/s (0.5–8, snapshotted with coalescing so a 8× lane still
 *  posts one snapshot per animation frame, not eight). */
export interface WatchStartReq {
  id: number;
  watch: { tankCount: number; speedHz: number };
}
/** Kill the watch lane; snapshots stop after the in-flight tick. */
export interface WatchStopReq {
  id: number;
  watchStop: true;
}
/** Freeze/unfreeze ONE lane in place — `true` pauses, `false` resumes. A
 *  boolean, not a start/stop pair, so the pause button toggles one field.
 *  `lane` scopes the request: absent (the wire default) targets BOTH lanes —
 *  which is what StepReq's watch??play fallback still expects — while the
 *  controllers always address their own lane, because a Play-tab hide must
 *  never freeze the Watch duel (and vice versa) through one shared flag. */
export interface PauseReq {
  id: number;
  pause: boolean;
  lane?: LaneKind;
}
/** Advance the paused watch lane exactly one tick (frame-step button). */
export interface StepReq {
  id: number;
  step: true;
}

/** Start (or restart) the play lane: human seat + `tankCount`−1 AIs.
 *  `difficulty` ∈ [0,1] maps to the opponent softmax temperature
 *  τ = (1 − difficulty) + 0.05; `aimAssist` snaps the human's shots to the
 *  ≤6° intercept cone. Always restarts the episode from a fresh arena. */
export interface PlayStartReq {
  id: number;
  play: { tankCount: number; teamMode: TeamMode; difficulty: number; aimAssist: boolean };
}

/** The human seat's control state for one tick. `seq` is a monotonic counter
 *  so the worker can drop stale packets that overtake each other in the
 *  queue; the message is deliberately fire-and-forget (see control()). */
export interface InputReq {
  id: number;
  input: { throttle: number; steer: number; fire: boolean; seq: number };
}

export type WorkerReq =
  | StartTrainingReq
  | StopTrainingReq
  | ReconfigureReq
  | WatchStartReq
  | WatchStopReq
  | PauseReq
  | StepReq
  | PlayStartReq
  | InputReq;

// ===== Responses (worker → main) =====

/** Which inference lane a lane-scoped push belongs to. Both lanes post on
 *  id 0, and the human's match and the spectator duel would otherwise be
 *  indistinguishable to the client fan-out (each controller ingesting both
 *  streams = two worlds interleaved into one canvas/HUD). */
export type LaneKind = "watch" | "play";

/** Progress push (id 0) or the ack to any request that moved the counters. */
export interface MetricsResp {
  id: number;
  metrics: Metrics;
}
/** Once per episode, before the first snapshot: the static world to draw.
 *  (A WatchStart re-request also re-pushes it: same-settings restarts keep
 *  the lane alive mid-episode, and the controller's live gate re-arms on this
 *  message — geometry is unchanged, so the repaint is free.) */
export interface ArenaResp {
  id: number;
  lane: LaneKind;
  arena: ArenaMsg;
}
/** The watch/play lane's world stream (id 0, coalesced to the render rate). */
export interface SnapshotResp {
  id: number;
  lane: LaneKind;
  snapshot: Snapshot;
}

/** Terminal state of one episode. `ranks` are tank ids best-first (winner
 *  first; elimination order, HP-ordered on timeout) so the scoreboard can
 *  render placements without re-deriving them; `kills`/`deaths` are
 *  index-aligned with `ranks` (not with tank id) — the placement table is the
 *  only consumer. `winner: null` = mutual destruction / full timeout draw.
 *  `reason: "reconfigured"` is the synthetic end the worker posts when a
 *  config Apply dropped a live lane: the match did not finish, the human
 *  decides on a rematch. */
export interface EpisodeEndMsg {
  winner: number | null;
  ranks: number[];
  kills: number[];
  deaths: number[];
  ticks: number;
  reason: "elimination" | "timeout" | "reconfigured";
}
export interface EpisodeEndResp {
  id: number;
  lane: LaneKind;
  episodeEnd: EpisodeEndMsg;
}
export interface ErrorResp {
  id: number;
  error: string;
}
export type WorkerResp = MetricsResp | ArenaResp | SnapshotResp | EpisodeEndResp | ErrorResp;

// ===== Discriminators (field presence, NOT a literal kind) =====

export function isStartReq(r: WorkerReq): r is StartTrainingReq {
  return "start" in r;
}
export function isStopReq(r: WorkerReq): r is StopTrainingReq {
  return "stop" in r;
}
export function isReconfigureReq(r: WorkerReq): r is ReconfigureReq {
  return "config" in r;
}
export function isWatchStartReq(r: WorkerReq): r is WatchStartReq {
  return "watch" in r;
}
export function isWatchStopReq(r: WorkerReq): r is WatchStopReq {
  return "watchStop" in r;
}
// Field presence, not truthiness: `pause: false` (resume) must still match.
export function isPauseReq(r: WorkerReq): r is PauseReq {
  return "pause" in r;
}
export function isStepReq(r: WorkerReq): r is StepReq {
  return "step" in r;
}
export function isPlayStartReq(r: WorkerReq): r is PlayStartReq {
  return "play" in r;
}
export function isInputReq(r: WorkerReq): r is InputReq {
  return "input" in r;
}

export function isMetrics(r: WorkerResp): r is MetricsResp {
  return "metrics" in r;
}
export function isArena(r: WorkerResp): r is ArenaResp {
  return "arena" in r;
}
export function isSnapshot(r: WorkerResp): r is SnapshotResp {
  return "snapshot" in r;
}
export function isEpisodeEnd(r: WorkerResp): r is EpisodeEndResp {
  return "episodeEnd" in r;
}
export function isError(r: WorkerResp): r is ErrorResp {
  return "error" in r;
}

/**
 * Plan defaults (SPEC §Training). Sample-reuse ratio = batchSize/trainEvery =
 * 2 rows per gradient step; ε per-episode decay 0.995 reaches epsMin 0.05 at
 * ~episode 600, leaving the bulk of a session running the near-greedy policy.
 * γ 0.97 spans ~33 ticks — a full bullet flight is 30.
 */
export function defaultConfig(): NetConfig {
  return {
    architecture: "mlp",
    hidden: [256, 256],
    dModel: 64,
    heads: 4,
    layers: 2,
    ffn: 128,
    learningRate: 3e-4,
    batchSize: 256,
    bufferSize: 40_000,
    gamma: 0.97,
    epsStart: 1.0,
    epsDecay: 0.995,
    epsMin: 0.05,
    lanes: 16,
    trainEvery: 128,
    targetSync: 1000,
    evalEvery: 100,
    evalEpisodes: 24,
    episodeSeconds: 45,
    tankCount: 4,
    doubleQ: true,
  };
}
