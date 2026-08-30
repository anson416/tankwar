// protocol.spec.ts — the main↔worker wire contract. The whole app hangs off
// these shapes: the client pairs responses by id, the worker discriminates by
// field presence, and one wrong key means a silently dropped message (no
// type tag to catch it at runtime). So this spec pins all three legs:
//  - every guard matches EXACTLY its own message (presence-discrimination is
//    only safe if no two shapes share a payload field name),
//  - the plan defaults are the numbers the SPEC trains with,
//  - the {arena}/{snapshot} payloads survive the actual crossing they exist
//    for — a structured clone ≈ JSON round trip — byte for byte.
import { describe, expect, it } from "vitest";
import { ARENA_H, ARENA_W, MAX_TICKS_CAP, TICK_HZ } from "../../src/game/constants";
import { createGame, step } from "../../src/game/engine";
import { toArenaMsg, toSnapshot } from "../../src/game/snapshot";
import {
  defaultConfig,
  isArena,
  isEpisodeEnd,
  isError,
  isInputReq,
  isMetrics,
  isPauseReq,
  isPlayStartReq,
  isReconfigureReq,
  isSnapshot,
  isStartReq,
  isStepReq,
  isStopReq,
  isWatchStartReq,
  isWatchStopReq,
  zeroMetrics,
  type EpisodeEndMsg,
  type WorkerReq,
  type WorkerResp,
} from "../../src/train-protocol";

// A real episode fixture: arena-generated layout, two FFA tanks, one fired
// tick so the snapshot carries live bullets rather than empty arrays.
const game = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 45, seed: 2026 });
step(game, [3, 3]); // coast/straight/fire for both — deterministic, reload is 0
const snapshot = toSnapshot(game, 1);
const arena = toArenaMsg(game);

const EPISODE_END: EpisodeEndMsg = {
  winner: 0,
  ranks: [0, 1],
  kills: [1, 0],
  deaths: [0, 1],
  ticks: 900,
  reason: "elimination",
};

// --- the nine request shapes, in guard-table order --------------------------
const REQS: WorkerReq[] = [
  { id: 1, start: true },
  { id: 2, stop: true },
  { id: 3, config: defaultConfig() },
  { id: 4, watch: { tankCount: 3, speedHz: 2 } },
  { id: 5, watchStop: true },
  { id: 6, pause: false }, // resume — presence must not be truthiness
  { id: 7, step: true },
  { id: 8, play: { tankCount: 2, teamMode: "allied", difficulty: 0.5, aimAssist: true } },
  { id: 9, input: { throttle: 1, steer: -1, fire: false, seq: 42 } },
];
const REQ_GUARDS: Array<[(r: WorkerReq) => boolean, number]> = [
  [isStartReq, 0],
  [isStopReq, 1],
  [isReconfigureReq, 2],
  [isWatchStartReq, 3],
  [isWatchStopReq, 4],
  [isPauseReq, 5],
  [isStepReq, 6],
  [isPlayStartReq, 7],
  [isInputReq, 8],
];

// --- the five response/push shapes ------------------------------------------
// The lane-scoped pushes carry `lane` at the source (the client routes each
// push only to the listeners registered for that lane); "watch" here is just
// the fixture lane — the guard table discriminates on the payload field.
const RESPS: WorkerResp[] = [
  { id: 0, metrics: zeroMetrics() },
  { id: 0, lane: "watch", arena },
  { id: 0, lane: "watch", snapshot },
  { id: 0, lane: "watch", episodeEnd: EPISODE_END },
  { id: 0, error: "backend: no webgpu" },
];
const RESP_GUARDS: Array<[(r: WorkerResp) => boolean, number]> = [
  [isMetrics, 0],
  [isArena, 1],
  [isSnapshot, 2],
  [isEpisodeEnd, 3],
  [isError, 4],
];

describe("protocol guards", () => {
  it("each request guard matches exactly its own message shape", () => {
    for (const [guard, own] of REQ_GUARDS) {
      REQS.forEach((r, i) => {
        expect(guard(r), `guard idx ${own} vs req idx ${i}`).toBe(i === own);
      });
    }
  });

  it("each response guard matches exactly its own message shape", () => {
    for (const [guard, own] of RESP_GUARDS) {
      RESPS.forEach((r, i) => {
        expect(guard(r), `guard idx ${own} vs resp idx ${i}`).toBe(i === own);
      });
    }
  });

  it("pause: false (resume) still discriminates as a PauseReq", () => {
    const resume: WorkerReq = { id: 10, pause: false };
    expect(isPauseReq(resume)).toBe(true);
  });

  it("an unrecognized request matches no guard (worker answers with a 0-id error)", () => {
    // Version-skew probe: a newer main bundle talking to a stale worker must
    // fall through every guard, never mis-discriminate into an existing one.
    const alien = { id: 99, warp: true } as unknown as WorkerReq;
    for (const [guard] of REQ_GUARDS) expect(guard(alien)).toBe(false);
    const alienResp = { id: 0, warp: true } as unknown as WorkerResp;
    for (const [guard] of RESP_GUARDS) expect(guard(alienResp)).toBe(false);
  });
});

describe("defaultConfig", () => {
  it("pins the plan defaults (SPEC §Training)", () => {
    expect(defaultConfig()).toEqual({
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
    });
  });

  it("returns a fresh object each call (Apply must not mutate a shared default)", () => {
    const a = defaultConfig();
    const b = defaultConfig();
    expect(a).not.toBe(b);
    expect(a.hidden).not.toBe(b.hidden);
    a.hidden.push(999);
    a.batchSize = 1;
    expect(defaultConfig().batchSize).toBe(256);
    expect(defaultConfig().hidden).toEqual([256, 256]);
  });

  it("is self-consistent with the scheduler's invariants", () => {
    const c = defaultConfig();
    // Sample-reuse ratio 2 rows per gradient step (plan §Network + training).
    expect(c.batchSize / c.trainEvery).toBe(2);
    expect(c.gamma).toBeGreaterThan(0);
    expect(c.gamma).toBeLessThan(1);
    expect(c.epsMin).toBeLessThanOrEqual(c.epsStart);
    expect(c.epsDecay).toBeGreaterThan(0);
    expect(c.epsDecay).toBeLessThan(1);
    expect(c.tankCount).toBeGreaterThanOrEqual(2);
    expect(c.tankCount).toBeLessThanOrEqual(4);
    // The default episode must fit under the hard tick cap (90 s).
    expect(c.episodeSeconds * TICK_HZ).toBeLessThanOrEqual(MAX_TICKS_CAP);
  });
});

describe("snapshot/arena payloads", () => {
  it("fixture is a live 2-tank tick, not an empty world", () => {
    expect(game.tanks).toHaveLength(2);
    expect(snapshot.tick).toBe(1);
    expect(snapshot.tanks).toHaveLength(2);
    expect(snapshot.bullets).toHaveLength(2); // both tanks fired on tick 1
    expect(arena.w).toBe(ARENA_W);
    expect(arena.h).toBe(ARENA_H);
    expect(arena.spawns.length).toBeGreaterThanOrEqual(2);
    expect(arena.walls.length + arena.crates.length).toBeGreaterThan(0);
  });

  it("snapshot survives a JSON round trip unchanged", () => {
    // postMessage structured-clones; JSON is the stricter superset check that
    // runs under node — if JSON can't preserve it, the renderer's copy won't
    // match the encoder's view of the same tick.
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  it("arena message survives a JSON round trip unchanged", () => {
    expect(JSON.parse(JSON.stringify(arena))).toEqual(arena);
  });
});
