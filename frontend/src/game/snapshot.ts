// snapshot.ts — the serialized world the worker ships to the main thread.
// DOM-free, tfjs-free, and allocation-light: a snapshot is plain JSON-able
// objects with SMALL ROUNDED NUMBERS, because it crosses postMessage 30×/s
// (watch/play lanes) and structured-clone cost scales with object count, not
// bytes — flat rows of numbers clone in microseconds.
//
// WHY a dedicated shape instead of shipping GameState: the live state is
// mutable and full of fields the renderer must not write back (rngState,
// nextBulletId, lastRewards). The snapshot is a read-only projection whose
// field names are the renderer's vocabulary (h/v for heading/speed, i for id),
// and because it is plain data it doubles as the exact serialization the
// determinism tests diff — if two replays produce equal snapshots every tick,
// they are the same episode.

import { ARENA_H, ARENA_W } from "./constants";
import type { GameEvent, GameState, Phase, PowerUpKind, Vec } from "./types";

/** One tank as the renderer/HUD sees it. Coordinates are rounded to 2
 *  decimals (a canvas cannot draw a thousandth of a unit; rounding also makes
 *  snapshot JSON stable and diffable). `powerups` lists the ACTIVE timer kind
 *  names — repair is instantaneous and never appears. */
export interface SnapshotTank {
  i: number;
  x: number;
  y: number;
  /** Hull heading, radians (y-down, clockwise-positive — types.ts). */
  h: number;
  /** Signed speed along the hull axis u/s (negative = reversing). */
  v: number;
  hp: number;
  alive: boolean;
  /** Seconds until the gun is ready (0 = now) — drives the HUD reload bar. */
  reload: number;
  team: number;
  sprite: 0 | 1;
  colorIdx: number;
  shield: boolean;
  powerups: string[];
  kills: number;
  deaths: number;
}

/** One live bullet. `bounces` is ricochets REMAINING (the HUD/particles fade a
 *  shell that is down to its last bounce). */
export interface SnapshotBullet {
  i: number;
  x: number;
  y: number;
  dir: number;
  owner: number;
  team: number;
  bounces: number;
  pierce: boolean;
}

/** One powerup on the floor. Despawn urgency is derivable (despawn − tick),
 *  but the renderer interpolates BETWEEN snapshots, so the observation's
 *  despawnFrac lives in the encoder — this row stays minimal. */
export interface SnapshotPowerUp {
  i: number;
  x: number;
  y: number;
  kind: PowerUpKind;
}

/** The whole visible world for one tick. `events` is THIS tick's juice queue
 *  (fire/hit/kill/explode/pickup/ricochet/spawn/phase/crate) — the effects and
 *  audio layers consume it and discard it; nothing else reads it. */
export interface Snapshot {
  /** Episode number (lane counter) — lets the renderer key effects per episode
   *  without trusting tick resets. */
  ep: number;
  tick: number;
  phase: Phase;
  over: boolean;
  tanks: SnapshotTank[];
  bullets: SnapshotBullet[];
  powerups: SnapshotPowerUp[];
  events: GameEvent[];
}

/** Round to 2 decimals — the only precision any consumer can use. */
function r2(v: number): number {
  return Math.round(v * 100) / 100;
}

/** Timer kinds in PowerUpKind order (repair excluded: instantaneous). */
const TIMER_KINDS: ReadonlyArray<{ kind: Exclude<PowerUpKind, "repair"> }> = [
  { kind: "rapid" },
  { kind: "shield" },
  { kind: "speed" },
  { kind: "bounce" },
  { kind: "pierce" },
];

/**
 * Project the live world into a transferable snapshot. Reads the state, never
 * mutates it; `events` is shared by reference (the engine replaces the array,
 * never mutates the old one, so the snapshot's view of its tick is stable).
 */
export function toSnapshot(game: GameState, ep: number): Snapshot {
  return {
    ep,
    tick: game.tick,
    phase: game.phase,
    over: game.over,
    tanks: game.tanks.map((t) => ({
      i: t.id,
      x: r2(t.x),
      y: r2(t.y),
      h: r2(t.heading),
      v: r2(t.speed),
      hp: t.hp,
      alive: t.alive,
      reload: r2(t.reload),
      team: t.team,
      sprite: t.sprite,
      colorIdx: t.colorIdx,
      shield: t.timers.shield > 0,
      powerups: TIMER_KINDS.filter((k) => t.timers[k.kind] > 0).map((k) => k.kind),
      kills: t.kills,
      deaths: t.deaths,
    })),
    bullets: game.bullets.map((b) => ({
      i: b.id,
      x: r2(b.x),
      y: r2(b.y),
      dir: r2(b.dir),
      owner: b.ownerId,
      team: b.team,
      bounces: b.bouncesLeft,
      pierce: b.pierce,
    })),
    powerups: game.powerups.map((p) => ({ i: p.id, x: r2(p.x), y: r2(p.y), kind: p.kind })),
    events: game.events,
  };
}

/** The once-per-episode static world: geometry the renderer draws and the
 *  minimap scales to, plus the seed so a client could regenerate what it is
 *  shown. Walls drop `hp` (indestructible); crates keep it (destructible). */
export interface ArenaMsg {
  walls: { x: number; y: number; w: number; h: number }[];
  crates: { x: number; y: number; w: number; h: number; hp: number }[];
  spawns: Vec[];
  w: number;
  h: number;
  seed: number;
}

/** Project the static arena. Called once per episode, before the first
 *  snapshot, so the renderer has the walls before anything moves. */
export function toArenaMsg(game: GameState): ArenaMsg {
  return {
    walls: game.walls.map((o) => ({ x: o.x, y: o.y, w: o.w, h: o.h })),
    crates: game.crates.map((o) => ({ x: o.x, y: o.y, w: o.w, h: o.h, hp: o.hp })),
    spawns: game.spawns.map((p) => ({ ...p })),
    w: ARENA_W,
    h: ARENA_H,
    seed: game.arenaSeed,
  };
}
