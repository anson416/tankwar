// types.ts — the engine's state contract. DOM-free, tfjs-free.
//
// WHY plain mutable interfaces instead of the big-two-style immutable copies:
// a tank-war tick touches hundreds of fields (16 lanes × 30 Hz in one worker)
// and every transition allocates only bullets/events, never the whole world.
// Determinism is guaranteed by the fixed step order + seeded rngState, not by
// structural sharing; snapshots handed to the renderer/network are serialized
// copies (see snapshot.ts), so nothing can observe a half-ticked state.
//
// CONVENTIONS (binding for every consumer):
//  - heading is RADIANS in y-down screen space: forward = (cos h, sin h),
//    increasing heading = clockwise on screen. Renderer draws sprite rotated
//    by (heading + 90°) because the PNGs face up.
//  - positions are world units in [0, ARENA_W] × [0, ARENA_H], origin top-left.
//  - all durations/timers are SECONDS remaining (float); all schedule fields
//    are absolute tick numbers (int) so pause/step never needs a wall clock.
//  - ids (tank, bullet, powerup) are unique per episode and never reused, so
//    keying by id is safe across snapshots.

/** 2-D vector/point in world units. */
export interface Vec {
  x: number;
  y: number;
}

/** Team structure of the match. Training is always "ffa"; "allied" is the
 *  human-vs-AIs mode (everyone on team 0 shares a side, allied damage off). */
export type TeamMode = "ffa" | "allied";

/** The six powerup kinds — order is load-bearing: it is the index order of
 *  `typeIdx` in the observation's POWERUPS block and of TankTimers' fields. */
export type PowerUpKind = "rapid" | "shield" | "speed" | "repair" | "bounce" | "pierce";

/** Episode phase: 0 normal, 1 escalation (from ESCALATION_FRAC), 2 sudden death
 *  (from SUDDEN_DEATH_FRAC). A union of literals, not an enum, so obs/snapshot
 *  code can copy it as a plain number and tests can pin exact values. */
export type Phase = 0 | 1 | 2;

/** A static rectangle on the arena: solid wall (indestructible) or crate
 *  (destructible cover that blasts when it breaks). */
export interface Obstacle {
  /** Top-left corner u. Walls/crates are axis-aligned; nothing rotates. */
  x: number;
  y: number;
  w: number;
  h: number;
  kind: "wall" | "crate";
  /** Remaining HP — 0 for walls (they cannot take damage; `kind` is the faster
   *  discriminator, hp only matters once kind === "crate"). */
  hp: number;
  /** Stable per-episode identity (crates only — walls are indestructible and
   *  never need it). Assigned by createGame; REQUIRED for bullet hit-marking:
   *  integrateBullets splices broken crates out of GameState.crates every tick,
   *  so an ARRAY INDEX would shift under a live bullet and both re-damage a
   *  moved crate and wrongly shield whoever now holds the stale slot (see
   *  integrateBullets). Optional in the type only so hand-built test fixtures
   *  stay terse; every engine-built crate carries one. */
  id?: number;
}

/** Per-tank powerup countdowns in SECONDS remaining; 0 = inactive. Field order
 *  matches the PowerUpKind index order and the SELF observation block. */
export interface TankTimers {
  rapid: number;
  shield: number;
  speed: number;
  bounce: number;
  pierce: number;
}

/** One tank's full dynamic state. The engine mutates these in place each tick;
 *  `wallContact`/`diedThisTick` are per-tick flags the reward/event pass reads
 *  and then clears. */
export interface Tank {
  /** Unique, never reused within an episode (index into GameState.tanks is NOT
   *  id-derived once deaths are involved — look up by id). */
  id: number;
  x: number;
  y: number;
  /** Hull heading, radians, y-down, clockwise-positive (see header). */
  heading: number;
  /** Signed speed along the hull axis u/s: positive forward, negative reverse —
   *  one scalar because tanks cannot strafe. */
  speed: number;
  hp: number;
  alive: boolean;
  /** Seconds until the gun can fire again (0 = ready); observable, which is why
   *  the behaviour policy may mask fire but the target max must not. */
  reload: number;
  /** This tank's live bullet count against MAX_BULLETS_PER_TANK — maintained by
   *  the bullet pass so the fire check is O(1) instead of a scan. */
  bulletsInFlight: number;
  kills: number;
  deaths: number;
  /** Team id: distinct per tank in "ffa", 0 = human+allies in "allied". */
  team: number;
  /** Which of the two sprite variants to draw (0 = A, 1 = B tread frame); the
   *  renderer animates A/B at 120 ms/frame while moving, purely cosmetic. */
  sprite: 0 | 1;
  /** Palette index for the tint atlas (0–3); also the scoreboard colour order. */
  colorIdx: number;
  timers: TankTimers;
  /** True while touching a wall/arena border this tick — reward pass docks
   *  REWARD_WALL_CONTACT, renderer can show a scrape flash. */
  wallContact: boolean;
  /** One-tick death flag: the rewards pass must charge REWARD_DEATH exactly once
   *  even though the corpse keeps existing until the episode resets. */
  diedThisTick: boolean;
}

/** One live bullet. */
export interface Bullet {
  id: number;
  /** Tank that fired it — for kill credit, ownerCode in the obs, and to skip the
   *  owner on the first sweep (MUZZLE_OFFSET already clears the hull). */
  ownerId: number;
  /** Owner's team at fire time (copied, so a posthumous bullet keeps its side). */
  team: number;
  x: number;
  y: number;
  /** Travel direction, radians, same convention as heading — independent of the
   *  owner's current heading because tanks can turn after firing. */
  dir: number;
  /** Ricochets remaining (BULLET_BOUNCES or BOUNCE_POWERUP at spawn, copied).
   *  0 ⇒ the next wall hit expires the bullet instead of reflecting it. */
  bouncesLeft: number;
  /** Pierce active at fire time: passes through tanks (each exactly once). */
  pierce: boolean;
  /** Seconds of flight remaining; expiry docks REWARD_BULLET_EXPIRED. */
  life: number;
  /** Ids of tanks this bullet has already damaged — what makes pierce "once per
   *  tank" and non-pierce "hits one tank then dies" the same bookkeeping. */
  hit: number[];
}

/** A powerup lying on the field. */
export interface PowerUp {
  id: number;
  kind: PowerUpKind;
  x: number;
  y: number;
  /** Absolute tick at which it disappears; despawnFrac in the obs is derived
   *  from this so urgency is visible to the policy without a timer field. */
  despawn: number;
}

/** Discrete juice/logging events emitted by the engine tick — the single
 *  channel the renderer's effects/audio consume. Order within a tick is the
 *  step order, so consumers can rely on "kill" arriving after its "hit". */
export type EventKind =
  | "fire"
  | "hit"
  | "kill"
  | "explode"
  | "pickup"
  | "ricochet"
  | "spawn"
  | "phase"
  | "crate";

/** One gameplay event. Optional fields are present only where they make sense
 *  (angle on fire/ricochet, tank on hit/kill, powerup on pickup, phase on phase)
 *  so effects code never has to guess which fields to read per kind. */
export interface GameEvent {
  kind: EventKind;
  /** World-space anchor for particles/sound panning. */
  x: number;
  y: number;
  angle?: number;
  /** Tank id the event concerns (victim for kill/hit). */
  tank?: number;
  powerup?: PowerUpKind;
  phase?: Phase;
}

/** The whole mutable world, plus the bookkeeping that makes it deterministic
 *  and resumable. Everything here is worker-local; the main thread only ever
 *  sees serialized snapshots of it. */
export interface GameState {
  /** Ticks elapsed since episode start; the engine's sole clock. */
  tick: number;
  tanks: Tank[];
  bullets: Bullet[];
  powerups: PowerUp[];
  /** Static indestructible geometry, fixed for the whole episode (the {arena}
   *  message ships it once, so snapshots stay small). */
  walls: Obstacle[];
  /** Destructible cover, same layout lifetime as walls (hp counts down). */
  crates: Obstacle[];
  /** Legal spawn points (arena-generated, connectivity-checked); the first N are
   *  used at episode start, later respawns pick from the free ones. */
  spawns: Vec[];
  phase: Phase;
  /** Tick budget for this episode (EPISODE_SECONDS_DEFAULT × TICK_HZ, clamped to
   *  MAX_TICKS_CAP); `tick >= maxTicks` ⇒ timeout placement scoring. */
  maxTicks: number;
  /** Absolute tick of the next powerup spawn — scheduled, never polled, so a
   *  paused/resumed lane cannot double-spawn. */
  nextPowerupTick: number;
  /** The PRNG counter threaded through every random draw (rng.ts). An integer,
   *  because a closure's hidden counter could not be checkpointed or replayed. */
  rngState: number;
  teamMode: TeamMode;
  /** Terminal flag: set on last-man-standing or tick-out; lanes must stop
   *  stepping but keep the state around for the end-of-episode report. */
  over: boolean;
  /** Seed the arena layout was generated from — the {arena} message and the
   *  replay/property tests key off it, so a layout is always reproducible. */
  arenaSeed: number;
  /** Last tick's reward per tank, index-aligned with `tanks` — lets the
   *  renderer/telemetry show float rewards without recomputing them. */
  lastRewards: number[];
  /** Discount the potential-shaping terms are built with. MUST equal the
   *  learner's NetConfig.gamma — γΦ(s′)−Φ(s) is policy-neutral for the γ it is
   *  constructed with, and a mismatch silently voids the Ng et al. (1999)
   *  guarantee (rewards.ts). createGame copies it from GameConfig so a lane can
   *  never drift from the net that trains on its rewards. */
  gamma: number;
  /** Events emitted THIS tick; cleared at the start of the next step, and
   *  drained by whoever serializes the snapshot. */
  events: GameEvent[];
  nextBulletId: number;
  nextPowerupId: number;
}
