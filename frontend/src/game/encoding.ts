// encoding.ts — the 282-float observation vector ONE agent sees.
// DOM-free, tfjs-free: this runs inside the train worker for every agent on
// every tick and inside node vitest, so no DOM/window/@tensorflow/Math.random
// may appear here, and it must stay a PURE function of GameState.
//
// WHY a fixed-slot flat vector instead of a variable-length entity list: the
// plan pins ONE MLP input (282 → 256 → 256 → 18) and the replay ring stores
// rows of exactly OBS_DIM floats, so the world must project into a constant
// shape. Entities that do not fit (a 4th tank, a 19th bullet) are not squeezed
// into a corner — they fall off the SORTED end and their slot reads the
// all-zero sentinel row (valid = 0), which the net learns to ignore. Sorting
// nearest-first is what makes "slot 0" mean "the thing that matters most" and
// makes the encoding a stable function of the world, not of entity ids.
//
// WHY everything is relative to the observer's own heading (for a world offset
// (dx,dy):  bx = dx·cos h + dy·sin h,  by = −dx·sin h + dy·cos h): self-play
// FFA shares ONE network across N agents whose hulls point any way at all. A
// body-frame encoding makes "tank 300 u dead ahead" look identical from every
// hull angle, so experience transfers between agents and seats, and no
// capacity is wasted re-learning rotations the world can already factor out.
// Bearings are atan2(by,bx)/π ∈ (−1, 1]; +0.5 therefore always means "90° to
// the hull's right" in y-down screen space.
//
// CONTRACT (plan §Observation — asserted index-by-index in encoding.spec.ts):
//   SELF 16 | TANKS 3×12 | BULLETS 18×7 | POWERUPS 4×6 | RAYS 16×5 = 282,
// every value clamped to [−1, 1], sentinel rows all zero, never NaN.
// All distances share the single normalizer NORM_DIST (arena diagonal), so a
// distance feature can never leave [0, 1] anywhere in the arena.
//
// Determinism: no randomness, no clocks, no field the engine does not already
// store — a paused/resumed lane or a replayed transition re-derives the
// bit-identical vector. Dead tanks are still valid observers (the engine needs
// their terminal transition) but are never ENTITIES: a corpse cannot act, so
// it occupies no TANKS slot and blocks no ray.

import {
  BOUNCE_POWERUP,
  BOUNCE_TIME,
  BULLET_SPEED,
  FIRE_COOLDOWN,
  MAX_BULLETS_GLOBAL,
  MAX_BULLETS_PER_TANK,
  MAX_SPEED_FWD,
  NORM_DIST,
  OBS_DIM,
  PIERCE_TIME,
  POWERUP_LIFETIME,
  RAPID_TIME,
  RAY_COUNT,
  RAY_LEN,
  SHIELD_TIME,
  SPEED_TIME,
  TANK_HP,
  TANK_RADIUS,
  TICK_HZ,
} from "./constants";
// physics.ts is a sibling engine module; `raycast(game, x, y, dir, maxLen)` is
// the pinned signature this file codes against (returns the NEAREST hit at or
// before maxLen as { dist, kind: "wall"|"crate"|"tank"|null, tankId }, with
// dist === maxLen when nothing is in range).
import { raycast } from "./physics";
import type { GameState, PowerUpKind, Tank } from "./types";

/** Total vector length. Re-exported rather than redefined: constants.ts owns
 *  the literal because the network input layer, the replay row width and this
 *  encoder must quote the SAME number, and a second copy is exactly the silent
 *  drift the constants module exists to prevent. */
export { OBS_DIM };

/** First float index of each block — the load-bearing layout table. Anything
 *  that slices the vector (attention tokens, HUD debug, tests) reads these
 *  instead of hard-coding an offset. */
export const OFF = { SELF: 0, TANKS: 16, BULLETS: 52, POWERUPS: 178, RAYS: 202 } as const;

/** Stride of one entity row within its block: 16 + 3·12 + 18·7 + 4·6 + 16·5
 *  must equal OBS_DIM (encoding.spec.ts pins the arithmetic, not just the
 *  constants, so a stride/offset mismatch cannot ship). */
export const TANK_SLOT = 12;
export const BULLET_SLOT = 7;
export const POWERUP_SLOT = 6;
export const RAY_SLOT = 5;

// Slot counts — module-private because callers never index past them by hand;
// they walk (OFF, *_SLOT) and read each row's valid flag instead.
/** Plan: FFA arena of 2–4 tanks ⇒ at most 3 OTHERS to encode. */
const TANK_SLOTS = 3;
/** Deliberately 18 < MAX_BULLETS_GLOBAL (24): a full 4-tank magazine dump does
 *  not fit, so the block keeps the 18 NEAREST and SELF[14] reports the overflow
 *  as an explicit "more chaos out there than I can see" feature. */
const BULLET_SLOTS = 18;
/** Equals MAX_POWERUPS_ON_FIELD — the cap the spawn pass enforces. */
const POWERUP_SLOTS = 4;

/** Ray origin offset from the tank centre u — just outside TANK_RADIUS, so the
 *  observer's own hull is transparent to its own rays (why: see the RAYS loop).
 *  RAY_LEN (300) ≫ TANK_RADIUS (20), so the shortened cast loses no range. */
const RAY_ORIGIN = TANK_RADIUS + 1e-6;

/** Index of each powerup kind in the observation. MUST mirror the declaration
 *  order of the `PowerUpKind` union in types.ts (rapid 0 … pierce 5), which is
 *  the order the plan's obs table pins; a swap here silently re-labels
 *  pickups for a trained net. Kept local: the network contract needs one
 *  authoritative table per module, not a second exported one. */
const POWERUP_IDX: Record<PowerUpKind, number> = {
  rapid: 0,
  shield: 1,
  speed: 2,
  repair: 3,
  bounce: 4,
  pierce: 5,
};

/** Combat powerups change how you FIGHT (fire rate, ricochet, punch-through);
 *  repair/shield/speed change how you SURVIVE. One feature splitting the six
 *  kinds into these two families lets the policy price "a pickup of the kind I
 *  want" without spending six one-hot slots per row. */
const POWERUP_COMBAT: readonly boolean[] = [true, false, false, false, true, true];

/**
 * Encode the world from `tankId`'s point of view into OBS_DIM floats.
 *
 * `out` may be a caller-owned buffer (the worker reuses one per agent per tick
 * — 282 floats × 4 agents × 30 Hz is churn worth avoiding); it is zero-filled
 * first, so callers always get a fully-written vector and never yesterday's
 * slot data. Without `out`, a fresh buffer is allocated.
 *
 * Throws RangeError for an unknown tank id or an out buffer shorter than
 * OBS_DIM — both are caller bugs (Float32Array writes past length are SILENT
 * no-ops, so a short buffer would otherwise ship a half-empty network input).
 */
export function encodeObs(game: GameState, tankId: number, out?: Float32Array): Float32Array {
  const buf = out ?? new Float32Array(OBS_DIM);
  if (buf.length < OBS_DIM) {
    throw new RangeError(`encodeObs: out buffer holds ${buf.length} floats, OBS_DIM is ${OBS_DIM}`);
  }
  // Zero, then write: the sentinel row IS the zero row, and a reused buffer
  // must not leak rows from the previous tick into slots this world leaves
  // empty (a Float32Array is plain memory — there is no "unset" to rely on).
  buf.fill(0, 0, OBS_DIM);
  const me = tankById(game, tankId);
  const cosH = Math.cos(me.heading);
  const sinH = Math.sin(me.heading);

  // --- SELF (OFF.SELF + 0..15) — my own state, in my own frame ---------------
  // [0] signed speed ÷ max forward: negative means reversing, so "backing off"
  // and "advancing" are one continuous feature instead of a hidden sign.
  put(buf, OFF.SELF + 0, me.speed / MAX_SPEED_FWD);
  // [1] RESERVED (was considered for angular velocity ÷ TURN_RATE). The engine
  // stores no per-tick turn history and this encoder must stay a pure function
  // of GameState — a paused, resumed or replayed lane has to re-derive the
  // identical vector — so an invented 0 "signal" is not written. The slot stays
  // a documented constant 0: layout stability now, an honest feature later.
  put(buf, OFF.SELF + 1, 0);
  put(buf, OFF.SELF + 2, me.hp / TANK_HP);
  // [3] reload readiness: 0 = just fired, 1 = gun free. Rapid fire shrinks the
  // cooldown, so a rapid-loaded tank reads ≥ 1 − RAPID_COOLDOWN/FIRE_COOLDOWN.
  // Floored at 0 (not just ±1): a negative readiness is meaningless, and the
  // engine's base cooldown never exceeds FIRE_COOLDOWN anyway.
  put(buf, OFF.SELF + 3, Math.max(0, 1 - me.reload / FIRE_COOLDOWN));
  put(buf, OFF.SELF + 4, me.bulletsInFlight / MAX_BULLETS_PER_TANK);
  // [5..9] powerup timers as FRACTION OF FULL DURATION remaining — 1.0 the
  // instant the pickup lands, → 0 as it runs out, so urgency is scale-free.
  put(buf, OFF.SELF + 5, me.timers.rapid / RAPID_TIME);
  put(buf, OFF.SELF + 6, me.timers.shield / SHIELD_TIME);
  put(buf, OFF.SELF + 7, me.timers.speed / SPEED_TIME);
  put(buf, OFF.SELF + 8, me.timers.bounce / BOUNCE_TIME);
  put(buf, OFF.SELF + 9, me.timers.pierce / PIERCE_TIME);
  // [10] team-mode flag: in "allied" the isAlly/ownerCode features below are
  // meaningful; in "ffa" they are structural zeros the net can discount.
  put(buf, OFF.SELF + 10, game.teamMode === "allied" ? 1 : 0);
  let alliesAlive = 0;
  for (const t of game.tanks) {
    if (t.id !== me.id && t.alive && isAlly(game, me, t)) alliesAlive++;
  }
  put(buf, OFF.SELF + 11, alliesAlive / 3); // 3 = max allies in a 4-tank field
  put(buf, OFF.SELF + 12, me.kills / 3);
  put(buf, OFF.SELF + 13, me.deaths / 3);
  // [14] bullets beyond the 18 encoded slots ÷ global cap: "more chaos out
  // there than I can see" — the arena can hold 24, the vector holds 18.
  put(buf, OFF.SELF + 14, Math.max(0, game.bullets.length - BULLET_SLOTS) / MAX_BULLETS_GLOBAL);
  // [15] spare — deliberately zero, keeping the block 16 floats wide.
  put(buf, OFF.SELF + 15, 0);

  // --- TANKS (3 nearest ALIVE others, dist ASC) ------------------------------
  const others = byDistance(
    game.tanks.filter((t) => t.alive && t.id !== me.id),
    me,
  );
  for (let s = 0; s < TANK_SLOTS; s++) {
    const e = others[s];
    if (!e) continue; // slot stays the all-zero sentinel (valid = 0)
    const t = e.entity;
    const base = OFF.TANKS + s * TANK_SLOT;
    const dx = t.x - me.x;
    const dy = t.y - me.y;
    put(buf, base + 0, 1); // valid
    put(buf, base + 1, e.d / NORM_DIST);
    put(buf, base + 2, bearing(dx, dy, cosH, sinH));
    // Their hull angle in MY frame: 0 = facing the way I face, ±1 = facing
    // opposite — enough to read "is this tank charging me or running".
    put(buf, base + 3, wrapPi(t.heading - me.heading) / Math.PI);
    put(buf, base + 4, t.speed / MAX_SPEED_FWD);
    put(buf, base + 5, t.hp / TANK_HP);
    put(buf, base + 6, isAlly(game, me, t) ? 1 : 0);
    put(buf, base + 7, Math.max(0, 1 - t.reload / FIRE_COOLDOWN)); // can THEY shoot now
    put(buf, base + 8, t.bulletsInFlight / MAX_BULLETS_PER_TANK);
    put(buf, base + 9, t.timers.shield > 0 ? 1 : 0); // shooting them is wasted
    // [10] closing speed: + = the gap is shrinking, − = opening. Uses BOTH
    // velocities (theirs minus mine), so "I charge a parked tank" and "a
    // charger comes at me while I stand" read the same — which is the point.
    put(buf, base + 10, closingSpeed(me, t, dx, dy, e.d));
    put(buf, base + 11, t.kills / 3); // threat hint: who is winning
  }

  // --- BULLETS (18 nearest, dist ASC) ----------------------------------------
  const shots = byDistance(game.bullets, me);
  for (let s = 0; s < BULLET_SLOTS; s++) {
    const e = shots[s];
    if (!e) continue;
    const b = e.entity;
    const base = OFF.BULLETS + s * BULLET_SLOT;
    put(buf, base + 0, 1); // valid
    put(buf, base + 1, e.d / NORM_DIST);
    put(buf, base + 2, bearing(b.x - me.x, b.y - me.y, cosH, sinH));
    // Travel direction in my frame — with bearing this separates "coming at
    // me" (bearing ~ 0, dirRel ~ ±1) from "crossing my path" (dirRel ~ ±0.5).
    put(buf, base + 3, wrapPi(b.dir - me.heading) / Math.PI);
    // Every bullet flies at BULLET_SPEED today, so this reads 1.0 by
    // construction — kept as a feature (not folded into bearing) because it is
    // what makes the row self-describing if bullet speeds ever diverge.
    put(buf, base + 4, BULLET_SPEED / BULLET_SPEED);
    // [5] owner: 1 mine (I must not dodge myself), 0.5 an ally's (allied mode
    // only — FFAs have no allies so 0.5 never occurs there), 0 enemy (dodge).
    put(buf, base + 5, b.ownerId === me.id ? 1 : isAllyTeam(game, me, b.team) ? 0.5 : 0);
    put(buf, base + 6, b.bouncesLeft / BOUNCE_POWERUP); // will it ricochet again
  }

  // --- POWERUPS (4 nearest, dist ASC) ----------------------------------------
  const pickups = byDistance(game.powerups, me);
  for (let s = 0; s < POWERUP_SLOTS; s++) {
    const e = pickups[s];
    if (!e) continue;
    const p = e.entity;
    const base = OFF.POWERUPS + s * POWERUP_SLOT;
    put(buf, base + 0, 1); // valid
    put(buf, base + 1, e.d / NORM_DIST);
    put(buf, base + 2, bearing(p.x - me.x, p.y - me.y, cosH, sinH));
    const idx = POWERUP_IDX[p.kind];
    put(buf, base + 3, idx / 5); // kind, ordinal-scaled into [0, 1]
    put(buf, base + 4, POWERUP_COMBAT[idx] ? 1 : 0);
    // Seconds left before it despawns ÷ full lifetime: a shared floor timer, so
    // "0.05" means sprint now regardless of kind.
    put(buf, base + 5, (p.despawn - game.tick) / TICK_HZ / POWERUP_LIFETIME);
  }

  // --- RAYS (16, body-relative, k = 0 dead ahead) ----------------------------
  for (let k = 0; k < RAY_COUNT; k++) {
    const base = OFF.RAYS + k * RAY_SLOT;
    const dir = me.heading + (k * 2 * Math.PI) / RAY_COUNT;
    // Cast from the HULL SURFACE and put the offset back, instead of from the
    // centre. WHY: raycast() has no observer notion, and a body containing the
    // ray origin reports as a hit at range 0 — so a centre cast would hit MY
    // OWN hull every time and all 16 rays would read "tank, 0 u" forever.
    // Starting just outside my radius makes my own body transparent while every
    // OTHER hit stays bit-identical to a centre cast (hit.dist + RAY_ORIGIN
    // re-references to the centre), which is the perception the plan describes.
    // A caster that learns to skip the origin tank returns the same answers
    // through this path unchanged, so no coordinated edit is needed.
    const hit = raycast(
      game,
      me.x + Math.cos(dir) * RAY_ORIGIN,
      me.y + Math.sin(dir) * RAY_ORIGIN,
      dir,
      Math.max(0, RAY_LEN - RAY_ORIGIN),
    );
    // Belt-and-braces own-id guard (a caster that reports the origin tank at a
    // nonzero range would otherwise show as a self-hit): never information.
    const tankHit = hit.kind === "tank" && hit.tankId !== me.id;
    // A miss returns maxLen, which + RAY_ORIGIN re-references to exactly
    // RAY_LEN ⇒ "clear to the edge of perception" reads 1.0.
    put(buf, base + 0, (hit.dist + RAY_ORIGIN) / RAY_LEN);
    put(buf, base + 1, hit.kind === "wall" ? 1 : 0);
    put(buf, base + 2, hit.kind === "crate" ? 1 : 0);
    put(buf, base + 3, tankHit ? 1 : 0);
    put(buf, base + 4, tankHit ? teamSign(game, me, hit.tankId) : 0);
  }

  return buf;
}

// --- helpers ------------------------------------------------------------------

/**
 * Clamp-and-write one feature. EVERY value in the vector passes through here,
 * so the [−1, 1] range the input layer (and any tanh-shaped initialization
 * assumption) expects holds by construction rather than by discipline.
 * NaN collapses to 0: a non-finite feature is a bug upstream, but one bad
 * float must not poison a whole replay batch through the network input.
 */
function put(out: Float32Array, i: number, v: number): void {
  out[i] = v !== v ? 0 : v < -1 ? -1 : v > 1 ? 1 : v;
}

/** Look up the observer. Tanks are never removed from `game.tanks` (corpses
 *  persist until the episode resets), so a miss means a bogus id — a caller
 *  bug worth a hard throw, not a silently empty vector. */
function tankById(game: GameState, id: number): Tank {
  for (const t of game.tanks) if (t.id === id) return t;
  throw new RangeError(`encodeObs: tank ${id} is not in this state`);
}

/** True when `other` fights on my side. Only possible in "allied" mode, where
 *  team 0 is shared; FFA gives every tank a unique team, so this is a
 *  structural false there and every isAlly feature degrades to a constant. */
function isAlly(game: GameState, me: Tank, other: Tank): boolean {
  return game.teamMode === "allied" && other.team === me.team;
}

/** isAlly for a bullet, which carries its owner's team COPIED AT FIRE TIME —
 *  the right thing to test, because a posthumous bullet keeps its side even
 *  though `tankById(owner)` would still work while corpses persist. */
function isAllyTeam(game: GameState, me: Tank, team: number): boolean {
  return game.teamMode === "allied" && team === me.team;
}

/** +1 ally / −1 enemy for a ray's tank hit. Defaults hostile: a hit the lookup
 *  cannot resolve (impossible by construction) must read as a threat, never as
 *  a friend — threat features should fail loud, not quiet. */
function teamSign(game: GameState, me: Tank, tankId: number): number {
  const t = game.tanks.find((c) => c.id === tankId);
  if (!t) return -1;
  return isAlly(game, me, t) ? 1 : -1;
}

/**
 * Rotate a world offset into my body frame and normalize its direction to
 * (−1, 1] by ÷π. bx/by are the forward/left components (bx > 0 = ahead).
 * The dx==0 && dy==0 guard is the atan2(0,0) case: JS happens to define it as
 * 0, but the branch documents that a co-located entity has NO meaningful
 * bearing and must not emit whatever the platform's zero-signs give.
 */
function bearing(dx: number, dy: number, cosH: number, sinH: number): number {
  if (dx === 0 && dy === 0) return 0;
  const bx = dx * cosH + dy * sinH;
  const by = -dx * sinH + dy * cosH;
  return Math.atan2(by, bx) / Math.PI;
}

/** Wrap an angle into [−π, π) so a heading DIFFERENCE is always the short way
 *  round: 3/4 turn CW must read as −1/4 turn CCW, not +3/4. */
function wrapPi(a: number): number {
  const twoPi = 2 * Math.PI;
  let w = (a + Math.PI) % twoPi;
  if (w < 0) w += twoPi;
  return w - Math.PI;
}

/**
 * −d(dist)/dt to `other`, i.e. how fast the gap is closing, normalized by max
 * forward speed so two max-speed head-ons clamp at 1 rather than overflowing.
 * Positive = closing, negative = opening; 0 when coincident (the unit vector
 * is undefined there and any answer would be noise). Assumes tanks move only
 * along their hull axis — true by construction (see types.ts `Tank.speed`).
 */
function closingSpeed(me: Tank, other: Tank, dx: number, dy: number, dist: number): number {
  if (dist < 1e-6) return 0;
  const ux = dx / dist;
  const uy = dy / dist;
  const mvx = me.speed * Math.cos(me.heading);
  const mvy = me.speed * Math.sin(me.heading);
  const tvx = other.speed * Math.cos(other.heading);
  const tvy = other.speed * Math.sin(other.heading);
  return -((tvx - mvx) * ux + (tvy - mvy) * uy) / MAX_SPEED_FWD;
}

/**
 * Nearest-first candidate list: pairs each entity with its world distance,
 * sorts ascending (id as the deterministic tiebreak so equal distances never
 * depend on array insertion order), WITHOUT mutating GameState's own arrays —
 * the engine's ordering is part of its step-order determinism and the encoder
 * must not perturb it. Local allocation is fine here: a handful of short-lived
 * arrays per agent-tick, nursery-collected, versus a selection network that
 * would obscure the intent for no measurable gain at n ≤ 24.
 */
function byDistance<T extends { x: number; y: number; id: number }>(
  entities: T[],
  me: Tank,
): { entity: T; d: number }[] {
  return entities
    .map((entity) => ({ entity, d: Math.hypot(entity.x - me.x, entity.y - me.y) }))
    .sort((a, b) => a.d - b.d || a.entity.id - b.entity.id);
}
