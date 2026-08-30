// engine.ts — the episode driver: world construction and the fixed-tick step.
// DOM-free, tfjs-free (the single train worker ticks 16+ lanes with this, and
// node vitest plays thousands of games with it), and fully deterministic:
// the only randomness is the seeded rngState threaded through randInt, and
// the step order below is FIXED because every stage reads the previous stage's
// output (a bullet spawns at the post-turn muzzle; a blast hits tanks whose
// position is already this tick's).
//
// STEP ORDER (plan §World constants — pinned, unit-tested by engine.spec):
//   capture reward view → tick++ → clear events/lastRewards/per-tick flags →
//   (1) tanks integrate (+containment re-resolve) → (2) timers decay + fire →
//   (3) bullets (2 substeps) → (4) powerups (pickup → spawn → despawn) →
//   (5) crate blasts → (6) deaths → (7) escalation + sudden-death drain →
//   (8) terminal check (+timeout placement) → (9) shaped rewards.
//
// WHY the extra containment sweeps after stage 1: integrateTank resolves each
// tank's wall/crate contact BEFORE its tank-vs-tank separation, so a tank
// pushed by a neighbour can end the tick a few units inside a cover face until
// its next resolve. That window is exactly what engine.spec's containment
// property forbids (centre ≥ 0.9·TANK_RADIUS from every solid face/corner —
// and what a renderer would show as clipping), so the engine re-resolves at
// dt = 0 — integrateTank with dt = 0 performs pure collision resolution (no
// turn, no move), reusing physics' single collision implementation instead of
// a second copy that could drift. Two sweeps: the first re-separates pairs (a
// crate escape can shove a tank onto a neighbour), the second runs with an
// EMPTY tank list so geometry alone has the last word on every position.

import { decode, ACTION_COUNT } from "./actions";
import type { Action } from "./actions";
import { generateArena } from "./arena";
import {
  ARENA_H,
  ARENA_W,
  BOUNCE_TIME,
  CRATE_BLAST_DAMAGE,
  CRATE_BLAST_RADIUS,
  DT,
  ESCALATION_FRAC,
  MAX_POWERUPS_ON_FIELD,
  MAX_TICKS_CAP,
  MUZZLE_OFFSET,
  PIERCE_TIME,
  POWERUP_LIFETIME,
  POWERUP_SPAWN_MAX_TICKS,
  POWERUP_SPAWN_MIN_TICKS,
  RAPID_TIME,
  REPAIR_AMOUNT,
  REWARD_PICKUP,
  REWARD_PICKUP_HEAL_BONUS,
  SHIELD_TIME,
  SPEED_TIME,
  SUDDEN_DEATH_DRAIN,
  SUDDEN_DEATH_FRAC,
  TANK_HP,
  TANK_RADIUS,
  TICK_HZ,
} from "./constants";
import { integrateBullets, integrateTank, tryFire } from "./physics";
import { randInt } from "./rng";
import { REWARD_GAMMA, captureRewardView, placementBonus, shapedStep } from "./rewards";
import type { GameState, PowerUpKind, Tank, TeamMode, Vec } from "./types";

/** Everything a lane needs to build one episode. */
export interface GameConfig {
  /** 2–4 tanks; the first `tankCount` arena spawns are used in order. */
  tankCount: 2 | 3 | 4;
  /** "ffa": every tank its own team. "allied": tank 0 (the human seat in play
   *  mode) is team 0, the AI opponents are team 1 — the confirmed
   *  "AI-team-vs-you" layout; physics blocks same-team bullet damage, so
   *  allied damage can only ever come from crate blasts. */
  teamMode: TeamMode;
  /** Episode length in seconds; capped by MAX_TICKS_CAP (90 s). */
  episodeSeconds: number;
  /** Episode seed: drives the arena layout AND the engine rngState. */
  seed: number;
  /** The learner's discount γ, threaded into the reward pass so the shaping
   *  terms γΦ(s′)−Φ(s) are built with the SAME γ the value function bootstraps
   *  with — the coupling the Ng et al. (1999) policy-neutrality proof needs.
   *  Optional only so pure-physics/reward fixtures can omit it: it defaults to
   *  REWARD_GAMMA (rewards.ts), the value defaultConfig().gamma pins. */
  gamma?: number;
}

/** Radius of a powerup's pickup circle u (added to TANK_RADIUS). Smaller than
 *  the 22×22 sprite: grabbing a heart must feel deliberate, not magnetic. */
const PICKUP_RADIUS = 14;

/** Powerup spawn candidates keep this much clearance u from every wall/crate
 *  face so an icon is never wedged into a corner its 34 u pickup circle can
 *  barely reach (and never overlaps the 20 u tank bodies that patrol past). */
const POWERUP_CLEAR = 30;

/** Powerup positions keep ≥ this distance u from every ALIVE tank (plan:
 *  no spawn-on-tank freebies, and a visible dash for everyone). */
const POWERUP_MIN_TANK_DIST = 150;

/** Rejection-sampling budget per spawn. A generated arena is mostly open floor
 *  (≥85% reachable), so 40 tries fail only on a packed endgame board with all
 *  tanks clustered — and then skipping the spawn is the correct, deterministic
 *  answer (the schedule simply retries next tick). */
const POWERUP_TRIES = 40;

/** Margin u from the arena border for spawn sampling — keeps every draw inside
 *  the sampled span on the floor (no clamping, no re-rolls). */
const POWERUP_MARGIN = 30;

/**
 * Powerup kind roulette over a single integer draw in [0, 100). Cumulative
 * thresholds are the plan's weights (.20 rapid, .15 shield, .20 speed,
 * .20 repair, .15 bounce, .10 pierce); the LAST matching row must be the
 * catch-all so a threshold edit can never yield `undefined`.
 */
const POWERUP_ROLL: ReadonlyArray<{ upTo: number; kind: PowerUpKind }> = [
  { upTo: 20, kind: "rapid" },
  { upTo: 35, kind: "shield" },
  { upTo: 55, kind: "speed" },
  { upTo: 75, kind: "repair" },
  { upTo: 90, kind: "bounce" },
  { upTo: 100, kind: "pierce" },
];

/** decode(2) = coast/straight/hold — the identity action for the containment
 *  pass and the safe fallback for a malformed action entry. */
const COAST: Action = decode(2);

/**
 * The raw action index for tank `i`. Total over any input: a non-integer,
 * negative, ≥ ACTION_COUNT or missing entry (a desynced buffer, a shorter
 * array) collapses to 2 = coast/straight/hold. decode() already clamps, but
 * its clamp lands on the TABLE EDGE (18 → rev/right/fire) — the contract here
 * is "garbage input coasts", so the guard runs before decode, not after.
 */
function rawAction(actions: number[], i: number): number {
  const a = actions[i];
  return typeof a === "number" && Number.isInteger(a) && a >= 0 && a < ACTION_COUNT ? a : 2;
}

/**
 * Build one episode. Deterministic in (seed, tankCount, teamMode,
 * episodeSeconds): the arena is generated from the seed, tanks take the first
 * spawns FACING THE ARENA CENTRE (everyone opens with a route into the fight,
 * not a wall), and the first powerup is scheduled on the same uniform
 * [POWERUP_SPAWN_MIN_TICKS, POWERUP_SPAWN_MAX_TICKS] gap every later spawn
 * uses — no special first-spawn rule to remember.
 */
export function createGame(cfg: GameConfig): GameState {
  const layout = generateArena(cfg.seed, cfg.tankCount);
  const maxTicks = Math.min(Math.ceil(cfg.episodeSeconds * TICK_HZ), MAX_TICKS_CAP);
  // Mix the seed before first use: rngState and the arena generator would
  // otherwise share their whole draw sequence, correlating layout shape with
  // powerup scheduling across episodes.
  let rngState = (cfg.seed ^ 0x9e3779b9) | 0;
  const [gap, s1] = randInt(rngState, POWERUP_SPAWN_MAX_TICKS - POWERUP_SPAWN_MIN_TICKS + 1);
  rngState = s1;
  // Crate identity counter (types.ts Obstacle.id): local to this episode is
  // enough — a bullet cannot outlive its own game, and crate ids only need to
  // be unique WITHIN the episode a bullet flies in.
  let nextCrateId = 0;
  const tanks: Tank[] = layout.spawns.slice(0, cfg.tankCount).map((sp, i) => ({
    id: i,
    x: sp.x,
    y: sp.y,
    heading: Math.atan2(ARENA_H / 2 - sp.y, ARENA_W / 2 - sp.x),
    speed: 0,
    hp: TANK_HP,
    alive: true,
    reload: 0,
    bulletsInFlight: 0,
    kills: 0,
    deaths: 0,
    // ffa: team == id (everyone hittable). allied: human seat vs the AI team.
    team: cfg.teamMode === "allied" ? (i === 0 ? 0 : 1) : i,
    // P1 draws the tank_01 art, everyone else tank_02 — a readability choice
    // (the human seat always looks like itself), encoded here so the renderer
    // and the replay agree without a second rule.
    sprite: i === 0 ? 0 : 1,
    colorIdx: i,
    timers: { rapid: 0, shield: 0, speed: 0, bounce: 0, pierce: 0 },
    wallContact: false,
    diedThisTick: false,
  }));
  return {
    tick: 0,
    tanks,
    bullets: [],
    powerups: [],
    // Copied per episode on purpose: integrateBullets SPLICES broken crates
    // out of game.crates, and the fallback layout is a shared singleton —
    // handing it the raw arrays would let one episode delete another's cover.
    // Each crate also gets a STABLE id (array position shifts under every
    // splice, and a live bullet's hit-marks outlive the shift by up to
    // BULLET_LIFETIME).
    walls: layout.walls.map((w) => ({ ...w })),
    crates: layout.crates.map((c) => ({ ...c, id: nextCrateId++ })),
    spawns: layout.spawns.map((p) => ({ ...p })),
    phase: 0,
    maxTicks,
    nextPowerupTick: POWERUP_SPAWN_MIN_TICKS + gap,
    rngState,
    teamMode: cfg.teamMode,
    over: false,
    arenaSeed: cfg.seed,
    // The shaping discount rides with the episode (see GameConfig.gamma).
    gamma: cfg.gamma ?? REWARD_GAMMA,
    lastRewards: new Array<number>(cfg.tankCount).fill(0),
    events: [],
    nextBulletId: 0,
    nextPowerupId: 0,
  };
}

/**
 * Advance one fixed tick. Mutates `game` in place; `actions[i]` drives tank i
 * (see rawAction for the malformed-input rule). After `over` is set the step
 * is a no-op — a lane that keeps calling must not consume rng draws or pay
 * rewards twice for the terminal tick.
 */
export function step(game: GameState, actions: number[]): void {
  if (game.over) return;
  // Freeze the pre-tick view FIRST: every reward term needs these values and
  // nothing below may observe a half-ticked world through them.
  const prev = captureRewardView(game);
  game.tick++;
  // Fresh arrays (not in-place clears) so a snapshot/serializer holding last
  // tick's events keeps a stable view of them.
  game.events = [];
  game.lastRewards = new Array<number>(game.tanks.length).fill(0);
  // Per-tick flags must never survive into the next tick's reward pass.
  // wallContact is cleared for EVERY tank here and OR'd back in by the
  // integrate passes below: a tank parked exactly on a face produces no
  // correction (physics' push>0 rule) and must read false, while a grinding
  // tank must keep the flag its first pass raised through the two dt = 0
  // re-resolves (which land exactly on the face and so correct nothing).
  for (const t of game.tanks) {
    t.diedThisTick = false;
    t.wallContact = false;
  }

  const acts = game.tanks.map((_, i) => decode(rawAction(actions, i)));

  // (1) tanks: turn, accelerate, move, resolve — then two dt = 0 re-resolves
  // (see header). Sweep 2 re-separates pairs; sweep 3 runs with an EMPTY tank
  // list so separation is a no-op and geometry gets the literal last word on
  // every tank's position — no tank is ever touched again after its own sweep
  // 3, so the containment invariant holds at tick end, not just mid-pass.
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (!t.alive) continue;
    integrateTank(t, acts[i], DT, game.walls, game.crates, game.tanks);
    const grinding = t.wallContact;
    integrateTank(t, COAST, 0, game.walls, game.crates, game.tanks);
    const reSeparated = t.wallContact;
    integrateTank(t, COAST, 0, game.walls, game.crates, []);
    // OR the three passes (see the flag clear above): active scraping and
    // tank-pile corrections flag; a body at rest exactly on a face does not.
    t.wallContact = grinding || reSeparated || t.wallContact;
  }

  // (2) timers decay, then fire on request. Timers decay BEFORE the fire check
  // so an expiring rapid/shield applies for exactly its full duration — a
  // bullet must snapshot the powerup character the tank still had when the
  // trigger was pulled (physics copies bounces/pierce at fire time).
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (!t.alive) continue;
    if (t.reload > 0) t.reload = Math.max(0, t.reload - DT);
    t.timers.rapid = Math.max(0, t.timers.rapid - DT);
    t.timers.shield = Math.max(0, t.timers.shield - DT);
    t.timers.speed = Math.max(0, t.timers.speed - DT);
    t.timers.bounce = Math.max(0, t.timers.bounce - DT);
    t.timers.pierce = Math.max(0, t.timers.pierce - DT);
    if (!acts[i].fire) continue;
    // Muzzle position is computed pre-fire (tryFire does not move the tank,
    // but the event should anchor where the shell was BORN, not where the
    // tank ended up).
    const mx = t.x + Math.cos(t.heading) * MUZZLE_OFFSET;
    const my = t.y + Math.sin(t.heading) * MUZZLE_OFFSET;
    if (tryFire(game, t)) {
      game.events.push({ kind: "fire", x: mx, y: my, angle: t.heading, tank: t.id });
    }
  }

  // (3) bullets. Crates are spliced out of game.crates as they break, so the
  // survivors of this snapshot ARE the crates that broke this tick — identity
  // comparison after the pass is the cheapest possible break log.
  const cratesBefore = game.crates.slice();
  integrateBullets(game, DT);
  // Physics subtracts damage without clamping; overkill floors at 0 so hp
  // stays in [0, TANK_HP] for the hp bars, the encoder and the rewards.
  for (const t of game.tanks) {
    if (t.hp < 0) t.hp = 0;
  }
  const broken = cratesBefore.filter((c) => game.crates.indexOf(c) === -1);

  // (4) powerups: pickup → spawn → despawn, in that order, so a fresh spawn
  // can never be collected in the same tick it appears (it spawns ≥150 u from
  // every tank anyway) and a despawn can never steal a same-tick pickup.
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (!t.alive) continue;
    for (let p = game.powerups.length - 1; p >= 0; p--) {
      const pu = game.powerups[p];
      const dx = t.x - pu.x;
      const dy = t.y - pu.y;
      const rr = TANK_RADIUS + PICKUP_RADIUS;
      if (dx * dx + dy * dy > rr * rr) continue;
      game.powerups.splice(p, 1);
      // Repair is the only instantaneous kind: it heals NOW or never (there
      // is no timer to carry), and only over-heals-bonus when it actually
      // healed — hoarding hearts at full HP is the base pickup, nothing more.
      let healed = false;
      switch (pu.kind) {
        case "rapid":
          t.timers.rapid = RAPID_TIME;
          break;
        case "shield":
          t.timers.shield = SHIELD_TIME;
          break;
        case "speed":
          t.timers.speed = SPEED_TIME;
          break;
        case "bounce":
          t.timers.bounce = BOUNCE_TIME;
          break;
        case "pierce":
          t.timers.pierce = PIERCE_TIME;
          break;
        case "repair":
          healed = t.hp < TANK_HP;
          t.hp = Math.min(TANK_HP, t.hp + REPAIR_AMOUNT);
          break;
      }
      game.events.push({ kind: "pickup", x: pu.x, y: pu.y, tank: t.id, powerup: pu.kind });
      game.lastRewards[i] += REWARD_PICKUP + (healed ? REWARD_PICKUP_HEAL_BONUS : 0);
    }
  }

  if (game.tick >= game.nextPowerupTick && game.powerups.length < MAX_POWERUPS_ON_FIELD) {
    const spot = samplePowerupSpot(game);
    if (spot) {
      const [roll, s2] = randInt(game.rngState, 100);
      game.rngState = s2;
      let kind: PowerUpKind = "pierce";
      for (const row of POWERUP_ROLL) {
        kind = row.kind;
        if (roll < row.upTo) break;
      }
      game.powerups.push({
        id: game.nextPowerupId++,
        kind,
        x: spot.x,
        y: spot.y,
        despawn: game.tick + POWERUP_LIFETIME * TICK_HZ,
      });
      game.events.push({ kind: "spawn", x: spot.x, y: spot.y, powerup: kind });
      // Schedule the NEXT spawn only on success — a failed placement retries
      // next tick (fresh draws) instead of pushing the schedule out, so the
      // spawn-to-spawn gap stays inside the scheduled [min, max] band.
      const [gap2, s3] = randInt(game.rngState, POWERUP_SPAWN_MAX_TICKS - POWERUP_SPAWN_MIN_TICKS + 1);
      game.rngState = s3;
      game.nextPowerupTick = game.tick + POWERUP_SPAWN_MIN_TICKS + gap2;
    }
  }
  // Expired floor items vanish (the blink in the last second is a renderer
  // concern; the sim has one honest rule: despawn tick, gone).
  if (game.powerups.length > 0) {
    game.powerups = game.powerups.filter((p) => p.despawn > game.tick);
  }

  // (5) crate blasts: every crate broken this tick punishes everyone standing
  // on it — shields are proof (they are the answer to "should I have backed
  // off?"). One explosion per crate, no tank field: this is juice, not the
  // bullet-expiry marker the reward pass reads.
  for (const c of broken) {
    const cx = c.x + c.w / 2;
    const cy = c.y + c.h / 2;
    game.events.push({ kind: "explode", x: cx, y: cy });
    const rr = CRATE_BLAST_RADIUS;
    for (const t of game.tanks) {
      if (!t.alive || t.timers.shield > 0) continue;
      const dx = t.x - cx;
      const dy = t.y - cy;
      if (dx * dx + dy * dy <= rr * rr) t.hp = Math.max(0, t.hp - CRATE_BLAST_DAMAGE);
    }
  }

  // (6) deaths: hp floored at 0 by the passes above; one shared kill routine
  // so a blast death and a shell death are indistinguishable downstream.
  for (const t of game.tanks) {
    if (t.alive && t.hp <= 0) killTank(game, t);
  }

  // (7) escalation — checked AFTER deaths so the phase a reward pass sees is
  // the phase this tick actually played out in. Sudden death wins the tie on
  // a short episode that crosses both thresholds in one tick (draining is the
  // stronger promise to the agents).
  const frac = game.tick / game.maxTicks;
  if (game.phase < 2 && frac >= SUDDEN_DEATH_FRAC) {
    game.phase = 2;
    game.events.push({ kind: "phase", x: ARENA_W / 2, y: ARENA_H / 2, phase: 2 });
  } else if (game.phase < 1 && frac >= ESCALATION_FRAC) {
    game.phase = 1;
    game.events.push({ kind: "phase", x: ARENA_W / 2, y: ARENA_H / 2, phase: 1 });
  }
  if (game.phase === 2) {
    const d = SUDDEN_DEATH_DRAIN * DT;
    for (const t of game.tanks) {
      if (!t.alive) continue;
      t.hp = Math.max(0, t.hp - d);
      if (t.hp <= 0) killTank(game, t); // the drain can kill — same routine
    }
  }

  // (8) terminal: last man standing, or the clock. A timeout (clock with two
  // or more still alive) pays the placement bonus; a last-man finish does not
  // (the kill already priced the outcome).
  let aliveCount = 0;
  for (const t of game.tanks) {
    if (t.alive) aliveCount++;
  }
  if (aliveCount <= 1 || game.tick >= game.maxTicks) {
    const byTimeout = aliveCount > 1 && game.tick >= game.maxTicks;
    game.over = true;
    if (byTimeout) {
      const bonus = placementBonus(game);
      for (let i = 0; i < bonus.length; i++) game.lastRewards[i] += bonus[i];
    }
  }

  // (9) rewards, last so they read the FINAL state of the tick (deaths,
  // phase, blast hp) — the same state the next tick's `prev` will be. The
  // shaping discount is this episode's own γ (GameConfig.gamma): shapedStep
  // must build γΦ(s′)−Φ(s) with the SAME γ the learner bootstraps with.
  const shaped = shapedStep(prev, game, game.gamma);
  for (let i = 0; i < shaped.length; i++) game.lastRewards[i] += shaped[i];
}

/**
 * Remove a tank from the living set: one death, one flag, one event pair. Its
 * shells die with it — silently (no expiry event, so the reward pass never
 * charges a dead tank for the magazines it took to the grave).
 */
function killTank(game: GameState, victim: Tank): void {
  victim.alive = false;
  victim.diedThisTick = true;
  victim.deaths++;
  victim.speed = 0;
  // Scoreboard credit: the same nearest-actor rule the reward pass documents
  // (rewards.ts header) — the shooter is almost always the tank closest to its
  // victim, and in FFA the residual error is unbiased across agents. Without
  // this, Tank.kills — the K/D the watch scoreboard and Metrics.killDeathRatio
  // publish — would never leave 0.
  let killer: Tank | null = null;
  let bestD2 = Infinity;
  for (const t of game.tanks) {
    if (!t.alive || t.id === victim.id || t.team === victim.team) continue;
    const dx = t.x - victim.x;
    const dy = t.y - victim.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      killer = t;
    }
  }
  if (killer) killer.kills++;
  const before = game.bullets.length;
  game.bullets = game.bullets.filter((b) => b.ownerId !== victim.id);
  victim.bulletsInFlight -= before - game.bullets.length;
  game.events.push({ kind: "kill", x: victim.x, y: victim.y, tank: victim.id });
  // Render-scale (80 u sheet frames) is the renderer's concern; no tank field
  // here — only physics' expired-miss explosions carry one.
  game.events.push({ kind: "explode", x: victim.x, y: victim.y });
}

/**
 * Rejection-sample a powerup position: inside the border margin, ≥
 * POWERUP_MIN_TANK_DIST from every ALIVE tank, POWERUP_CLEAR clear of every
 * wall/crate face. Consumes exactly two draws per try (win or lose) so the
 * rng stream stays a pure function of the tick sequence. Returns null after
 * POWERUP_TRIES failures — the caller skips this tick's spawn and retries on
 * the next (the schedule is not advanced on failure).
 */
function samplePowerupSpot(game: GameState): Vec | null {
  const spanX = ARENA_W - 2 * POWERUP_MARGIN;
  const spanY = ARENA_H - 2 * POWERUP_MARGIN;
  const minD2 = POWERUP_MIN_TANK_DIST * POWERUP_MIN_TANK_DIST;
  for (let tries = 0; tries < POWERUP_TRIES; tries++) {
    const [rx, s1] = randInt(game.rngState, spanX);
    game.rngState = s1;
    const [ry, s2] = randInt(game.rngState, spanY);
    game.rngState = s2;
    const x = POWERUP_MARGIN + rx;
    const y = POWERUP_MARGIN + ry;
    let ok = true;
    for (const t of game.tanks) {
      if (!t.alive) continue;
      const dx = t.x - x;
      const dy = t.y - y;
      if (dx * dx + dy * dy < minD2) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    for (const boxes of [game.walls, game.crates]) {
      for (const b of boxes) {
        if (
          x > b.x - POWERUP_CLEAR &&
          x < b.x + b.w + POWERUP_CLEAR &&
          y > b.y - POWERUP_CLEAR &&
          y < b.y + b.h + POWERUP_CLEAR
        ) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    if (ok) return { x, y };
  }
  return null;
}

/** Terminal predicate for lanes and tests. */
export function isOver(game: GameState): boolean {
  return game.over;
}
