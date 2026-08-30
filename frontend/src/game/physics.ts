// physics.ts — tank motion, fire control, bullet flight and line-of-sight.
// DOM-free, tfjs-free (imported by the worker engine AND node tests), and
// fully deterministic: every step is a pure function of the mutable state plus
// the fixed dt — no random draws at all (that is rng.ts's job, and physics
// uses none), no wall clocks, no iteration over Sets in a
// hash-dependent order (Set iteration in JS is insertion-ordered, which is why
// the deferred-removal bookkeeping below stays reproducible).
//
// WHY one module owning motion+bullets+rays: the plan's step order is
// actions → tanks (turn, accelerate, resolve) → fire → bullets → collisions,
// and each stage reads the stage before it's output (a bullet spawns at the
// muzzle of the post-turn hull; a ray starts from the post-resolve centre).
// Splitting them across files would force either a wider GameState contract or
// ordering assumptions nobody could unit-test; here physics.spec pins the order
// directly.
//
// Collision model: tanks are CIRCLES (TANK_RADIUS, see types.ts for why), cover
// is AABBs, so everything reduces to circle-vs-AABB and circle-vs-circle.
// Bullet anti-tunneling is geometric, not probabilistic: BULLET_SPEED/2
// substeps advance 840·dt/2 = 14 u per sweep, well under the 40 u tank diameter
// and any wall thickness, so a contact can never be stepped over.
//
// Determinism note on removals: bullets are removed by a single filter at the
// END of integrateBullets (never spliced mid-loop, which would skip neighbours)
// and crates are marked hp ≤ 0 during the pass but spliced once at the end —
// the deferral keeps the crate ARRAY stable for the tick so the deadCrates
// index list stays valid while it accumulates; hit-mark identity does NOT rely
// on it (crate marks key off the stable Obstacle.id, which survives splicing).

import type { Action } from "./actions";
import {
  ACCEL,
  ARENA_H,
  ARENA_W,
  BOUNCE_POWERUP,
  BRAKE_DECEL,
  BULLET_BOUNCES,
  BULLET_DAMAGE,
  BULLET_LIFETIME,
  BULLET_RADIUS,
  BULLET_SPEED,
  BULLET_SUBSTEPS,
  COAST_DECEL,
  FIRE_COOLDOWN,
  MAX_BULLETS_GLOBAL,
  MAX_BULLETS_PER_TANK,
  MAX_SPEED_FWD,
  MAX_SPEED_REV,
  MUZZLE_OFFSET,
  RAPID_COOLDOWN,
  SPEED_MULT,
  TURN_MULT,
  TURN_RATE,
  TANK_RADIUS,
} from "./constants";
import type { Bullet, GameEvent, GameState, Obstacle, Tank } from "./types";

/** Nearest hit along a ray. `dist` is measured from the ray origin to the
 *  surface (not the centre), and equals maxLen with kind === null on a miss —
 *  the encoder writes that straight into the RAYS block as dist/maxLen. */
export interface RayHit {
  dist: number;
  kind: "wall" | "crate" | "tank" | null;
  /** Hit tank's id, or −1 when kind is not "tank" (−1 rather than 0 so a
   *  consumer can never mistake the sentinel for tank 0). */
  tankId: number;
}

// --- tank integration --------------------------------------------------------

/**
 * Advance one tank by dt: steering, then the throttle/acceleration model, then
 * the move, then collision resolution. Order is the plan's step order and is
 * observable (turning before accelerating means a tank commits its NEW heading
 * for this tick's displacement — aim-then-move, which is what makes dodging a
 * turn commitment rather than a strafe).
 *
 * Acceleration model (all rates from constants, dt-scaled):
 *   fwd input  : moving backwards ⇒ brake toward 0 at BRAKE_DECEL, else
 *                accelerate toward +MAX_SPEED_FWD at ACCEL
 *   rev input  : mirror image toward −MAX_SPEED_REV
 *   coast/none : decay toward 0 at COAST_DECEL, never crossing it
 * The speed powerup multiplies the CAP (both directions) and TURN_RATE, never
 * the accelerations — a boosted tank reaches a higher top speed in the same
 * 0.5 s, which is what makes the pickup worth a detour.
 *
 * Resolution pushes this tank only (its callers integrate every tank in turn,
 * so a pair separates fully within one pass — see separateTanks). Sets
 * tank.wallContact true iff it touched an obstacle, another tank or the arena
 * border anywhere in this call; the reward pass docks REWARD_WALL_CONTACT on
 * that flag, so it must be recomputed every tick, never accumulated.
 */
export function integrateTank(
  tank: Tank,
  action: Action,
  dt: number,
  walls: Obstacle[],
  crates: Obstacle[],
  tanks: Tank[],
): void {
  // 1. steering — left (steer 0) is CCW = heading−, right (2) is CW = heading+,
  // in y-down screen space (see types.ts conventions). No wrap: a 2700-tick
  // episode accumulates at most ~380 rad, far inside double precision, and
  // cos/sin are periodic — wrapping would add a branch for zero benefit.
  const turnMult = tank.timers.speed > 0 ? TURN_MULT : 1;
  if (action.steer === 0) tank.heading -= TURN_RATE * turnMult * dt;
  else if (action.steer === 2) tank.heading += TURN_RATE * turnMult * dt;

  // 2. throttle → signed speed along the hull axis. Every branch clamps at the
  // target so a big dt can overshoot into the wrong sign (a coasting tick must
  // stop AT 0, not glide backwards through it).
  const speedMult = tank.timers.speed > 0 ? SPEED_MULT : 1;
  if (action.throttle === 1) {
    // forward input: brake out of reverse first, then accelerate forward
    if (tank.speed < 0) tank.speed = Math.min(0, tank.speed + BRAKE_DECEL * dt);
    else tank.speed = Math.min(MAX_SPEED_FWD * speedMult, tank.speed + ACCEL * dt);
  } else if (action.throttle === 2) {
    // reverse input: brake out of forward first, then accelerate backward
    if (tank.speed > 0) tank.speed = Math.max(0, tank.speed - BRAKE_DECEL * dt);
    else tank.speed = Math.max(-MAX_SPEED_REV * speedMult, tank.speed - ACCEL * dt);
  } else {
    // coast (or "no input"): friction toward 0, sign-preserving
    if (tank.speed > 0) tank.speed = Math.max(0, tank.speed - COAST_DECEL * dt);
    else tank.speed = Math.min(0, tank.speed + COAST_DECEL * dt);
  }

  // 3. move along the hull axis (tanks cannot strafe — one signed scalar).
  tank.x += Math.cos(tank.heading) * tank.speed * dt;
  tank.y += Math.sin(tank.heading) * tank.speed * dt;

  // 4. resolve: push out of solid cover, then other live tanks, then clamp to
  // the arena. Contact semantics: wallContact means THIS tick's motion had to
  // be corrected (a push-out or an active clamp) — grinding into a wall flags
  // every tick, which is what the per-tick REWARD_WALL_CONTACT prices, while a
  // tank parked exactly on the boundary with zero speed does not re-flag
  // (pushOutOfBox reports contact only when it actually moved the tank, and
  // the border clamps below test strictly, mirroring each other). The
  // engine's step ORs this flag across its three resolve passes.
  let contact = false;
  for (const w of walls) contact = pushOutOfBox(tank, w) || contact;
  for (let i = 0; i < crates.length; i++) {
    // hp ≤ 0 crates are dead already (removed at the end of the bullet pass);
    // a corpse must not block a tank for the ticks it still exists.
    if (crates[i].hp <= 0) continue;
    contact = pushOutOfBox(tank, crates[i]) || contact;
  }
  contact = separateTanks(tank, tanks) || contact;
  // Border clamp AFTER obstacle push-out so a box push can never eject a tank
  // through the arena edge (the clamp is the last word on containment).
  if (tank.x < TANK_RADIUS) {
    tank.x = TANK_RADIUS;
    contact = true;
  } else if (tank.x > ARENA_W - TANK_RADIUS) {
    tank.x = ARENA_W - TANK_RADIUS;
    contact = true;
  }
  if (tank.y < TANK_RADIUS) {
    tank.y = TANK_RADIUS;
    contact = true;
  } else if (tank.y > ARENA_H - TANK_RADIUS) {
    tank.y = ARENA_H - TANK_RADIUS;
    contact = true;
  }
  tank.wallContact = contact;
}

/**
 * Push a tank circle out of one AABB. Diagonal contacts resolve along the
 * centre→closest-point normal (the honest corner normal); a centre that ended
 * up INSIDE the box escapes along its shallowest axis. Returns true ONLY when
 * an actual correction was applied: an exact touch (d == R, push == 0) is
 * where a push-out LANDS a tank (dx/d is exactly ±1 on a face contact, so the
 * result is float-exact), and reporting it as contact would re-flag a tank
 * parked at rest every tick — an endless REWARD_WALL_CONTACT for doing
 * nothing, and a scrape signal that never clears. The tolerance (not a bare
 * push > 0) absorbs ulp-penetration landings (d an ulp under R ⇒ push ~1e-14),
 * which are measurement noise on the same parked state, not grinding.
 */
const PUSH_EPS = 1e-9;

function pushOutOfBox(tank: Tank, o: Obstacle): boolean {
  const cx = Math.min(Math.max(tank.x, o.x), o.x + o.w);
  const cy = Math.min(Math.max(tank.y, o.y), o.y + o.h);
  const dx = tank.x - cx;
  const dy = tank.y - cy;
  const d2 = dx * dx + dy * dy;
  if (d2 > TANK_RADIUS * TANK_RADIUS) return false;
  if (d2 > 1e-12) {
    // normal case: push straight out along the contact normal
    const d = Math.sqrt(d2);
    const push = TANK_RADIUS - d;
    if (push <= PUSH_EPS) return false; // exact touch — nothing to correct
    tank.x += (dx / d) * push;
    tank.y += (dy / d) * push;
  } else {
    // centre inside the box (spawned overlapping, or a fast tick): escape along
    // whichever face is nearest — the shallowest penetration is the least
    // surprising exit and needs no normal direction at all.
    const left = tank.x - o.x;
    const right = o.x + o.w - tank.x;
    const top = tank.y - o.y;
    const bottom = o.y + o.h - tank.y;
    const m = Math.min(left, right, top, bottom);
    if (m === left) tank.x = o.x - TANK_RADIUS;
    else if (m === right) tank.x = o.x + o.w + TANK_RADIUS;
    else if (m === top) tank.y = o.y - TANK_RADIUS;
    else tank.y = o.y + o.h + TANK_RADIUS;
  }
  return true;
}

/**
 * Keep `tank` out of every other ALIVE tank. Pushes THIS tank by the full
 * overlap (not half): integrateTank runs per tank in order, and a full push
 * fully separates a pair the first time either of them resolves — a half-push
 * would only halve the overlap per resolution and leave a geometric tail of
 * sub-unit intersections for several ticks. Corpses are passable: a wreck must
 * not wall off a corridor for the rest of the episode.
 */
function separateTanks(tank: Tank, tanks: Tank[]): boolean {
  let contact = false;
  for (const other of tanks) {
    if (other === tank || !other.alive) continue;
    const dx = tank.x - other.x;
    const dy = tank.y - other.y;
    const min = TANK_RADIUS * 2;
    const d2 = dx * dx + dy * dy;
    if (d2 >= min * min) continue;
    contact = true;
    const d = Math.sqrt(d2);
    if (d > 1e-9) {
      const push = min - d;
      tank.x += (dx / d) * push;
      tank.y += (dy / d) * push;
    } else {
      // perfectly stacked (a pathological spawn): deterministic escape along
      // +x; the border clamp below re-contains it if that leaves the arena.
      tank.x += min;
    }
  }
  return contact;
}

// --- fire control ------------------------------------------------------------

/**
 * Attempt to fire `tank`'s gun. Returns false (and mutates nothing) when the
 * gun is reloading, the tank is dead, or a magazine/global cap is hit; on
 * success spawns the bullet at the muzzle and starts the cooldown. The CALLER
 * (engine) pushes the "fire" event — physics owns state, not juice.
 *
 * The bullet copies owner id AND team at fire time: a tank can die while its
 * shell is in flight and the shell must keep its side for allied-damage rules,
 * plus bounces/pierce snapshots at fire time so a powerup expiring mid-flight
 * does not retroactively change a bullet's character.
 */
export function tryFire(game: GameState, tank: Tank): boolean {
  if (!tank.alive || tank.reload > 0) return false;
  if (tank.bulletsInFlight >= MAX_BULLETS_PER_TANK) return false;
  if (game.bullets.length >= MAX_BULLETS_GLOBAL) return false;
  const bullet: Bullet = {
    id: game.nextBulletId++,
    ownerId: tank.id,
    team: tank.team,
    x: tank.x + Math.cos(tank.heading) * MUZZLE_OFFSET,
    y: tank.y + Math.sin(tank.heading) * MUZZLE_OFFSET,
    dir: tank.heading,
    bouncesLeft: tank.timers.bounce > 0 ? BOUNCE_POWERUP : BULLET_BOUNCES,
    pierce: tank.timers.pierce > 0,
    life: BULLET_LIFETIME,
    hit: [],
  };
  game.bullets.push(bullet);
  tank.bulletsInFlight++;
  tank.reload = tank.timers.rapid > 0 ? RAPID_COOLDOWN : FIRE_COOLDOWN;
  return true;
}

// --- bullet integration ------------------------------------------------------

/**
 * Advance every live bullet dt seconds in BULLET_SUBSTEPS sweeps, resolving
 * contact per sweep (border/wall/crate reflect-or-die, tank damage), then
 * remove dead and expired bullets in one filter pass.
 *
 * bullet.hit bookkeeping: tank ids appear POSITIVE, crates NEGATIVE as
 * −(crateId+1) (engine-assigned Obstacle.id, so crate 0 is −1, never −0; the
 * encoding is engine-internal — snapshot.ts never serializes hit[]). This one
 * array drives three rules at once — pierce hits each tank at most once, a
 * bullet damages a given crate at most once, and "expired without hitting
 * anything" (hit.length === 0) is what marks the owner's miss for
 * REWARD_BULLET_EXPIRED. Marks must be keyed by STABLE crate id, not array
 * index: broken crates are spliced out of game.crates (below) while bullets
 * live up to BULLET_LIFETIME, so an index mark would shift under a flying
 * shell — re-damaging the crate that moved into the stale slot and wrongly
 * shielding the one that moved out (breaking constants.ts's 3-hit crate
 * invariant). A marked crate stays SOLID cover: the mark suppresses only the
 * damage, never the reflect/push-out (see the crate pass).
 */
export function integrateBullets(game: GameState, dt: number): void {
  const subDt = dt / BULLET_SUBSTEPS;
  const events = game.events;
  // ids of bullets that died to contact this tick (expired-by-life is derived
  // in the removal pass; contact deaths must not also be charged as expiry)
  const killed = new Set<number>();
  // indices of crates broken this tick — spliced once after the sweeps
  const deadCrates: number[] = [];

  for (const b of game.bullets) b.life -= dt;

  for (let step = 0; step < BULLET_SUBSTEPS; step++) {
    for (const b of game.bullets) {
      if (killed.has(b.id)) continue;
      b.x += Math.cos(b.dir) * BULLET_SPEED * subDt;
      b.y += Math.sin(b.dir) * BULLET_SPEED * subDt;

      // arena border: reflect exactly like a wall face AND consume a bounce —
      // the plan is explicit that bounds are walls, not free mirrors, so a
      // plain shot hugging a border dies on its second border contact.
      if (b.x < BULLET_RADIUS) {
        b.x = BULLET_RADIUS;
        if (!bounceOrKill(b, "x", events, killed)) continue;
      } else if (b.x > ARENA_W - BULLET_RADIUS) {
        b.x = ARENA_W - BULLET_RADIUS;
        if (!bounceOrKill(b, "x", events, killed)) continue;
      }
      if (b.y < BULLET_RADIUS) {
        b.y = BULLET_RADIUS;
        if (!bounceOrKill(b, "y", events, killed)) continue;
      } else if (b.y > ARENA_H - BULLET_RADIUS) {
        b.y = ARENA_H - BULLET_RADIUS;
        if (!bounceOrKill(b, "y", events, killed)) continue;
      }

      // solid walls: nearest contact wins; one contact per sweep is enough
      // (walls never overlap by arena construction, and a bullet is smaller
      // than the 14 u sweep so two wall faces in one sweep cannot both matter)
      for (const w of game.walls) {
        const axis = bulletBoxContact(b, w);
        if (axis !== null) {
          if (!bounceOrKill(b, axis, events, killed)) break;
          break; // reflect at most once per sweep; re-contact waits for the next
        }
      }
      if (killed.has(b.id)) continue;

      // crates: destructible cover. Damage is applied even when the bullet
      // dies on this contact — the hit is real either way. The CONTACT test
      // runs before the damage gate: a crate this bullet already damaged is
      // still solid cover, so it must reflect/push out exactly like an intact
      // one — only the DAMAGE is once-per-crate (mark in hit[]), otherwise a
      // chipped crate would turn transparent to the shell that chipped it
      // (shells fly through intact cover and corner clips re-contact on the
      // very next sweep).
      for (let i = 0; i < game.crates.length; i++) {
        const crate = game.crates[i];
        if (crate.hp <= 0) continue; // broken earlier this same tick
        const mark = -(crate.id ?? i) - 1; // stable id; index fallback for id-less fixtures
        const axis = bulletBoxContact(b, crate);
        if (axis === null) continue;
        if (!b.hit.includes(mark)) {
          crate.hp -= BULLET_DAMAGE;
          b.hit.push(mark);
          if (crate.hp <= 0) {
            deadCrates.push(i);
            // break event now (correct order vs hit/ricochet); the splice
            // itself is deferred so the index list below stays valid
            events.push({ kind: "crate", x: crate.x + crate.w / 2, y: crate.y + crate.h / 2 });
          }
        }
        // reflect about the crate face actually touched (same rule as walls)
        if (!bounceOrKill(b, axis, events, killed)) break;
        break; // one crate contact per sweep, like walls
      }
      if (killed.has(b.id)) continue;

      // tanks: alive, not the owner, on a hittable side (FFA: everyone;
      // allied: not the shooter's team), not already damaged by THIS bullet,
      // and not shielded. Kill credit is deliberately NOT touched here — the
      // engine reads hp and its own kill events, so physics stays reward-free.
      for (const t of game.tanks) {
        if (!t.alive || t.id === b.ownerId) continue;
        if (game.teamMode === "allied" && t.team === b.team) continue;
        if (b.hit.includes(t.id)) continue;
        if (t.timers.shield > 0) continue;
        const dx = b.x - t.x;
        const dy = b.y - t.y;
        const rr = TANK_RADIUS + BULLET_RADIUS;
        if (dx * dx + dy * dy > rr * rr) continue;
        t.hp -= BULLET_DAMAGE;
        b.hit.push(t.id);
        events.push({ kind: "hit", x: t.x, y: t.y, tank: t.id });
        if (!b.pierce) {
          killed.add(b.id);
          events.push({ kind: "explode", x: b.x, y: b.y });
          break;
        }
        // pierce: keep flying, but this tank is now in hit[] — no re-damage
      }
    }
  }

  // Break the marked crates, highest index first so the pending indices stay
  // valid while splicing (removing index 5 does not shift index 3).
  for (let k = deadCrates.length - 1; k >= 0; k--) game.crates.splice(deadCrates[k], 1);

  // Single removal pass: contact-dead and life-expired bullets leave together,
  // owners' magazines are credited back here (by id lookup, not array index —
  // tanks keep their id but the array is ordered by spawn), and ONLY a bullet
  // that expired without ever damaging anything is reported to the engine as
  // its owner's miss (explode + tank = ownerId is the "expired" marker).
  const stillFlying: Bullet[] = [];
  for (const b of game.bullets) {
    const expired = b.life <= 0;
    if (!killed.has(b.id) && !expired) {
      stillFlying.push(b);
      continue;
    }
    const owner = game.tanks.find((t) => t.id === b.ownerId);
    if (owner) owner.bulletsInFlight--;
    if (expired && !killed.has(b.id) && b.hit.length === 0) {
      events.push({ kind: "explode", x: b.x, y: b.y, tank: b.ownerId });
    }
  }
  game.bullets = stillFlying;
}

/**
 * One border/wall/crate contact for a bullet: reflect the normal-axis velocity
 * while bounces remain (pushing a ricochet ping), else mark the bullet dead
 * with a small explosion. Axis is the FACE axis ("x" = vertical face, reflect
 * the x velocity component): as an angle, mirroring x is dir → π − dir and
 * mirroring y is dir → −dir — exact, no atan2 rounding in the hot path.
 * Returns false when the bullet died (caller must stop processing it).
 */
function bounceOrKill(
  b: Bullet,
  axis: "x" | "y",
  events: GameEvent[],
  killed: Set<number>,
): boolean {
  if (b.bouncesLeft > 0) {
    b.bouncesLeft--;
    b.dir = axis === "x" ? Math.PI - b.dir : -b.dir;
    events.push({ kind: "ricochet", x: b.x, y: b.y, angle: b.dir });
    return true;
  }
  killed.add(b.id);
  events.push({ kind: "explode", x: b.x, y: b.y });
  return false;
}

/**
 * Circle-vs-AABB contact for a BULLET: returns the face axis to reflect about,
 * or null on no contact. Also pushes the bullet out so the reflection starts
 * from the surface rather than re-contacting next sweep. Mirror of
 * pushOutOfBox's geometry (closest point, inside-escape by shallowest face),
 * but corners answer with the DEEPER axis — for a corner clip either choice is
 * defensible; the deeper axis is the one the bullet is actually entering.
 */
function bulletBoxContact(b: Bullet, o: Obstacle): "x" | "y" | null {
  const cx = Math.min(Math.max(b.x, o.x), o.x + o.w);
  const cy = Math.min(Math.max(b.y, o.y), o.y + o.h);
  const dx = b.x - cx;
  const dy = b.y - cy;
  const d2 = dx * dx + dy * dy;
  if (d2 > BULLET_RADIUS * BULLET_RADIUS) return null;
  if (d2 > 1e-12) {
    const d = Math.sqrt(d2);
    const push = BULLET_RADIUS - d;
    b.x += (dx / d) * push;
    b.y += (dy / d) * push;
    // face hits have exactly one non-zero component; corner clips take the
    // dominant one. ≥ (not >) makes a perfect 45° corner hit reflect about x,
    // a pinned deterministic choice rather than an accident of comparison.
    return Math.abs(dx) >= Math.abs(dy) ? "x" : "y";
  }
  // centre inside the box: escape and reflect about the shallowest face
  const left = b.x - o.x;
  const right = o.x + o.w - b.x;
  const top = b.y - o.y;
  const bottom = o.y + o.h - b.y;
  const m = Math.min(left, right, top, bottom);
  if (m === left) {
    b.x = o.x - BULLET_RADIUS;
    return "x";
  }
  if (m === right) {
    b.x = o.x + o.w + BULLET_RADIUS;
    return "x";
  }
  if (m === top) {
    b.y = o.y - BULLET_RADIUS;
    return "y";
  }
  b.y = o.y + o.h + BULLET_RADIUS;
  return "y";
}

// --- raycasting (perception) -------------------------------------------------

/**
 * Nearest wall/crate/tank along a ray from (x, y) in direction `dir`, within
 * maxLen. Exact analytic intersection (slabs for boxes, quadratic for circles)
 * rather than marching: identical answers at every angle (a marcher can slip
 * past a thin face between samples), and dist is exact to float rounding —
 * the encoder feeds these distances straight into the RAYS observation block.
 * Corpses (hp ≤ 0 crates, dead tanks) are transparent.
 *
 * Ties keep the earlier class in walls → crates → tanks order (strictly-nearer
 * replaces), so the result is a total function of the state, not of iteration
 * luck.
 */
export function raycast(
  game: GameState,
  x: number,
  y: number,
  dir: number,
  maxLen: number,
): RayHit {
  const dx = Math.cos(dir);
  const dy = Math.sin(dir);
  let best: RayHit = { dist: maxLen, kind: null, tankId: -1 };
  for (const w of game.walls) {
    const t = rayAABB(x, y, dx, dy, w, maxLen);
    if (t !== null && t < best.dist) best = { dist: t, kind: "wall", tankId: -1 };
  }
  for (let i = 0; i < game.crates.length; i++) {
    const c = game.crates[i];
    if (c.hp <= 0) continue;
    const t = rayAABB(x, y, dx, dy, c, maxLen);
    if (t !== null && t < best.dist) best = { dist: t, kind: "crate", tankId: -1 };
  }
  for (const t of game.tanks) {
    if (!t.alive) continue;
    const hit = rayCircle(x, y, dx, dy, t.x, t.y, TANK_RADIUS, maxLen);
    if (hit !== null && hit < best.dist) best = { dist: hit, kind: "tank", tankId: t.id };
  }
  return best;
}

/** Exact ray→AABB entry distance via slabs, or null. (dx, dy) must be unit.
 *  A ray starting inside returns 0 — an eye inside cover sees that cover. */
function rayAABB(
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  o: Obstacle,
  maxLen: number,
): number | null {
  let tmin = 0;
  let tmax = maxLen;
  if (Math.abs(dx) < 1e-12) {
    // ray parallel to the y-axis: inside the x-slab or it can never enter
    if (ox < o.x || ox > o.x + o.w) return null;
  } else {
    let t1 = (o.x - ox) / dx;
    let t2 = (o.x + o.w - ox) / dx;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tmin = Math.max(tmin, t1);
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return null;
  }
  if (Math.abs(dy) < 1e-12) {
    if (oy < o.y || oy > o.y + o.h) return null;
  } else {
    let t1 = (o.y - oy) / dy;
    let t2 = (o.y + o.h - oy) / dy;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tmin = Math.max(tmin, t1);
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return null;
  }
  return tmin;
}

/** Exact ray→circle entry distance (unit direction), or null. Behind the origin
 *  is a miss; a start already inside reports contact at 0. */
function rayCircle(
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  cx: number,
  cy: number,
  r: number,
  maxLen: number,
): number | null {
  const mx = ox - cx;
  const my = oy - cy;
  const b = mx * dx + my * dy;
  const c = mx * mx + my * my - r * r;
  const disc = b * b - c; // a = |d|² = 1 for a unit direction
  if (disc < 0) return null;
  const root = Math.sqrt(disc);
  const t = -b - root; // near root: the entry point
  if (t >= 0) return t <= maxLen ? t : null;
  // near root behind the origin: either we started inside (contact at 0) or
  // the whole circle is behind us (miss)
  return c <= 0 ? 0 : null;
}
