// rewards.ts — the per-tank reward vector: event rewards, damage/kill
// attribution, the timeout placement bonus, and the potential-based shaping
// terms. DOM-free, tfjs-free (runs in the train worker for 16 lanes at 30 Hz
// AND in node vitest; no Math.random — rewards are a pure function of two
// state views, which is what makes the property tests exact rather than
// statistical).
//
// WHY a lightweight RewardView instead of the whole GameState: scoring a tick
// needs the PRE-tick values of a handful of fields (hp, positions, alive
// flags, bullet positions) plus the post-tick world. Deep-cloning the world
// 30×/s × 16 lanes would dominate the worker budget, so the engine captures
// this structural slice instead (captureRewardView). A full GameState is
// assignable to RewardView — the engine passes `game` itself as the "current"
// state — while the pre-tick copy allocates ~30 small rows, not a world.
//
// THE ATTRIBUTION HEURISTIC (documented simplification): physics deliberately
// stays reward-free (see integrateBullets), so a bullet's owner is not part of
// the state diff a reward pass can read — the "hit" event carries only the
// victim. Each victim's HP loss is therefore credited to the NEAREST living
// tank on another team. In practice the shooter almost always is the closest
// tank to its victim (shells fly straight at 840 u/s and live 2.5 s, so a hit
// is a short-range event); the error is rare and, critically for learning,
// UNBIASED — in an FFA every one of the N−1 others is an interchangeable
// creditor, so no agent is systematically over- or under-credited. The same
// nearest-actor rule prices kills (nearest enemy) and crate breaks (nearest
// tank to the rubble, i.e. whoever was fighting over that cover).
//
// SUDDEN-DEATH DRAIN: the arena's HP drain is nobody's shot. It is EXCLUDED
// from damage-dealt credit (the arena must not hand out kill/damage rewards)
// but INCLUDED in damage-taken (idling at low HP during sudden death must
// hurt — that is the drain's whole point, see constants.ts).

import {
  DT,
  NORM_DIST,
  PHI_DANGER_RADIUS,
  PHI_DANGER_WEIGHT,
  PSI_APPROACH_WEIGHT,
  REWARD_ALLIED_FIRE,
  REWARD_BULLET_EXPIRED,
  REWARD_CRATE_BREAK,
  REWARD_DAMAGE_DEALT,
  REWARD_DAMAGE_TAKEN,
  REWARD_DEATH,
  REWARD_KILL,
  REWARD_PLACEMENT_MAX,
  REWARD_SURVIVAL,
  REWARD_SURVIVAL_ESC_MULT,
  REWARD_SURVIVAL_SD_MULT,
  REWARD_WALL_CONTACT,
  SUDDEN_DEATH_DRAIN,
} from "./constants";
import type { GameEvent, GameState, Phase, TeamMode } from "./types";

/**
 * Discount factor used ONLY by the shaping terms. It must equal the learner's
 * γ (train-protocol.ts defaultConfig().gamma = 0.97): potential shaping is
 * policy-neutral for the γ it is built with — a mismatch means the return the
 * value function discounts and the reward the env adds disagree about Φ, and
 * the Ng et al. (1999) guarantee quietly stops holding. This constant is the
 * DEFAULT and the drift-catcher (rewards.spec pins it against
 * defaultConfig().gamma); the engine threads the episode's ACTUAL γ through
 * GameState.gamma (GameConfig.gamma ← NetConfig.gamma) so a user-configured
 * learner reshapes its env to match rather than silently diverging.
 */
export const REWARD_GAMMA = 0.97;

/** The per-tank fields the reward pass reads, copied pre-tick by the engine. */
export interface RewardTank {
  id: number;
  x: number;
  y: number;
  hp: number;
  alive: boolean;
  team: number;
  /** One-tick death flag — read from the CURRENT state only (the pre-tick
   *  copy carries it so the same row type serves both views). */
  diedThisTick: boolean;
  /** Set by integrateTank when this tick's motion had to be corrected. */
  wallContact: boolean;
}

/** The per-bullet fields the danger potential reads. */
export interface RewardBullet {
  x: number;
  y: number;
  ownerId: number;
  /** Owner's team at fire time (copied onto the bullet) — hostility test. */
  team: number;
}

/**
 * Structural slice of GameState every reward function reads. Tanks/bullets are
 * copied ROWS in the engine's pre-tick capture (values frozen at capture time)
 * but a live GameState satisfies the interface directly, so `cur` is never
 * copied — rewards must see the post-tick world, not a stale one.
 */
export interface RewardView {
  tanks: RewardTank[];
  bullets: RewardBullet[];
  /** This tick's events (crate breaks, expired bullets). Index-aligned with
   *  nothing — events carry ids/positions, never indices. */
  events: GameEvent[];
  teamMode: TeamMode;
  /** Episode phase AFTER the tick's escalation step (survival multiplier and
   *  the drain share both key off the phase in effect at reward time). */
  phase: Phase;
}

/**
 * Freeze the fields the reward pass reads from `game` into plain rows. Called
 * by the engine exactly once per tick, BEFORE anything moves; the rows are
 * dead copies, so later mutation of the world cannot retroactively change
 * Φ(prev) or a hp delta. Tanks are copied by ARRAY POSITION — the engine never
 * reorders or removes tanks (corpses stay in place), so prev[i] and cur[i] are
 * the same tank for the whole episode.
 */
export function captureRewardView(game: GameState): RewardView {
  return {
    tanks: game.tanks.map((t) => ({
      id: t.id,
      x: t.x,
      y: t.y,
      hp: t.hp,
      alive: t.alive,
      team: t.team,
      diedThisTick: t.diedThisTick,
      wallContact: t.wallContact,
    })),
    // Bullet rows are copied too (positions move every sweep); ≤24 of them.
    bullets: game.bullets.map((b) => ({ x: b.x, y: b.y, ownerId: b.ownerId, team: b.team })),
    events: game.events,
    teamMode: game.teamMode,
    phase: game.phase,
  };
}

// --- potential functions ------------------------------------------------------
//
// Φ/Ψ enter the return ONLY as γΦ(s′)−Φ(s) / γΨ(s′)−Ψ(s) (shapedStep), so they
// telescope to a state-history-only term and can never change the optimal
// policy (Ng et al. 1999) — which is why these are the only reward inputs
// allowed to look at raw enemy bullet geometry. Both are 0 for a dead tank:
// a corpse has no danger to dodge and no fight to close, and the zero is what
// makes the death tick absorb the full −Φ drop exactly once.

/**
 * Danger potential Φ for tank `tankId`:
 *   Φ = PHI_DANGER_WEIGHT · max(0, 1 − dNearestHostileBullet / PHI_DANGER_RADIUS)
 * ∈ [PHI_DANGER_WEIGHT, 0]. Hostile = fired by another TEAM (own and ally
 * shells are geometry, not threats — dodging your own ricochet is not what the
 * term should teach). No hostile bullet in range ⇒ 0.
 */
export function potentialDanger(view: RewardView, tankId: number): number {
  const me = view.tanks.find((t) => t.id === tankId);
  if (!me || !me.alive) return 0;
  let best = Infinity;
  for (const b of view.bullets) {
    if (b.team === me.team) continue;
    const dx = b.x - me.x;
    const dy = b.y - me.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < best) best = d2;
  }
  if (best === Infinity) return 0;
  return PHI_DANGER_WEIGHT * Math.max(0, 1 - Math.sqrt(best) / PHI_DANGER_RADIUS);
}

/**
 * Approach potential Ψ for tank `tankId`:
 *   Ψ = PSI_APPROACH_WEIGHT · (1 − dNearestAliveEnemy / NORM_DIST) ∈ [0, weight].
 * The FFA replacement for the original's 100/distance term: linear in distance
 * over the whole arena (the diagonal normalizer keeps it in [0, 1]) so it has
 * no hugging singularity at contact. No living enemy ⇒ 0 (episode decided).
 */
export function potentialApproach(view: RewardView, tankId: number): number {
  const me = view.tanks.find((t) => t.id === tankId);
  if (!me || !me.alive) return 0;
  let best = Infinity;
  for (const t of view.tanks) {
    if (t.id === me.id || !t.alive || t.team === me.team) continue;
    const dx = t.x - me.x;
    const dy = t.y - me.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < best) best = d2;
  }
  if (best === Infinity) return 0;
  return PSI_APPROACH_WEIGHT * (1 - Math.sqrt(best) / NORM_DIST);
}

// --- event rewards ------------------------------------------------------------

/** Survival multiplier for a phase: 1 / ×2 / ×4 (constants.ts). */
function survivalMult(phase: Phase): number {
  return phase === 2 ? REWARD_SURVIVAL_SD_MULT : phase === 1 ? REWARD_SURVIVAL_ESC_MULT : 1;
}

/** Nearest living tank to (x, y) subject to `accept`, or −1. Deliberately
 *  scans the CURRENT rows: attribution asks "who is on top of this victim
 *  NOW", which is where a shooter ends a hit. */
function nearest(
  view: RewardView,
  x: number,
  y: number,
  accept: (t: RewardTank) => boolean,
): number {
  let best = -1;
  let bestD2 = Infinity;
  for (const t of view.tanks) {
    if (!t.alive || !accept(t)) continue;
    const dx = t.x - x;
    const dy = t.y - y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = t.id;
    }
  }
  return best;
}

/** Index of tank `id` in the (never reordered) tank array, or −1. */
function idxOf(view: RewardView, id: number): number {
  return view.tanks.findIndex((t) => t.id === id);
}

/**
 * Per-tank event rewards for one tick, `prev` → `cur`. Output is index-aligned
 * with the tank array (== lastRewards). Total over any pair of views — no
 * NaN, no gaps — so a caller can never lose a tank's reward to a degenerate
 * state.
 *
 * Terms (per tank, all from constants.ts):
 *   survival      +REWARD_SURVIVAL × phase multiplier, alive at tick start
 *   damage dealt  +REWARD_DAMAGE_DEALT per HP attributed via nearest enemy
 *   damage taken  −REWARD_DAMAGE_TAKEN per HP lost (drain included)
 *   kill          +REWARD_KILL to the nearest enemy of a tank that died
 *   death         −REWARD_DEATH exactly once (diedThisTick is a one-tick flag)
 *   crate break   +REWARD_CRATE_BREAK to the nearest tank per "crate" event
 *   wall contact  −REWARD_WALL_CONTACT while scraping
 *   expired shell −REWARD_BULLET_EXPIRED per "explode" event carrying the
 *                 owner id (physics emits that shape ONLY for a shell that
 *                 expired having hit nothing)
 *   allied fire   −REWARD_ALLIED_FIRE per HP to the nearest same-team tank
 *                 (allied mode only; FFA teams are unique so the term is
 *                 unreachable there by construction)
 */
export function stepRewards(prev: RewardView, cur: RewardView): number[] {
  const n = Math.min(prev.tanks.length, cur.tanks.length);
  const out = new Array<number>(n).fill(0);

  // Survival pays for having been alive at the START of the tick — the tank
  // that dies mid-tick still banked the approach it survived through.
  for (let i = 0; i < n; i++) {
    if (prev.tanks[i].alive) out[i] += REWARD_SURVIVAL * survivalMult(cur.phase);
  }

  // Wall scrape: only live tanks (a corpse keeps its last contact flag).
  for (let i = 0; i < n; i++) {
    if (cur.tanks[i].alive && cur.tanks[i].wallContact) out[i] += REWARD_WALL_CONTACT;
  }

  // Expired misses: physics marks them "explode" + owner id. Blast/death
  // explosions (engine-emitted) carry no tank field and never land here.
  for (const e of cur.events) {
    if (e.kind !== "explode" || e.tank === undefined) continue;
    const i = idxOf(cur, e.tank);
    if (i >= 0) out[i] += REWARD_BULLET_EXPIRED;
  }

  // Crate breaks: whoever was nearest the rubble was fighting over it.
  for (const e of cur.events) {
    if (e.kind !== "crate") continue;
    const id = nearest(cur, e.x, e.y, () => true);
    const i = id >= 0 ? idxOf(cur, id) : -1;
    if (i >= 0) out[i] += REWARD_CRATE_BREAK;
  }

  // HP deltas: damage taken by the loser, damage dealt attributed to the
  // nearest living tank on another team (see header — documented heuristic).
  const drain = cur.phase === 2 ? SUDDEN_DEATH_DRAIN * DT : 0;
  for (let j = 0; j < n; j++) {
    const pv = prev.tanks[j];
    const cv = cur.tanks[j];
    const gross = Math.max(0, pv.hp - cv.hp);
    if (gross <= 0) continue;
    // The loser always pays for every HP it lost, drain included.
    out[j] += REWARD_DAMAGE_TAKEN * gross;
    // Dealt-credit share: subtract the arena's drain for survivors (a tank
    // that died this tick lost its HP to a shot, not to the drain, because
    // the drain only reaches tanks that are still alive in `cur`).
    const net = cv.alive ? Math.max(0, gross - drain) : gross;
    if (net <= 0) continue;
    const shooter = nearest(cur, cv.x, cv.y, (t) => t.id !== cv.id && t.team !== cv.team);
    if (shooter >= 0) out[idxOf(cur, shooter)] += REWARD_DAMAGE_DEALT * net;
    // Friendly fire: only blasts can damage a teammate (bullets are team-
    // blocked in physics), and the nearest teammate is charged the same way
    // the nearest enemy is credited — one heuristic, applied to both sides.
    if (cur.teamMode === "allied") {
      const mate = nearest(cur, cv.x, cv.y, (t) => t.id !== cv.id && t.team === cv.team);
      if (mate >= 0) out[idxOf(cur, mate)] += REWARD_ALLIED_FIRE * net;
    }
  }

  // Deaths: the victim pays once (the flag is cleared at the top of every
  // step), the nearest enemy banks the kill. In allied mode a blast-kill of a
  // teammate credits the nearest enemy — noisy, rare, and self-averaging.
  for (let j = 0; j < n; j++) {
    if (!cur.tanks[j].diedThisTick) continue;
    out[j] += REWARD_DEATH;
    const cv = cur.tanks[j];
    const killer = nearest(cur, cv.x, cv.y, (t) => t.id !== cv.id && t.team !== cv.team);
    if (killer >= 0) out[idxOf(cur, killer)] += REWARD_KILL;
  }

  return out;
}

/**
 * Timeout placement bonus per tank (index-aligned): rank r of N scores
 * +REWARD_PLACEMENT_MAX·(N−1−r)/(N−1) — full bonus for the leader, 0 for
 * last. Alive tanks rank first by hp (desc, ties by id order — the array is
 * never reordered, so id order IS the stable fallback); the dead share the
 * remaining ranks in id order (individual death ticks are not tracked in
 * GameState, and among the dead the residual differences are worth less than
 * the field it would cost to track them). N < 2 degenerates to a full bonus
 * for the lone tank (a solo survivor is rank 0 of 1 by definition).
 */
export function placementBonus(game: GameState): number[] {
  const n = game.tanks.length;
  if (n < 2) return game.tanks.map(() => REWARD_PLACEMENT_MAX);
  const order = game.tanks.map((_, i) => i).sort((a, b) => {
    const A = game.tanks[a];
    const B = game.tanks[b];
    if (A.alive !== B.alive) return A.alive ? -1 : 1; // alive ranks first
    return B.hp - A.hp; // then by hp desc; stable sort keeps id order on ties
  });
  const out = new Array<number>(n).fill(0);
  for (let rank = 0; rank < n; rank++) {
    out[order[rank]] = (REWARD_PLACEMENT_MAX * (n - 1 - rank)) / (n - 1);
  }
  return out;
}

/**
 * Shaped per-tank reward for one tick: stepRewards plus the potential
 * differences γΦ(cur)−Φ(prev) and γΨ(cur)−Ψ(prev). Summed over any episode
 * the shaping terms reduce to a boundary-only term (they cannot reward a
 * loop, only reshape the path), which is why they — and nothing else — may
 * read enemy bullet positions. A tank that DIES this tick contributes
 * −Φ(prev) once (γ·0 − Φ(prev)); the corpse's potential stays 0 after.
 *
 * `gamma` is the LEARNER's discount (engine.step passes GameState.gamma,
 * which createGame copies from GameConfig.gamma ← NetConfig.gamma): the
 * policy-neutrality proof is γ-specific, so the shaping MUST be rebuilt with
 * whatever γ the value function bootstraps with. The REWARD_GAMMA default
 * keeps the pure-reward fixtures (and the drift pin in rewards.spec) on the
 * plan's 0.97.
 */
export function shapedStep(prev: RewardView, cur: RewardView, gamma: number = REWARD_GAMMA): number[] {
  const out = stepRewards(prev, cur);
  const n = Math.min(out.length, cur.tanks.length);
  for (let i = 0; i < n; i++) {
    const id = cur.tanks[i].id;
    // γ discounts ONLY the successor's potential: γΦ(s′) − Φ(s). Discounting
    // the difference instead would make the shaping itself shrink with γ and
    // the return no longer telescope to a history-only term — the whole
    // policy-neutrality argument (Ng et al. 1999) rests on this exact form.
    out[i] +=
      gamma * potentialDanger(cur, id) -
      potentialDanger(prev, id) +
      gamma * potentialApproach(cur, id) -
      potentialApproach(prev, id);
  }
  return out;
}
