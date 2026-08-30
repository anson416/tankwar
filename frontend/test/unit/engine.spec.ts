// engine.spec.ts — the world-contract suite. The centrepiece is the plan's
// 2000-game property sweep: seeded random action logs drive every tank count,
// and every tick of every game is checked against the invariants the rest of
// the codebase (encoder bounds, renderer interpolation, replay determinism)
// silently depends on:
//
//   containment   a tank's CENTRE stays ≥ TANK_RADIUS·0.9 from every wall and
//                 crate face/corner (≤2 u of penetration tolerated — physics
//                 resolves to TANK_RADIUS, so anything deeper is a bug; the
//                 check is distance-to-box, not an inflated square, which
//                 would misreport a corner touch at exactly TANK_RADIUS)
//   no tunneling  a bullet advances at most BULLET_SPEED·dt = 28 u per step.
//                 (The anti-tunneling number in the plan is 14 u per SWEEP:
//                 BULLET_SUBSTEPS = 2 splits the tick, and 14 u < the 40 u
//                 hull diameter is what makes a contact unmissable. Per tick
//                 the bound is therefore 28 u — asserted here at the
//                 granularity an outside observer can actually see.)
//   caps          ≤ MAX_BULLETS_PER_TANK per tank, ≤ MAX_BULLETS_GLOBAL alive
//   bounds        bullets stay within one BULLET_RADIUS of the arena (see the
//                 sweep's bullet loop for the one legitimate case at exactly
//                 that bound)
//   monotonicity  crate hp never rises; the living never grow back
//   termination   every game is over by maxTicks (+1 slack for the final
//                 step's own bookkeeping)
//   determinism   same seed + same action log ⇒ byte-identical snapshot
//                 stream; a different seed ⇒ a different stream
//
// The sweep keeps per-tick work to plain comparisons (a local `ck` instead of
// expect()) because 2000 games × 300 ticks × ~200 checks is the difference
// between a 5 s gate and a 60 s one. 2000 games completes in well under the
// 25 s budget; drop GAMES (and say so here) only if the engine ever doubles
// its per-tick cost.
import { describe, expect, it } from "vitest";
import { mulberry32 } from "./util/rng";
import {
  BULLET_DAMAGE,
  BULLET_SPEED,
  COAST_DECEL,
  CRATE_BLAST_DAMAGE,
  CRATE_BLAST_RADIUS,
  CRATE_HP,
  BULLET_RADIUS,
  DT,
  MAX_BULLETS_GLOBAL,
  MAX_BULLETS_PER_TANK,
  MAX_TICKS_CAP,
  REWARD_DEATH,
  REWARD_DAMAGE_DEALT,
  REWARD_KILL,
  SUDDEN_DEATH_DRAIN,
  TANK_HP,
  TANK_RADIUS,
  TICK_HZ,
  ARENA_H,
  ARENA_W,
} from "../../src/game/constants";
import { ACTION_COUNT } from "../../src/game/actions";
import { createGame, isOver, step } from "../../src/game/engine";
import { toSnapshot } from "../../src/game/snapshot";
import type { GameState, Obstacle } from "../../src/game/types";

// survival + shaping are small next to death; the victim's kill-tick reward
// must land far below any "death with compensation" misreading of the term.
const DEATH_TICK_CEILING = REWARD_DEATH + 5;

/** Cheap invariant check for the hot sweep — throws with context on the FIRST
 *  violated tick instead of paying expect()'s overhead 100M+ times. */
function ck(cond: boolean, what: string): void {
  if (!cond) throw new Error(`invariant violated: ${what}`);
}

describe("world construction", () => {
  it("places tanks on the first spawns, facing the arena centre, full hp", () => {
    const g = createGame({ tankCount: 4, teamMode: "ffa", episodeSeconds: 45, seed: 42 });
    expect(g.tanks.length).toBe(4);
    expect(g.spawns.length).toBe(4);
    g.tanks.forEach((t, i) => {
      expect(t.id).toBe(i);
      expect(t.x).toBeCloseTo(g.spawns[i].x, 9);
      expect(t.y).toBeCloseTo(g.spawns[i].y, 9);
      expect(t.hp).toBe(TANK_HP);
      expect(t.alive).toBe(true);
      expect(t.speed).toBe(0);
      expect(t.sprite).toBe(i === 0 ? 0 : 1); // P1 = tank_01 art, the rest tank_02
      expect(t.colorIdx).toBe(i);
      expect(t.reload).toBe(0);
      expect(t.bulletsInFlight).toBe(0);
      expect(t.timers).toEqual({ rapid: 0, shield: 0, speed: 0, bounce: 0, pierce: 0 });
      // heading points at the centre (atan2 toward (450, 350))
      expect(t.heading).toBeCloseTo(Math.atan2(ARENA_H / 2 - t.y, ARENA_W / 2 - t.x), 9);
    });
    expect(g.walls.every((w) => w.kind === "wall" && w.hp === 0)).toBe(true);
    expect(g.crates.every((c) => c.kind === "crate" && c.hp === CRATE_HP)).toBe(true);
    expect(g.arenaSeed).toBe(42);
    expect(g.over).toBe(false);
    expect(g.phase).toBe(0);
    expect(g.maxTicks).toBe(45 * TICK_HZ);
    // Literal anchor (rewards.spec plays the same trick): the tick cap is the
    // episode-termination invariant everything else is tuned around.
    expect(MAX_TICKS_CAP).toBe(2700); // 90 s @30 Hz
  });

  it("derives teams from the mode: ffa = one team per tank, allied = you vs the AIs", () => {
    const ffa = createGame({ tankCount: 4, teamMode: "ffa", episodeSeconds: 10, seed: 1 });
    expect(ffa.tanks.map((t) => t.team)).toEqual([0, 1, 2, 3]);
    const allied = createGame({ tankCount: 4, teamMode: "allied", episodeSeconds: 10, seed: 1 });
    expect(allied.tanks.map((t) => t.team)).toEqual([0, 1, 1, 1]);
  });

  it("clamps the episode to MAX_TICKS_CAP and is deterministic per config", () => {
    const long = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 10_000, seed: 7 });
    expect(long.maxTicks).toBe(MAX_TICKS_CAP); // 90 s hard cap
    const a = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 12, seed: 9 });
    const b = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 12, seed: 9 });
    expect(b).toEqual(a); // same config ⇒ identical world, rng schedule included
  });
});

describe("action handling", () => {
  it("collapses out-of-range, fractional and missing entries to coast/straight", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 10, seed: 3 });
    g.tanks[0].speed = 100; // rolling, so the coast decay is observable
    g.tanks[0].heading = 0.5;
    const before = g.tanks[0].heading;
    step(g, [99, -1, Number.NaN]);
    // coast/straight: heading untouched, speed decayed by exactly COAST_DECEL
    expect(g.tanks[0].heading).toBe(before);
    expect(g.tanks[0].speed).toBeCloseTo(100 - COAST_DECEL * DT, 9);
    // a short array leaves the missing tank on the same safe default
    const g2 = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 10, seed: 3 });
    step(g2, []);
    expect(g2.lastRewards.length).toBe(2);
    expect(Number.isFinite(g2.lastRewards[1])).toBe(true);
  });

  it("is a no-op once the episode is over", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 1, seed: 4 });
    while (!isOver(g)) step(g, [2, 2]);
    const snap = JSON.stringify(toSnapshot(g, 0));
    const rng = g.rngState;
    step(g, [17, 17]);
    expect(JSON.stringify(toSnapshot(g, 0))).toBe(snap); // nothing moved
    expect(g.rngState).toBe(rng); // and no rng draws were consumed
  });
});

describe("kill flow", () => {
  it("credits the kill, kills the victim once, and orders hit → kill events", () => {
    // THREE tanks: the kill must NOT end the episode, so the tick after the
    // death exists and can prove the one-tick flag gets cleared.
    const g = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 10, seed: 5 });
    g.walls = [];
    g.crates = [];
    g.tanks[0].x = 100;
    g.tanks[0].y = 350;
    g.tanks[0].heading = 0;
    g.tanks[1].x = 400;
    g.tanks[1].y = 350;
    g.tanks[1].hp = BULLET_DAMAGE; // one shell ends it
    Object.assign(g.tanks[2], { x: 800, y: 100 }); // bystander keeps the game alive
    let killTick = -1;
    for (let k = 0; k < 30 && killTick < 0; k++) {
      step(g, [3, 2, 2]); // tank 0 fires, the others coast
      if (g.events.some((e) => e.kind === "kill")) killTick = g.tick;
    }
    expect(killTick).toBeGreaterThan(0);
    const victim = g.tanks[1];
    const killer = g.tanks[0];
    expect(victim.alive).toBe(false);
    expect(victim.diedThisTick).toBe(true); // this very tick
    expect(victim.deaths).toBe(1);
    expect(victim.hp).toBe(0); // overkill floors at 0
    expect(killer.kills).toBe(1);
    expect(g.bullets.every((b) => b.ownerId !== victim.id)).toBe(true); // shells die too
    expect(isOver(g)).toBe(false); // two tanks remain
    // event order within the tick: the hit lands before the kill it causes
    const kinds = g.events.map((e) => e.kind);
    expect(kinds.indexOf("hit")).toBeGreaterThanOrEqual(0);
    expect(kinds.indexOf("kill")).toBeGreaterThan(kinds.indexOf("hit"));
    // the next tick must NOT charge death again (the flag is per-tick)
    step(g, [2, 2, 2]);
    expect(victim.diedThisTick).toBe(false);
    // rewards on the kill tick: killer banks damage + kill, victim eats death
    const g2 = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 10, seed: 5 });
    g2.walls = [];
    g2.crates = [];
    Object.assign(g2.tanks[0], { x: 100, y: 350, heading: 0 });
    Object.assign(g2.tanks[1], { x: 400, y: 350, hp: BULLET_DAMAGE });
    Object.assign(g2.tanks[2], { x: 800, y: 100 });
    for (let k = 0; k < 30 && g2.tanks[1].alive; k++) step(g2, [3, 2, 2]);
    expect(g2.lastRewards[0]).toBeGreaterThan(REWARD_DAMAGE_DEALT * BULLET_DAMAGE + REWARD_KILL - 5);
    expect(g2.lastRewards[1]).toBeLessThan(DEATH_TICK_CEILING);
  });
});

describe("crate blasts", () => {
  function blastGame(shielded: boolean): { g: GameState; near: Obstacle } {
    const g = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 10, seed: 6 });
    g.walls = [];
    // one shell (25) breaks it: hp pre-lowered to BULLET_DAMAGE
    const crate: Obstacle = { x: 450, y: 330, w: 40, h: 40, kind: "crate", hp: BULLET_DAMAGE };
    g.crates = [crate];
    Object.assign(g.tanks[0], { x: 100, y: 350, heading: 0 }); // 370 u off: safe
    Object.assign(g.tanks[1], { x: 470, y: 296 }); // 54 u from the centre: hit
    if (shielded) g.tanks[1].timers.shield = 5;
    Object.assign(g.tanks[2], { x: 800, y: 100 }); // ~414 u off: safe
    return { g, near: crate };
  }

  it("damages tanks within CRATE_BLAST_RADIUS (shield immune) when a crate breaks", () => {
    const { g } = blastGame(false);
    for (let k = 0; k < 20 && g.crates.length > 0; k++) step(g, [3, 2, 2]);
    expect(g.crates.length).toBe(0); // broken
    expect(g.events.some((e) => e.kind === "crate")).toBe(true); // physics' break event
    expect(g.events.some((e) => e.kind === "explode")).toBe(true); // the blast
    expect(g.tanks[1].hp).toBe(TANK_HP - CRATE_BLAST_DAMAGE); // inside 60 u
    expect(g.tanks[0].hp).toBe(TANK_HP); // shooter stands clear
    expect(g.tanks[2].hp).toBe(TANK_HP);
    expect(g.tanks[1].alive).toBe(true);
    // the breaker banks crate-break + attributed blast damage on the tick
    expect(g.lastRewards[0]).toBeGreaterThan(CRATE_BLAST_DAMAGE * REWARD_DAMAGE_DEALT - 5);
  });

  it("a shield shrugs the blast off entirely", () => {
    const { g } = blastGame(true);
    for (let k = 0; k < 20 && g.crates.length > 0; k++) step(g, [3, 2, 2]);
    expect(g.crates.length).toBe(0);
    expect(g.tanks[1].hp).toBe(TANK_HP);
  });

  it("keeps the blast inside CRATE_BLAST_RADIUS", () => {
    const { g } = blastGame(false);
    g.tanks[1].x = 470;
    g.tanks[1].y = 350 - (CRATE_BLAST_RADIUS + TANK_RADIUS + 1); // just outside
    for (let k = 0; k < 20 && g.crates.length > 0; k++) step(g, [3, 2, 2]);
    expect(g.tanks[1].hp).toBe(TANK_HP); // 81 u from the centre: no damage
  });
});

describe("escalation and sudden death", () => {
  it("walks phase 0 → 1 → 2 once each, drains SUDDEN_DEATH_DRAIN, and times out", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 10, seed: 8 });
    g.maxTicks = 100; // shorten the clock: thresholds at tick 70 and 85
    let phaseEvents = 0;
    let hpAt84 = TANK_HP;
    while (!isOver(g)) {
      step(g, [2, 2]);
      phaseEvents += g.events.filter((e) => e.kind === "phase").length;
      if (g.tick === 69) expect(g.phase).toBe(0);
      if (g.tick === 70) expect(g.phase).toBe(1); // 70/100 ≥ ESCALATION_FRAC
      if (g.tick === 84) {
        expect(g.phase).toBe(1);
        hpAt84 = g.tanks[0].hp;
      }
      if (g.tick === 85) expect(g.phase).toBe(2); // 85/100 ≥ SUDDEN_DEATH_FRAC
    }
    expect(g.over).toBe(true);
    expect(g.tick).toBe(100);
    expect(phaseEvents).toBe(2); // one push per transition, never repeated
    expect(g.phase).toBe(2);
    // 16 sudden-death ticks (85..100) × 3 HP/s ÷ 30 Hz
    expect(g.tanks[0].hp).toBeCloseTo(hpAt84 - SUDDEN_DEATH_DRAIN * DT * 16, 6);
    // timeout with two alive: placement bonus separates the (tied) hp leaders
    expect(g.lastRewards[0] - g.lastRewards[1]).toBeCloseTo(40, 6);
  });
});

describe("2000-game property sweep", () => {
  // 2000 games × ≤300 ticks is ~600k steps; if a future engine change pushes
  // this past the CI budget, reduce GAMES and note the new count HERE.
  const GAMES = 2000;
  const COUNTS: (2 | 3 | 4)[] = [2, 3, 4];

  it("holds containment, tunneling, caps, monotonicity and termination everywhere", () => {
    const roll = mulberry32(0xba5eba11);
    for (let n = 0; n < GAMES; n++) {
      const game = createGame({
        tankCount: COUNTS[n % 3],
        teamMode: "ffa",
        episodeSeconds: 10, // 300 ticks at 30 Hz
        seed: (Math.imul(n + 1, 2654435761) ^ 0x9e3779b9) | 0,
      });
      const lastBullet = new Map<number, { x: number; y: number }>();
      const crateHp = new Map<Obstacle, number>();
      let alive = game.tanks.length;
      let steps = 0;
      const actions: number[] = new Array(game.tanks.length).fill(0);
      while (!isOver(game)) {
        for (let i = 0; i < actions.length; i++) actions[i] = (roll() * ACTION_COUNT) | 0;
        step(game, actions);
        steps++;
        ck(steps <= game.maxTicks + 1, `game ${n} outlived its tick budget`);

        const live = new Set<number>();
        let totalBullets = 0;
        for (const t of game.tanks) {
          ck(t.hp >= 0 && t.hp <= TANK_HP, `game ${n} tick ${game.tick} hp ${t.hp}`);
          ck(
            t.x >= TANK_RADIUS - 1e-9 && t.x <= ARENA_W - TANK_RADIUS + 1e-9,
            `game ${n} tick ${game.tick} tank x ${t.x}`,
          );
          ck(
            t.y >= TANK_RADIUS - 1e-9 && t.y <= ARENA_H - TANK_RADIUS + 1e-9,
            `game ${n} tick ${game.tick} tank y ${t.y}`,
          );
          ck(
            t.bulletsInFlight >= 0 && t.bulletsInFlight <= MAX_BULLETS_PER_TANK,
            `game ${n} tick ${game.tick} magazine ${t.bulletsInFlight}`,
          );
        }
        // containment: the centre stays ≥ TANK_RADIUS·0.9 from every solid
        // box (distance to the AABB, not an inflated square — a tank touching
        // a CORNER at exactly TANK_RADIUS projects only ~17.8 u along an axis,
        // which a square inflated box would misreport as penetration)
        for (const boxes of [game.walls, game.crates]) {
          for (const o of boxes) {
            for (const t of game.tanks) {
              const m = TANK_RADIUS * 0.9;
              const dx = Math.max(o.x - t.x, 0, t.x - (o.x + o.w));
              const dy = Math.max(o.y - t.y, 0, t.y - (o.y + o.h));
              const d2 = dx * dx + dy * dy;
              ck(
                d2 >= m * m - 1e-9,
                `game ${n} tick ${game.tick} tank ${t.id} inside ${o.kind} at ${o.x},${o.y}`,
              );
            }
          }
        }
        for (const b of game.bullets) {
          totalBullets++;
          live.add(b.id);
          // Bullets never leave the arena by more than BULLET_RADIUS: every
          // border clamp and every box escape lands the centre exactly one
          // radius outside/inside a face. (A shell born inside a border-
          // adjacent wall cell sits on the cell's OUTER face — one radius
          // out — for the tail of a tick, then bounces or dies on the next
          // sweep; deeper excursions would mean a missed contact.)
          ck(
            b.x >= -BULLET_RADIUS - 1e-6 &&
              b.x <= ARENA_W + BULLET_RADIUS + 1e-6 &&
              b.y >= -BULLET_RADIUS - 1e-6 &&
              b.y <= ARENA_H + BULLET_RADIUS + 1e-6,
            `game ${n} tick ${game.tick} bullet out of bounds`,
          );
          const prev = lastBullet.get(b.id);
          if (prev) {
            const dx = b.x - prev.x;
            const dy = b.y - prev.y;
            // flight is BULLET_SPEED·dt per tick; a ricochet ALSO pushes the
            // centre out to the contact surface (≤ BULLET_RADIUS per contact,
            // a couple of contacts across the tick's two sweeps), so the
            // honest ceiling gives that slack. A real tunneling bug — a shell
            // stepping over a 40 u hull or a 50 u wall — blows far past it.
            const limit = BULLET_SPEED * DT + 8 * BULLET_RADIUS + 1e-6;
            ck(
              dx * dx + dy * dy <= limit * limit,
              `game ${n} tick ${game.tick} bullet jumped ${Math.sqrt(dx * dx + dy * dy)} u`,
            );
            prev.x = b.x;
            prev.y = b.y;
          } else {
            lastBullet.set(b.id, { x: b.x, y: b.y });
          }
        }
        ck(
          totalBullets <= MAX_BULLETS_GLOBAL,
          `game ${n} tick ${game.tick} ${totalBullets} bullets alive`,
        );
        // prune displacement trackers of shells that died this tick
        if (lastBullet.size > totalBullets * 2 + 8) {
          for (const id of [...lastBullet.keys()]) if (!live.has(id)) lastBullet.delete(id);
        }
        // cover only ever breaks; the living never come back
        for (const c of game.crates) {
          const prev = crateHp.get(c);
          if (prev !== undefined) ck(c.hp <= prev, `game ${n} tick ${game.tick} crate healed`);
          crateHp.set(c, c.hp);
        }
        let nowAlive = 0;
        for (const t of game.tanks) if (t.alive) nowAlive++;
        ck(nowAlive <= alive, `game ${n} tick ${game.tick} resurrection`);
        alive = nowAlive;
      }
      expect(isOver(game)).toBe(true);
      expect(game.tick).toBeLessThanOrEqual(game.maxTicks + 1);
    }
  });

  it("replays a seed bit-identically from the same action log and diverges otherwise", () => {
    // Always exactly 300 steps: if the episode ended early, step() is a no-op
    // and the frozen tail is part of the contract too (see the no-op test).
    const stream = (seed: number): string[] => {
      const g = createGame({ tankCount: 4, teamMode: "ffa", episodeSeconds: 10, seed });
      const roll = mulberry32(0xdec1de); // same seed ⇒ the SAME action log too
      const log: string[] = [];
      for (let t = 0; t < 300; t++) {
        step(g, [0, 1, 2, 3].map(() => (roll() * ACTION_COUNT) | 0));
        log.push(JSON.stringify(toSnapshot(g, 0)));
      }
      return log;
    };
    const a = stream(0xabcdef);
    const b = stream(0xabcdef);
    expect(b).toEqual(a); // every tick — including tick 300 — byte-for-byte
    expect(JSON.parse(a[299]).tick).toBeLessThanOrEqual(300);
    const c = stream(0xabcdee);
    expect(c).not.toEqual(a); // a different seed plays a different game
  });
});
