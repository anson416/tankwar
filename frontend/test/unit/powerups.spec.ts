// powerups.spec.ts — the powerup contract: spawn cadence is scheduled inside
// the [8 s, 12 s] band and is a pure function of the seed; the field holds at
// most 4; a spawn is ≥150 u from every living tank and clear of every face;
// floor items despawn after exactly 20 s; each of the six kinds applies its
// exact effect magnitude, expires on time, and repair caps at TANK_HP.
//
// Effect tests INJECT a powerup into game.powerups (a plain public array) and
// step once — the spawn pipeline is covered separately by the seeded cadence
// run, and injection lets each test name the one behaviour it pins instead of
// waiting out an 8–12 s schedule per case.
import { describe, expect, it } from "vitest";
import {
  ARENA_H,
  ARENA_W,
  BOUNCE_POWERUP,
  BOUNCE_TIME,
  BULLET_BOUNCES,
  BULLET_DAMAGE,
  DT,
  FIRE_COOLDOWN,
  MAX_POWERUPS_ON_FIELD,
  MAX_SPEED_FWD,
  PIERCE_TIME,
  POWERUP_LIFETIME,
  POWERUP_SPAWN_MAX_TICKS,
  POWERUP_SPAWN_MIN_TICKS,
  RAPID_COOLDOWN,
  RAPID_TIME,
  REPAIR_AMOUNT,
  REWARD_PICKUP,
  REWARD_PICKUP_HEAL_BONUS,
  SHIELD_TIME,
  SPEED_MULT,
  SPEED_TIME,
  TANK_HP,
  TANK_RADIUS,
  TICK_HZ,
} from "../../src/game/constants";
import { createGame, isOver, step } from "../../src/game/engine";
import type { GameState, PowerUpKind, Tank } from "../../src/game/types";

type TimerKind = Exclude<PowerUpKind, "repair">;

/** A 2-tank FFA arena with the generated geometry removed, so effect tests
 *  can place tanks/bullets on a bare board without a wall in the flight path. */
function bareGame(): GameState {
  const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 90, seed: 0xfeed });
  g.walls = [];
  g.crates = [];
  return g;
}

/** Place both tanks on a horizontal firing line, 0 aiming +x at 1. */
function duelRange(g: GameState, x0: number, x1: number, y: number): void {
  g.tanks[0].x = x0;
  g.tanks[0].y = y;
  g.tanks[0].heading = 0;
  g.tanks[1].x = x1;
  g.tanks[1].y = y;
}

/** Drop `kind` `offset` u to the +x side of `at` — 0 offset lands it on the
 *  tank (instant pickup); the next step's pickup pass does the rest. */
function inject(g: GameState, kind: PowerUpKind, at: { x: number; y: number }, offset = 0): void {
  g.powerups.push({
    id: g.nextPowerupId++,
    kind,
    x: at.x + offset,
    y: at.y,
    despawn: g.tick + POWERUP_LIFETIME * TICK_HZ,
  });
}

/** Step until the gun is ready (a 1 s cooldown is 30 ticks of decay; counting
 *  ticks exactly would race the float edge of 30·(1/30)). */
function waitReload(g: GameState): void {
  for (let k = 0; k < 40 && g.tanks[0].reload > 0; k++) step(g, [2, 2]);
}

describe("pickup effects", () => {
  it("grants each timer kind its exact full duration", () => {
    const kinds: [TimerKind, number][] = [
      ["rapid", RAPID_TIME],
      ["shield", SHIELD_TIME],
      ["speed", SPEED_TIME],
      ["bounce", BOUNCE_TIME],
      ["pierce", PIERCE_TIME],
    ];
    for (const [kind, seconds] of kinds) {
      const g = bareGame();
      inject(g, kind, g.tanks[0]);
      step(g, [2, 2]);
      // pickup runs AFTER this tick's timer decay, so the value is undecayed
      expect(g.tanks[0].timers[kind]).toBe(seconds);
      expect(g.powerups.length).toBe(0); // consumed, not merely touched
      expect(g.events.some((e) => e.kind === "pickup" && e.powerup === kind)).toBe(true);
    }
  });

  it("heals REPAIR_AMOUNT and pays the heal bonus only when it actually healed", () => {
    const hurt = bareGame();
    hurt.tanks[0].hp = 50;
    inject(hurt, "repair", hurt.tanks[0]);
    step(hurt, [2, 2]);
    expect(hurt.tanks[0].hp).toBe(50 + REPAIR_AMOUNT); // 90
    // The two parked tanks are mirror-symmetric, so survival and the shaping
    // terms are IDENTICAL for both — their reward gap is exactly the pickup
    // (+8) plus the heal bonus (+4), with no other term leaking in.
    expect(hurt.lastRewards[0] - hurt.lastRewards[1]).toBeCloseTo(
      REWARD_PICKUP + REWARD_PICKUP_HEAL_BONUS,
      9,
    );
    expect(hurt.lastRewards[0]).toBeGreaterThan(REWARD_PICKUP);

    const full = bareGame();
    inject(full, "repair", full.tanks[0]);
    step(full, [2, 2]);
    expect(full.tanks[0].hp).toBe(TANK_HP); // a full-HP tank stays at TANK_HP
    expect(full.lastRewards[0] - full.lastRewards[1]).toBeCloseTo(REWARD_PICKUP, 9);
  });

  it("caps repair at TANK_HP even from just below it", () => {
    const g = bareGame();
    g.tanks[0].hp = TANK_HP - 5;
    inject(g, "repair", g.tanks[0]);
    step(g, [2, 2]);
    expect(g.tanks[0].hp).toBe(TANK_HP); // min(TANK_HP, 95 + 40)
  });

  it("makes the gun rapid while the timer lives, then reverts to FIRE_COOLDOWN", () => {
    const g = bareGame();
    duelRange(g, 100, 800, 350);
    inject(g, "rapid", g.tanks[0]);
    step(g, [3, 2]); // tick 1: fire on the BASE cooldown, then pick rapid up
    expect(g.tanks[0].reload).toBe(FIRE_COOLDOWN);
    expect(g.tanks[0].timers.rapid).toBe(RAPID_TIME);
    waitReload(g);
    step(g, [3, 2]); // next shot while rapid is still up (8 s ≫ 1 s)
    expect(g.tanks[0].reload).toBe(RAPID_COOLDOWN);
    // deep into the window (230 of 240 ticks) the rapid cooldown still applies
    for (let k = 0; k < 199; k++) step(g, [2, 2]);
    g.tanks[0].reload = 0;
    step(g, [3, 2]);
    expect(g.tanks[0].reload).toBe(RAPID_COOLDOWN);
    // the tick the timer expires, the gun is back to a full second
    g.tanks[0].timers.rapid = DT; // dies in this tick's decay
    g.tanks[0].reload = 0;
    step(g, [3, 2]);
    expect(g.tanks[0].timers.rapid).toBe(0);
    expect(g.tanks[0].reload).toBe(FIRE_COOLDOWN);
  });

  it("speed raises the CAP to SPEED_MULT·MAX_SPEED_FWD, not the acceleration", () => {
    const g = bareGame();
    g.tanks[0].x = 100;
    g.tanks[0].y = 100;
    g.tanks[1].x = 800;
    g.tanks[1].y = 100;
    inject(g, "speed", g.tanks[0]);
    step(g, [8, 8]); // pickup tick
    for (let k = 0; k < 30; k++) step(g, [8, 8]); // 30 ticks: past both caps
    expect(g.tanks[0].speed).toBeCloseTo(MAX_SPEED_FWD * SPEED_MULT, 6); // 336
    expect(g.tanks[1].speed).toBeCloseTo(MAX_SPEED_FWD, 6); // 240, no pickup
  });

  it("bounce ammo spawns shells with BOUNCE_POWERUP ricochets (plain: BULLET_BOUNCES)", () => {
    const g = bareGame();
    duelRange(g, 100, 800, 100);
    inject(g, "bounce", g.tanks[0]);
    step(g, [3, 2]); // this shell leaves BEFORE the pickup stage runs — plain
    expect(g.bullets[0].bouncesLeft).toBe(BULLET_BOUNCES);
    waitReload(g);
    step(g, [3, 2]); // the next shell snapshots the active bounce powerup
    expect(g.bullets[g.bullets.length - 1].bouncesLeft).toBe(BOUNCE_POWERUP);
  });

  it("pierce shells hit each tank in line exactly once and keep flying", () => {
    const g = bareGame();
    duelRange(g, 100, 400, 350);
    const third: Tank = {
      id: 2,
      x: 500,
      y: 350,
      heading: 0,
      speed: 0,
      hp: TANK_HP,
      alive: true,
      reload: 0,
      bulletsInFlight: 0,
      kills: 0,
      deaths: 0,
      team: 2,
      sprite: 1,
      colorIdx: 2,
      timers: { rapid: 0, shield: 0, speed: 0, bounce: 0, pierce: 0 },
      wallContact: false,
      diedThisTick: false,
    };
    g.tanks.push(third);
    inject(g, "pierce", g.tanks[0]);
    step(g, [3, 2, 2]); // pickup tick (this tick's shell is plain and dies on 1)
    waitReload(g);
    step(g, [3, 2, 2]); // the REAL pierce shell: snapshots pierce at fire time
    for (let k = 0; k < 20; k++) step(g, [2, 2, 2]); // ~560 u of flight
    // tank 1 already ate the plain warm-up shell (−25), then the pierce shell
    // damages it once more and passes on to tank 3
    expect(g.tanks[1].hp).toBe(TANK_HP - 2 * BULLET_DAMAGE);
    expect(third.hp).toBe(TANK_HP - BULLET_DAMAGE);
    expect(g.bullets.length).toBe(1); // passed through both, still airborne
  });

  it("a shielded tank shrugs off a direct hit", () => {
    const g = bareGame();
    duelRange(g, 100, 500, 350);
    inject(g, "shield", g.tanks[1]);
    step(g, [2, 2]); // tank 1 picks the shield up
    step(g, [3, 2]); // tank 0 fires straight at it
    for (let k = 0; k < 15; k++) step(g, [2, 2]); // ~420 u of flight
    expect(g.tanks[1].hp).toBe(TANK_HP);
    expect(g.events.every((e) => e.kind !== "hit")).toBe(true);
  });
});

describe("pickup geometry", () => {
  it("collects at TANK_RADIUS + 14 and not a unit farther", () => {
    const close = bareGame();
    inject(close, "rapid", close.tanks[0], 33); // 33 < 34 = 20 + 14
    step(close, [2, 2]);
    expect(close.powerups.length).toBe(0);
    expect(close.tanks[0].timers.rapid).toBe(RAPID_TIME);

    const far = bareGame();
    inject(far, "rapid", far.tanks[0], 35); // 35 > 34: out of reach
    step(far, [2, 2]);
    expect(far.powerups.length).toBe(1);
    expect(far.tanks[0].timers.rapid).toBe(0);
  });
});

describe("floor lifetime", () => {
  it("despawns after exactly POWERUP_LIFETIME (20 s = 600 ticks), uncollected", () => {
    const g = bareGame();
    inject(g, "speed", { x: g.tanks[0].x, y: g.tanks[0].y + 100 }); // 100 u off: untouchable
    const injected = g.powerups[0].id;
    const life = POWERUP_LIFETIME * TICK_HZ;
    for (let k = 1; k <= life; k++) {
      step(g, [2, 2]);
      // the engine's own schedule also spawns during this window — track the
      // injected item by id, not by field size
      const mine = g.powerups.find((p) => p.id === injected);
      expect(mine === undefined).toBe(k === life); // gone ON the despawn tick
    }
  });
});

describe("spawn pipeline (seeded 90 s run)", () => {
  const SECONDS = 90;

  /** Play a coast-only episode (tanks parked at their spawns), returning every
   *  spawn event. Coast-only keeps the tanks fixed, so the ≥150 u rule is
   *  checked at its most static — and nobody ever picks anything up, so the
   *  field population is spawn/despawn alone. */
  function run(seed: number): { tick: number; x: number; y: number; kind: PowerUpKind }[] {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: SECONDS, seed });
    const spawns: { tick: number; x: number; y: number; kind: PowerUpKind }[] = [];
    while (!isOver(g)) {
      step(g, [2, 2]);
      expect(g.powerups.length).toBeLessThanOrEqual(MAX_POWERUPS_ON_FIELD);
      for (const e of g.events) {
        if (e.kind !== "spawn") continue;
        expect(e.powerup).toBeDefined();
        spawns.push({ tick: g.tick, x: e.x, y: e.y, kind: e.powerup as PowerUpKind });
      }
    }
    expect(g.tick).toBe(SECONDS * TICK_HZ);
    return spawns;
  }

  it("spawns on the [8 s, 12 s] schedule, ≥150 u from every tank, clear of faces", () => {
    const spawns = run(0xc0ffee);
    expect(spawns.length).toBeGreaterThanOrEqual(6); // 2700 ticks at ≤360 per gap
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: SECONDS, seed: 0xc0ffee });
    for (let i = 0; i < spawns.length; i++) {
      if (i > 0) {
        const gap = spawns[i].tick - spawns[i - 1].tick;
        expect(gap).toBeGreaterThanOrEqual(POWERUP_SPAWN_MIN_TICKS); // never early
        expect(gap).toBeLessThanOrEqual(POWERUP_SPAWN_MAX_TICKS); // never late
      } else {
        expect(spawns[0].tick).toBeGreaterThanOrEqual(POWERUP_SPAWN_MIN_TICKS);
      }
      // ≥150 u from both parked tanks (spawns are the fixed quarter points)
      for (const sp of g.spawns) {
        expect(Math.hypot(spawns[i].x - sp.x, spawns[i].y - sp.y)).toBeGreaterThanOrEqual(150);
      }
      // not inside — or wedged against — any obstacle face
      for (const o of [...g.walls, ...g.crates]) {
        const wedged =
          spawns[i].x > o.x - TANK_RADIUS &&
          spawns[i].x < o.x + o.w + TANK_RADIUS &&
          spawns[i].y > o.y - TANK_RADIUS &&
          spawns[i].y < o.y + o.h + TANK_RADIUS;
        expect(wedged).toBe(false);
      }
      // the spawn sits inside the play field with the sampling margin
      expect(spawns[i].x).toBeGreaterThanOrEqual(0);
      expect(spawns[i].x).toBeLessThanOrEqual(ARENA_W);
      expect(spawns[i].y).toBeGreaterThanOrEqual(0);
      expect(spawns[i].y).toBeLessThanOrEqual(ARENA_H);
    }
    // the roulette reaches the field over a full episode (deterministic seed,
    // so this is a pinned fact about seed 0xc0ffee, not a statistical bet)
    expect(new Set(spawns.map((s) => s.kind)).size).toBeGreaterThanOrEqual(3);
  });

  it("books despawn at spawn tick + POWERUP_LIFETIME·TICK_HZ", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 20, seed: 0xbeef });
    let spawnedAt = -1;
    while (!isOver(g)) {
      step(g, [2, 2]);
      if (spawnedAt < 0 && g.powerups.length > 0) {
        spawnedAt = g.tick;
        expect(g.powerups[0].despawn - g.tick).toBe(POWERUP_LIFETIME * TICK_HZ);
      }
    }
    expect(spawnedAt).toBeGreaterThan(0); // 600-tick episode still spawns once
  });

  it("replays identically from the same seed and diverges on a different one", () => {
    const a = run(0x1234);
    const b = run(0x1234);
    expect(b).toEqual(a); // same ticks, positions and kinds, in order
    const c = run(0x1235);
    expect(c).not.toEqual(a);
  });
});
