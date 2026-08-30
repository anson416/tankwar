// physics.spec.ts — the movement/bullet/ray contract: ricochet reflection about
// the AABB face normal, the documented accel/coast/brake stop distances, turn
// rate ceilings, fire-control cooldowns and magazine caps, pierce-once, and the
// raycaster's exact distances. Fixtures are minimal (one wall, two tanks) so a
// failure names the broken law rather than a layout.
import { describe, expect, it } from "vitest";
import { decode } from "../../src/game/actions";
import {
  ACCEL,
  BOUNCE_POWERUP,
  BULLET_BOUNCES,
  BULLET_DAMAGE,
  BULLET_LIFETIME,
  BULLET_RADIUS,
  BULLET_SPEED,
  BULLET_SUBSTEPS,
  COAST_DECEL,
  BRAKE_DECEL,
  CRATE_HP,
  DT,
  FIRE_COOLDOWN,
  MAX_BULLETS_GLOBAL,
  MAX_BULLETS_PER_TANK,
  MAX_SPEED_FWD,
  MAX_SPEED_REV,
  MUZZLE_OFFSET,
  RAPID_COOLDOWN,
  RAY_LEN,
  SPEED_MULT,
  TANK_HP,
  TANK_RADIUS,
  TURN_MULT,
  TURN_RATE,
} from "../../src/game/constants";
import {
  integrateBullets,
  integrateTank,
  raycast,
  tryFire,
} from "../../src/game/physics";
import { nextState, rand } from "../../src/game/rng";
import type { Bullet, GameState, Obstacle, Tank } from "../../src/game/types";

// A 40×40 block centred at (450, 350) — faces at x=430/470, y=330/370.
const BLOCK: Obstacle = { x: 430, y: 330, w: 40, h: 40, kind: "wall", hp: 0 };

/** A tank at rest, facing +x, unless `over` says otherwise. */
function makeTank(id: number, x: number, y: number, over: Partial<Tank> = {}): Tank {
  return {
    id,
    x,
    y,
    heading: 0,
    speed: 0,
    hp: TANK_HP,
    alive: true,
    reload: 0,
    bulletsInFlight: 0,
    kills: 0,
    deaths: 0,
    team: id,
    sprite: 0,
    colorIdx: id % 4,
    timers: { rapid: 0, shield: 0, speed: 0, bounce: 0, pierce: 0 },
    wallContact: false,
    diedThisTick: false,
    ...over,
  };
}

/** Minimal but fully-populated world (every field the physics pass reads). */
function makeGame(over: Partial<GameState> = {}): GameState {
  return {
    tick: 0,
    tanks: [],
    bullets: [],
    powerups: [],
    walls: [],
    crates: [],
    spawns: [],
    phase: 0,
    maxTicks: 1350,
    nextPowerupTick: 240,
    rngState: 0x5eed1234 | 0,
    teamMode: "ffa",
    over: false,
    arenaSeed: 1,
    gamma: 0.97, // the shaping default (rewards.ts REWARD_GAMMA); fixtures never shape
    lastRewards: [],
    events: [],
    nextBulletId: 1,
    nextPowerupId: 1,
    ...over,
  };
}

function makeBullet(over: Partial<Bullet> = {}): Bullet {
  return {
    id: 1,
    ownerId: 0,
    team: 0,
    x: 0,
    y: 0,
    dir: 0,
    bouncesLeft: BULLET_BOUNCES,
    pierce: false,
    life: BULLET_LIFETIME,
    hit: [],
    ...over,
  };
}

describe("bullet flight geometry anchors", () => {
  it("pins the anti-tunneling arithmetic literally: 840 u/s over 2 substeps = 14 u/sweep", () => {
    // The no-tunneling argument (a 14 u sweep « the 40 u tank diameter and any
    // wall thickness) is arithmetic over these three literals — a retune of
    // any of them silently changes the guarantee, so they are pinned raw.
    expect(BULLET_SPEED).toBe(840);
    expect(BULLET_SUBSTEPS).toBe(2);
    expect(BULLET_SPEED / BULLET_SUBSTEPS * DT).toBeLessThan(2 * TANK_RADIUS);
  });
});

describe("tank acceleration model", () => {
  it("reaches MAX_SPEED_FWD from rest in 0.5 s (15 ticks of ACCEL 480)", () => {
    const t = makeTank(0, 450, 350);
    for (let k = 1; k <= 15; k++) {
      integrateTank(t, decode(8), DT, [], [], []); // 8 = fwd straight
      // closed form: min(cap, ACCEL·dt·k) — 16 u/s per tick
      const want = Math.min(MAX_SPEED_FWD, (ACCEL * DT) * k);
      expect(t.speed).toBeCloseTo(want, 6);
    }
    expect(t.speed).toBeCloseTo(MAX_SPEED_FWD, 6); // 240 at exactly 0.5 s
  });

  it("caps reverse at MAX_SPEED_REV and accelerates backward at ACCEL", () => {
    const t = makeTank(0, 450, 350);
    for (let k = 1; k <= 14; k++) {
      integrateTank(t, decode(14), DT, [], [], []); // 14 = rev straight
      const want = Math.max(-MAX_SPEED_REV, -(ACCEL * DT) * k);
      expect(t.speed).toBeCloseTo(want, 6);
    }
    expect(t.speed).toBeCloseTo(-MAX_SPEED_REV, 6); // −140, never below
  });

  it("coasts to a stop in ~0.67 s (COAST_DECEL 360): 0 after 20 ticks, not after 19", () => {
    const t = makeTank(0, 450, 350, { speed: MAX_SPEED_FWD });
    for (let k = 1; k <= 19; k++) {
      integrateTank(t, decode(2), DT, [], [], []); // 2 = coast straight
      expect(t.speed).toBeCloseTo(MAX_SPEED_FWD - COAST_DECEL * DT * k, 6);
    }
    expect(t.speed).toBeGreaterThan(0); // still rolling at 0.633 s
    integrateTank(t, decode(2), DT, [], [], []);
    expect(t.speed).toBe(0); // stops AT zero at ~0.667 s …
    for (let k = 0; k < 10; k++) integrateTank(t, decode(2), DT, [], [], []);
    expect(t.speed).toBe(0); // … and never coasts backwards through it
  });

  it("brakes (opposite throttle) to a stop in ~0.34 s: hard stop, no sign flip", () => {
    const t = makeTank(0, 450, 350, { speed: MAX_SPEED_FWD });
    for (let k = 1; k <= 10; k++) {
      integrateTank(t, decode(14), DT, [], [], []); // rev held while rolling fwd
      expect(t.speed).toBeCloseTo(MAX_SPEED_FWD - BRAKE_DECEL * DT * k, 6);
    }
    expect(t.speed).toBeGreaterThan(0); // ~6.7 u/s left at 0.333 s
    integrateTank(t, decode(14), DT, [], [], []);
    expect(t.speed).toBe(0); // stopped by 0.367 s, clamped at 0 not negative
  });

  it("the speed powerup raises the speed CAP (×SPEED_MULT), not the accel", () => {
    const t = makeTank(0, 450, 350, { timers: { rapid: 0, shield: 0, speed: 5, bounce: 0, pierce: 0 } });
    for (let k = 0; k < 30; k++) integrateTank(t, decode(8), DT, [], [], []);
    expect(t.speed).toBeCloseTo(MAX_SPEED_FWD * SPEED_MULT, 6); // 336
  });
});

describe("turn rate", () => {
  it("turns at exactly TURN_RATE per second (left = heading−, right = heading+)", () => {
    const right = makeTank(0, 450, 350);
    integrateTank(right, decode(10), DT, [], [], []); // fwd right
    expect(right.heading).toBeCloseTo(TURN_RATE * DT, 9); // 0.12 rad/tick
    const left = makeTank(0, 450, 350);
    integrateTank(left, decode(6), DT, [], [], []); // fwd left
    expect(left.heading).toBeCloseTo(-TURN_RATE * DT, 9);
  });

  it("never exceeds TURN_RATE·TURN_MULT per tick, and that ceiling is exact with speed active", () => {
    const boosted = TURN_RATE * TURN_MULT;
    const t = makeTank(0, 450, 350, { timers: { rapid: 0, shield: 0, speed: 5, bounce: 0, pierce: 0 } });
    for (let k = 0; k < 90; k++) {
      const before = t.heading;
      integrateTank(t, decode(10), DT, [], [], []);
      const step = t.heading - before;
      expect(step).toBeLessThanOrEqual(boosted * DT + 1e-9);
    }
    expect(t.heading).toBeCloseTo(90 * boosted * DT, 9); // 0.138 rad/tick
    // without the powerup the ceiling is plain TURN_RATE (boost never leaks)
    const plain = makeTank(0, 450, 350);
    for (let k = 0; k < 90; k++) {
      const before = plain.heading;
      integrateTank(plain, decode(10), DT, [], [], []);
      expect(plain.heading - before).toBeLessThanOrEqual(TURN_RATE * DT + 1e-9);
    }
    expect(plain.heading).toBeCloseTo(90 * TURN_RATE * DT, 9);
  });
});

describe("tank resolution", () => {
  it("clamps to the arena border and flags wallContact while grinding, not at rest", () => {
    const t = makeTank(0, 12, 350, { heading: Math.PI, speed: MAX_SPEED_FWD });
    integrateTank(t, decode(2), DT, [], [], []);
    expect(t.x).toBe(TANK_RADIUS); // 20, pushed back inside
    expect(t.wallContact).toBe(true);
    integrateTank(t, decode(2), DT, [], [], []); // still driving into the wall…
    expect(t.wallContact).toBe(true); // …so it keeps scraping: flag again
    t.speed = 0; // parked exactly on the boundary, no throttle
    t.wallContact = false;
    integrateTank(t, decode(2), DT, [], [], []);
    expect(t.x).toBe(TANK_RADIUS);
    expect(t.wallContact).toBe(false); // resting is not scraping — no reward tick
  });

  it("separates overlapping tanks (full overlap on the integrated tank)", () => {
    const a = makeTank(0, 450, 350);
    const b = makeTank(1, 462, 350); // 12 apart, 40 needed
    integrateTank(a, decode(2), DT, [], [], [a, b]);
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    expect(Math.sqrt(dx * dx + dy * dy)).toBeGreaterThanOrEqual(TANK_RADIUS * 2 - 1e-6);
    expect(a.wallContact).toBe(true);
  });

  it("is pushed out of a crate and reports contact", () => {
    const t = makeTank(0, 560, 350, { speed: MAX_SPEED_FWD });
    const crate: Obstacle = { x: 600, y: 330, w: 40, h: 40, kind: "crate", hp: CRATE_HP };
    for (let k = 0; k < 3; k++) integrateTank(t, decode(8), DT, [], [crate], [t]);
    expect(t.x).toBeCloseTo(600 - TANK_RADIUS, 6); // resting on the face
    expect(t.wallContact).toBe(true);
  });
});

describe("fire control", () => {
  it("starts exactly FIRE_COOLDOWN, or RAPID_COOLDOWN while rapid is active", () => {
    const g = makeGame({ tanks: [makeTank(0, 450, 350)] });
    expect(tryFire(g, g.tanks[0])).toBe(true);
    expect(g.tanks[0].reload).toBe(FIRE_COOLDOWN); // exact, no float slop
    expect(g.bullets.length).toBe(1);
    g.tanks[0].reload = 0;
    g.tanks[0].timers.rapid = 3;
    expect(tryFire(g, g.tanks[0])).toBe(true);
    expect(g.tanks[0].reload).toBe(RAPID_COOLDOWN);
  });

  it("spawns at the muzzle along the heading, outside the hull", () => {
    const g = makeGame({ tanks: [makeTank(0, 450, 350)] });
    g.tanks[0].heading = 0.7;
    expect(tryFire(g, g.tanks[0])).toBe(true);
    const b = g.bullets[0];
    expect(b.x).toBeCloseTo(450 + Math.cos(0.7) * MUZZLE_OFFSET, 9);
    expect(b.y).toBeCloseTo(350 + Math.sin(0.7) * MUZZLE_OFFSET, 9);
    expect(b.dir).toBe(0.7);
    expect(b.ownerId).toBe(0);
  });

  it("refuses while reloading, when dead, at 6 in flight, and at the 24-bullet global cap", () => {
    const g = makeGame({ tanks: [makeTank(0, 450, 350)] });
    const t = g.tanks[0];
    t.reload = 0.5;
    expect(tryFire(g, t)).toBe(false);
    t.reload = 0;
    t.alive = false;
    expect(tryFire(g, t)).toBe(false);
    t.alive = true;
    t.bulletsInFlight = MAX_BULLETS_PER_TANK;
    expect(tryFire(g, t)).toBe(false); // the 7th shell with a full magazine
    t.bulletsInFlight = MAX_BULLETS_PER_TANK - 1;
    for (let i = 0; i < MAX_BULLETS_GLOBAL; i++) {
      g.bullets.push(makeBullet({ id: 100 + i, ownerId: 1 }));
    }
    expect(tryFire(g, t)).toBe(false); // arena-wide cap
    expect(g.bullets.length).toBe(MAX_BULLETS_GLOBAL);
  });
});

describe("ricochet reflects about the AABB face normal", () => {
  // Each case asserts v' = v − 2(v·n)n computed HERE from the face normal,
  // independently of the implementation's axis bookkeeping — this is the
  // reflection-law pin, not a round-trip through the same math.
  const cases: { name: string; face: "left" | "right" | "top" | "bottom"; dir: number }[] = [
    { name: "left face, +0.3 rad", face: "left", dir: 0.3 },
    { name: "left face, −0.3 rad", face: "left", dir: -0.3 },
    { name: "right face, π−0.3 rad", face: "right", dir: Math.PI - 0.3 },
    { name: "right face, π+0.3 rad", face: "right", dir: Math.PI + 0.3 },
    { name: "top face, π/2−0.3 rad", face: "top", dir: Math.PI / 2 - 0.3 },
    { name: "top face, π/2+0.3 rad", face: "top", dir: Math.PI / 2 + 0.3 },
    { name: "bottom face, −π/2−0.3 rad", face: "bottom", dir: -Math.PI / 2 - 0.3 },
    { name: "bottom face, −π/2+0.3 rad", face: "bottom", dir: -Math.PI / 2 + 0.3 },
  ];
  const normals: Record<string, [number, number]> = {
    left: [-1, 0],
    right: [1, 0],
    top: [0, -1],
    bottom: [0, 1],
  };

  for (const c of cases) {
    it(`case: ${c.name}`, () => {
      const n = normals[c.face];
      // block centre (450,350) + n·(20 half-width + 26 standoff): 26 u off the
      // face, on the side the normal points to — 22 u of flight to contact
      const start = {
        x: 450 + n[0] * 46,
        y: 350 + n[1] * 46,
      };
      const b = makeBullet({ x: start.x, y: start.y, dir: c.dir });
      const g = makeGame({ walls: [BLOCK], tanks: [makeTank(0, 100, 100)] });
      g.bullets.push(b);
      integrateBullets(g, DT);
      // the independent reflection: incoming v, face normal n (unit)
      const vx = Math.cos(c.dir);
      const vy = Math.sin(c.dir);
      const dot = vx * n[0] + vy * n[1]; // < 0: travelling into the face
      expect(dot).toBeLessThan(0);
      const wx = vx - 2 * dot * n[0];
      const wy = vy - 2 * dot * n[1];
      expect(b.bouncesLeft).toBe(BULLET_BOUNCES - 1);
      expect(g.bullets.length).toBe(1);
      expect(g.events.filter((e) => e.kind === "ricochet").length).toBe(1);
      expect(Math.cos(b.dir)).toBeCloseTo(wx, 6);
      expect(Math.sin(b.dir)).toBeCloseTo(wy, 6);
      expect(g.events[0].kind === "ricochet" && g.events[0].angle !== undefined).toBe(true);
    });
  }

  it("holds the same reflection law for seeded random angles on all four faces", () => {
    let s = makeGame().rngState;
    for (let i = 0; i < 24; i++) {
      const faces: ("left" | "right" | "top" | "bottom")[] = ["left", "right", "top", "bottom"];
      const face = faces[i % 4];
      const n = normals[face];
      s = nextState(s);
      const u1 = rand(s);
      s = nextState(s);
      const u2 = rand(s);
      const tilt = 0.05 + u1 * 0.5; // ≤ 0.6 rad keeps the face the dominant approach
      const base = face === "left" ? 0 : face === "right" ? Math.PI : face === "top" ? Math.PI / 2 : -Math.PI / 2;
      const dir = base + (u2 < 0.5 ? tilt : -tilt);
      const b = makeBullet({ x: 450 + n[0] * 46, y: 350 + n[1] * 46, dir });
      const g = makeGame({ walls: [BLOCK], tanks: [makeTank(0, 100, 100)] });
      g.bullets.push(b);
      integrateBullets(g, DT);
      const vx = Math.cos(dir);
      const vy = Math.sin(dir);
      const dot = vx * n[0] + vy * n[1];
      expect(Math.cos(b.dir)).toBeCloseTo(vx - 2 * dot * n[0], 6);
      expect(Math.sin(b.dir)).toBeCloseTo(vy - 2 * dot * n[1], 6);
      expect(b.bouncesLeft).toBe(BULLET_BOUNCES - 1);
    }
  });

  it("a wall thinner than one substep's 14 u sweep still cannot be tunnelled", () => {
    const thin: Obstacle = { x: 448, y: 340, w: 4, h: 20, kind: "wall", hp: 0 };
    const b = makeBullet({ x: 420, y: 350, dir: 0 }); // 28 u from the near face
    const g = makeGame({ walls: [thin], tanks: [makeTank(0, 100, 100)] });
    g.bullets.push(b);
    integrateBullets(g, DT);
    // a naive 28 u/tick advance lands at 448+14 = past the far face; the
    // substepped sweep must have contacted it instead
    expect(g.events.some((e) => e.kind === "ricochet" || e.kind === "explode")).toBe(true);
    expect(b.x).toBeLessThanOrEqual(452 + BULLET_RADIUS);
  });

  it("the arena border bounces while bounces remain and dies when they are spent", () => {
    const g = makeGame({ tanks: [makeTank(0, 100, 100)] });
    const bouncer = makeBullet({ x: 30, y: 350, dir: Math.PI }); // heading −x
    g.bullets.push(bouncer);
    integrateBullets(g, DT);
    expect(g.bullets.length).toBe(1);
    expect(Math.cos(bouncer.dir)).toBeGreaterThan(0); // now travelling +x
    g.bullets = [];
    g.events.length = 0;
    const spent = makeBullet({ x: 30, y: 350, dir: Math.PI, bouncesLeft: 0 });
    g.bullets.push(spent);
    integrateBullets(g, DT);
    expect(g.bullets.length).toBe(0); // bounds consume bounces like any wall
    expect(g.events.some((e) => e.kind === "explode")).toBe(true);
  });
});

describe("bullet vs tank", () => {
  function battlefield(pierce: boolean): GameState {
    const owner = makeTank(0, 100, 100);
    const t1 = makeTank(1, 500, 350);
    const t2 = makeTank(2, 560, 350);
    const g = makeGame({ tanks: [owner, t1, t2] });
    g.bullets.push(
      makeBullet({ x: 300, y: 350, dir: 0, pierce, bouncesLeft: pierce ? BULLET_BOUNCES : 0 }),
    );
    owner.bulletsInFlight = 1;
    return g;
  }

  it("a pierce bullet damages each tank in its path exactly once and keeps flying", () => {
    const g = battlefield(true);
    for (let k = 0; k < 15; k++) integrateBullets(g, DT); // ~420 u travelled
    expect(g.tanks[1].hp).toBe(TANK_HP - BULLET_DAMAGE);
    expect(g.tanks[2].hp).toBe(TANK_HP - BULLET_DAMAGE);
    expect(g.bullets.length).toBe(1);
    expect(g.bullets[0].hit).toEqual([1, 2]);
    expect(g.events.filter((e) => e.kind === "hit" && e.tank === 1).length).toBe(1);
    expect(g.events.filter((e) => e.kind === "hit" && e.tank === 2).length).toBe(1);
  });

  it("a plain bullet dies on its first tank hit, crediting the magazine back", () => {
    const g = battlefield(false);
    for (let k = 0; k < 10; k++) integrateBullets(g, DT);
    expect(g.tanks[1].hp).toBe(TANK_HP - BULLET_DAMAGE);
    expect(g.tanks[2].hp).toBe(TANK_HP); // never reached
    expect(g.bullets.length).toBe(0);
    expect(g.tanks[0].bulletsInFlight).toBe(0);
    expect(g.events.filter((e) => e.kind === "hit").length).toBe(1);
    expect(g.events.some((e) => e.kind === "explode")).toBe(true);
  });

  it("a shielded tank is untouched and the bullet passes through without marking a hit", () => {
    const g = makeGame({ tanks: [makeTank(0, 100, 100), makeTank(2, 500, 350, { timers: { rapid: 0, shield: 2, speed: 0, bounce: 0, pierce: 0 } })] });
    g.bullets.push(makeBullet({ x: 300, y: 350, dir: 0, bouncesLeft: 0 }));
    for (let k = 0; k < 15; k++) integrateBullets(g, DT);
    expect(g.tanks[1].hp).toBe(TANK_HP);
    expect(g.events.filter((e) => e.kind === "hit").length).toBe(0);
  });

  it("allied mode blocks same-team hits but not enemy hits", () => {
    const ally = makeTank(1, 500, 350, { team: 0 });
    const g = makeGame({ teamMode: "allied", tanks: [makeTank(0, 100, 100, { team: 0 }), ally] });
    g.bullets.push(makeBullet({ x: 300, y: 350, dir: 0, team: 0, bouncesLeft: 0 }));
    integrateBullets(g, DT);
    for (let k = 0; k < 9; k++) integrateBullets(g, DT);
    expect(ally.hp).toBe(TANK_HP);
    const enemy = makeTank(1, 500, 350, { team: 1 });
    const g2 = makeGame({ teamMode: "allied", tanks: [makeTank(0, 100, 100, { team: 0 }), enemy] });
    g2.bullets.push(makeBullet({ x: 300, y: 350, dir: 0, team: 0, bouncesLeft: 0 }));
    for (let k = 0; k < 10; k++) integrateBullets(g2, DT);
    expect(enemy.hp).toBe(TANK_HP - BULLET_DAMAGE);
  });

  it("an expired miss reports the owner; an expired bullet that hit something does not", () => {
    const owner = makeTank(0, 100, 100);
    const g = makeGame({ tanks: [owner] });
    owner.bulletsInFlight = 1;
    g.bullets.push(makeBullet({ x: 700, y: 350, dir: 0, life: 0.02 }));
    integrateBullets(g, DT);
    expect(g.bullets.length).toBe(0);
    expect(owner.bulletsInFlight).toBe(0);
    expect(g.events.some((e) => e.kind === "explode" && e.tank === 0)).toBe(true);

    const g2 = makeGame({ tanks: [makeTank(0, 100, 100)] });
    g2.bullets.push(makeBullet({ id: 2, x: 700, y: 350, dir: 0, life: 0.02, hit: [2] })); // pierced tank 2 earlier
    integrateBullets(g2, DT);
    expect(g2.bullets.length).toBe(0);
    expect(g2.events.every((e) => e.kind !== "explode" || e.tank === undefined)).toBe(true);
  });
});

describe("bullet vs crate", () => {
  const crate: Obstacle = { x: 600, y: 330, w: 40, h: 40, kind: "crate", hp: CRATE_HP };

  it("takes BULLET_DAMAGE per hit, breaks on the third (3×25 > 60), and emits its event", () => {
    const g = makeGame({ crates: [{ ...crate }], tanks: [makeTank(0, 100, 100)] });
    g.bullets.push(makeBullet({ x: 576, y: 350, dir: 0 }));
    integrateBullets(g, DT);
    expect(g.crates.length).toBe(1);
    expect(g.crates[0].hp).toBe(CRATE_HP - BULLET_DAMAGE); // 35
    expect(g.bullets[0].hit).toEqual([-1]); // crate 0 recorded as −(0+1)

    for (let shot = 2; shot <= 3; shot++) {
      g.bullets = [];
      g.events.length = 0;
      g.bullets.push(makeBullet({ id: shot, x: 576, y: 350, dir: 0 }));
      integrateBullets(g, DT);
    }
    expect(g.crates.length).toBe(0); // 3×25 = 75 > 60
    const broke = g.events.find((e) => e.kind === "crate");
    expect(broke).toBeDefined();
    expect(broke?.x).toBeCloseTo(620, 6); // crate centre
    expect(broke?.y).toBeCloseTo(350, 6);
  });

  it("the bounce-ammo powerup grants BOUNCE_POWERUP ricochets", () => {
    const g = makeGame({ tanks: [makeTank(0, 450, 350)] });
    g.tanks[0].timers.bounce = 5;
    expect(tryFire(g, g.tanks[0])).toBe(true);
    expect(g.bullets[0].bouncesLeft).toBe(BOUNCE_POWERUP);
  });
});

describe("raycast", () => {
  it("reports a wall face at its exact distance (100 u)", () => {
    const g = makeGame({ walls: [{ x: 100, y: 340, w: 40, h: 20, kind: "wall", hp: 0 }] });
    const hit = raycast(g, 0, 350, 0, RAY_LEN);
    expect(hit.kind).toBe("wall");
    expect(hit.tankId).toBe(-1);
    expect(hit.dist).toBeCloseTo(100, 9);
  });

  it("reports an enemy tank with its id at the circle's entry distance", () => {
    const g = makeGame({ tanks: [makeTank(7, 450, 350)] });
    const hit = raycast(g, 200, 350, 0, RAY_LEN);
    expect(hit.kind).toBe("tank");
    expect(hit.tankId).toBe(7);
    expect(hit.dist).toBeCloseTo(250 - TANK_RADIUS, 6); // surface, not centre
  });

  it("returns maxLen/null on an empty arena and clips at maxLen", () => {
    const empty = raycast(makeGame(), 450, 350, 0, RAY_LEN);
    expect(empty.kind).toBe(null);
    expect(empty.dist).toBe(RAY_LEN);
    const clipped = raycast(makeGame({ tanks: [makeTank(3, 450, 350)] }), 200, 350, 0, 100);
    expect(clipped.kind).toBe(null); // tank entry at 230 is beyond the 100 u ray
    expect(clipped.dist).toBe(100);
  });

  it("prefers the nearest hit among wall/crate/tank and sees through corpses", () => {
    const g = makeGame({
      walls: [{ x: 100, y: 340, w: 40, h: 20, kind: "wall", hp: 0 }],
      crates: [{ x: 200, y: 340, w: 40, h: 20, kind: "crate", hp: CRATE_HP }, { x: 240, y: 340, w: 40, h: 20, kind: "crate", hp: 0 }],
      tanks: [makeTank(7, 450, 350), makeTank(8, 60, 350, { alive: false })],
    });
    expect(raycast(g, 0, 350, 0, RAY_LEN).kind).toBe("wall"); // 100 < 200 < 230
    const g2 = makeGame({
      crates: [{ x: 200, y: 340, w: 40, h: 20, kind: "crate", hp: CRATE_HP }],
      tanks: [makeTank(7, 450, 350)],
    });
    const hit = raycast(g2, 0, 350, 0, RAY_LEN);
    expect(hit.kind).toBe("crate"); // 200 beats the tank at 230
    expect(hit.dist).toBeCloseTo(200, 9);
    // dead crate (hp 0) and dead tank are transparent
    const g3 = makeGame({
      crates: [{ x: 200, y: 340, w: 40, h: 20, kind: "crate", hp: 0 }],
      tanks: [makeTank(8, 500, 350, { alive: false })],
    });
    expect(raycast(g3, 0, 350, 0, RAY_LEN).kind).toBe(null);
  });
});
