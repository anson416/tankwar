// encoding.spec.ts — pins the 282-float observation contract: block layout,
// body-frame rotation, entity sorting, sentinel rows, perspective mirroring and
// the ray fan. These are SPEC-level anchors (plan §Observation): the network's
// input layer and every replay row are laid out by exactly this table, so a
// silent layout change would train a net on scrambled features — hence the
// arithmetic here is asserted, not assumed.
//
// The ray assertions use `raycast` from physics.ts (the real module, same as
// production). The chosen geometry is analytic — axis-aligned slabs and tank
// circles whose entry distances are exact by hand — so the expectations hold
// for any correct caster and do not encode this module's implementation.
import { describe, expect, it } from "vitest";
import {
  BULLET_SLOT,
  OBS_DIM,
  OFF,
  POWERUP_SLOT,
  RAY_SLOT,
  TANK_SLOT,
  encodeObs,
} from "../../src/game/encoding";
import {
  ARENA_H,
  ARENA_W,
  BOUNCE_POWERUP,
  CRATE_HP,
  MAX_BULLETS_GLOBAL,
  NORM_DIST,
  POWERUP_LIFETIME,
  RAY_LEN,
  TANK_HP,
  TANK_RADIUS,
  TICK_HZ,
} from "../../src/game/constants";
import { mulberry32 } from "./util/rng";
import type {
  Bullet,
  GameState,
  Obstacle,
  PowerUp,
  PowerUpKind,
  Tank,
  TeamMode,
} from "../../src/game/types";

// --- state builders -------------------------------------------------------------
//
// A GameState with every field the type demands but only the geometry a test
// cares about: ids are unique per game, obstacles are axis-aligned, and the
// defaults put the first tank at the arena centre facing +x (heading 0), which
// makes "ahead / right / behind" readable straight off the coordinates.

function emptyGame(teamMode: TeamMode = "ffa"): GameState {
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
    rngState: 1,
    teamMode,
    over: false,
    arenaSeed: 1,
    gamma: 0.97, // the shaping default (rewards.ts REWARD_GAMMA); the encoder never reads it
    lastRewards: [],
    events: [],
    nextBulletId: 1,
    nextPowerupId: 1,
  };
}

function addTank(game: GameState, patch: Partial<Tank> = {}): Tank {
  const t: Tank = {
    id: game.tanks.length,
    x: 450,
    y: 350,
    heading: 0,
    speed: 0,
    hp: TANK_HP,
    alive: true,
    reload: 0,
    bulletsInFlight: 0,
    kills: 0,
    deaths: 0,
    team: game.teamMode === "allied" ? 0 : game.tanks.length,
    sprite: 0,
    colorIdx: game.tanks.length,
    timers: { rapid: 0, shield: 0, speed: 0, bounce: 0, pierce: 0 },
    wallContact: false,
    diedThisTick: false,
    ...patch,
  };
  game.tanks.push(t);
  return t;
}

function addBullet(game: GameState, patch: Partial<Bullet>): Bullet {
  const b: Bullet = {
    id: 1000 + game.bullets.length,
    ownerId: 0,
    team: 0,
    x: 450,
    y: 350,
    dir: 0,
    bouncesLeft: 1,
    pierce: false,
    life: 2,
    hit: [],
    ...patch,
  };
  game.bullets.push(b);
  return b;
}

function addPowerup(game: GameState, patch: Partial<PowerUp> & { kind: PowerUpKind }): PowerUp {
  const p: PowerUp = {
    id: 2000 + game.powerups.length,
    x: 450,
    y: 350,
    despawn: 600,
    ...patch,
  };
  game.powerups.push(p);
  return p;
}

function addRect(
  game: GameState,
  kind: Obstacle["kind"],
  x: number,
  y: number,
  w: number,
  h: number,
): Obstacle {
  const o: Obstacle = { x, y, w, h, kind, hp: kind === "crate" ? CRATE_HP : 0 };
  (kind === "wall" ? game.walls : game.crates).push(o);
  return o;
}

// --- layout ---------------------------------------------------------------------

describe("layout contract", () => {
  it("pins OBS_DIM = 282 (the network input width and the replay row width)", () => {
    expect(OBS_DIM).toBe(282);
  });

  it("blocks are contiguous: 16 + 3·12 + 18·7 + 4·6 + 16·5 = 282", () => {
    // Asserting the SUMS, not restating the offsets, is what catches a stride
    // edited without moving the next block's start (or vice versa).
    expect(OFF.SELF).toBe(0);
    expect(OFF.TANKS).toBe(OFF.SELF + 16); // 16
    expect(OFF.BULLETS).toBe(OFF.TANKS + 3 * TANK_SLOT); // 16 + 36 = 52
    expect(OFF.POWERUPS).toBe(OFF.BULLETS + 18 * BULLET_SLOT); // 52 + 126 = 178
    expect(OFF.RAYS).toBe(OFF.POWERUPS + 4 * POWERUP_SLOT); // 178 + 24 = 202
    expect(OBS_DIM).toBe(OFF.RAYS + 16 * RAY_SLOT); // 202 + 80 = 282
    expect(16 + 36 + 126 + 24 + 80).toBe(OBS_DIM);
  });
});

// --- body frame -----------------------------------------------------------------

describe("body frame (own-heading reference)", () => {
  it("target dead ahead ⇒ bearing 0 and the true normalized distance", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 300, y: 350, heading: 0 });
    addTank(game, { id: 1, x: 600, y: 350 }); // 300 u along the hull axis
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.TANKS + 2]).toBeCloseTo(0, 6);
    expect(obs[OFF.TANKS + 1]).toBeCloseTo(300 / NORM_DIST, 6);
  });

  it("target dead behind ⇒ |bearing| = 1", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 300, y: 350, heading: 0 });
    addTank(game, { id: 1, x: 100, y: 350 });
    const obs = encodeObs(game, me.id);
    expect(Math.abs(obs[OFF.TANKS + 2])).toBeCloseTo(1, 6);
    expect(obs[OFF.TANKS + 1]).toBeCloseTo(200 / NORM_DIST, 6);
  });

  it("target 90° to the hull's right reads +0.5 (y-down handedness pinned)", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: 0 });
    addTank(game, { id: 1, x: 450, y: 550 }); // +y on screen = right of heading 0
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.TANKS + 2]).toBeCloseTo(0.5, 6);
  });

  it("turning the hull rotates the frame, not the numbers", () => {
    // The same 300 u dead-ahead engagement with the hull pointing +y instead of
    // +x: a broken rotation formula (wrong sign, swapped cos/sin) shows up here
    // as a bearing/dist change even though the tactics are identical.
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: Math.PI / 2 });
    addTank(game, { id: 1, x: 450, y: 650 });
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.TANKS + 2]).toBeCloseTo(0, 6);
    expect(obs[OFF.TANKS + 1]).toBeCloseTo(300 / NORM_DIST, 6);
  });
});

// --- SELF -----------------------------------------------------------------------

describe("SELF block", () => {
  it("writes every one of the 16 indices from the observer's own fields", () => {
    const game = emptyGame("allied");
    game.tick = 120;
    const me = addTank(game, {
      id: 0,
      x: 100,
      y: 100,
      speed: -120,
      hp: 50,
      reload: 0.25,
      bulletsInFlight: 3,
      kills: 1,
      deaths: 2,
      team: 0,
      timers: { rapid: 4, shield: 1.5, speed: 4, bounce: 5, pierce: 5 },
    });
    addTank(game, { id: 1, team: 0 }); // living ally
    addTank(game, { id: 2, team: 0, alive: false }); // dead ally: must NOT count
    addTank(game, { id: 3, team: 1 }); // enemy
    for (let i = 0; i < 20; i++) {
      addBullet(game, { ownerId: 1, team: 0, x: 500 + i, y: 600 });
    }
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.SELF + 0]).toBeCloseTo(-120 / 240, 6); // signed speed
    expect(obs[OFF.SELF + 1]).toBe(0); // reserved slot, documented constant
    expect(obs[OFF.SELF + 2]).toBeCloseTo(50 / TANK_HP, 6);
    expect(obs[OFF.SELF + 3]).toBeCloseTo(0.75, 6); // 1 − 0.25/1.0
    expect(obs[OFF.SELF + 4]).toBeCloseTo(3 / 6, 6);
    expect(obs[OFF.SELF + 5]).toBeCloseTo(4 / 8, 6); // rapid
    expect(obs[OFF.SELF + 6]).toBeCloseTo(1.5 / 3, 6); // shield
    expect(obs[OFF.SELF + 7]).toBeCloseTo(4 / 8, 6); // speed
    expect(obs[OFF.SELF + 8]).toBeCloseTo(5 / 10, 6); // bounce
    expect(obs[OFF.SELF + 9]).toBeCloseTo(5 / 10, 6); // pierce
    expect(obs[OFF.SELF + 10]).toBe(1); // allied mode
    expect(obs[OFF.SELF + 11]).toBeCloseTo(1 / 3, 6); // exactly one LIVING ally
    expect(obs[OFF.SELF + 12]).toBeCloseTo(1 / 3, 6);
    expect(obs[OFF.SELF + 13]).toBeCloseTo(2 / 3, 6);
    expect(obs[OFF.SELF + 14]).toBeCloseTo(2 / 24, 6); // 20 live bullets, 18 slots
    expect(obs[OFF.SELF + 15]).toBe(0);
  });

  it("reads 0 for the team flag and the ally count in ffa", () => {
    const game = emptyGame(); // ffa: unique team per tank, nobody is an ally
    const me = addTank(game, { id: 0 });
    addTank(game, { id: 1 });
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.SELF + 10]).toBe(0);
    expect(obs[OFF.SELF + 11]).toBe(0);
  });

  it("reload readiness saturates at both ends and never goes negative", () => {
    const game = emptyGame();
    const ready = addTank(game, { id: 0, reload: 0 });
    const justFired = addTank(game, { id: 1, reload: 1.0, x: 600 });
    const overshot = addTank(game, { id: 2, reload: 2.0, x: 800 }); // not engine-reachable
    expect(encodeObs(game, ready.id)[OFF.SELF + 3]).toBeCloseTo(1, 6);
    expect(encodeObs(game, justFired.id)[OFF.SELF + 3]).toBeCloseTo(0, 6);
    expect(encodeObs(game, overshot.id)[OFF.SELF + 3]).toBe(0); // clamped, not −1
  });
});

// --- TANKS ----------------------------------------------------------------------

describe("TANKS block", () => {
  it("sorts others by distance and leaves the unused slots all-zero", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350 });
    addTank(game, { id: 1, x: 700, y: 350 }); // 250 u
    addTank(game, { id: 2, x: 550, y: 350 }); // 100 u — nearest, so slot 0
    addTank(game, { id: 3, x: 850, y: 350 }); // 400 u
    const obs = encodeObs(game, me.id);
    // Compare the STORED (normalized) value: multiplying back by NORM_DIST
    // would amplify float32 rounding by ~1140× and tell us nothing new.
    const u = (s: number, worldU: number) => expect(obs[OFF.TANKS + s * TANK_SLOT + 1]).toBeCloseTo(worldU / NORM_DIST, 6);
    u(0, 100);
    u(1, 250);
    u(2, 400);
    for (let i = 0; i < TANK_SLOT; i++) {
      expect(obs[OFF.TANKS + 3 * TANK_SLOT + i]).toBe(0); // 4th slot: sentinel
    }
  });

  it("writes the full row for the nearest other tank", () => {
    const game = emptyGame("allied");
    const me = addTank(game, { id: 0, x: 100, y: 200, heading: 0, speed: 60, team: 0 });
    addTank(game, {
      id: 1,
      x: 300,
      y: 200, // dead ahead, 200 u
      heading: Math.PI,
      speed: -120,
      hp: 40,
      reload: 0.5,
      bulletsInFlight: 2,
      kills: 3,
      team: 0,
      timers: { rapid: 0, shield: 1, speed: 0, bounce: 0, pierce: 0 },
    });
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.TANKS + 0]).toBe(1); // valid
    expect(obs[OFF.TANKS + 1]).toBeCloseTo(200 / NORM_DIST, 6);
    expect(obs[OFF.TANKS + 2]).toBeCloseTo(0, 6); // bearing
    expect(Math.abs(obs[OFF.TANKS + 3])).toBeCloseTo(1, 6); // faces opposite to me
    expect(obs[OFF.TANKS + 4]).toBeCloseTo(-120 / 240, 6); // their signed speed
    expect(obs[OFF.TANKS + 5]).toBeCloseTo(40 / TANK_HP, 6);
    expect(obs[OFF.TANKS + 6]).toBe(1); // shared team in allied mode ⇒ ally
    expect(obs[OFF.TANKS + 7]).toBeCloseTo(0.5, 6); // their reload readiness
    expect(obs[OFF.TANKS + 8]).toBeCloseTo(2 / 6, 6);
    expect(obs[OFF.TANKS + 9]).toBe(1); // shield up
    expect(obs[OFF.TANKS + 11]).toBeCloseTo(3 / 3, 6); // their kills
  });

  it("closing speed is + when the gap shrinks and − when it opens, whoever moves", () => {
    // I stand, they charge head-on.
    const a = emptyGame();
    const meA = addTank(a, { id: 0, x: 450, y: 350, speed: 0 });
    addTank(a, { id: 1, x: 650, y: 350, heading: Math.PI, speed: 120 });
    expect(encodeObs(a, meA.id)[OFF.TANKS + 10]).toBeCloseTo(0.5, 6);

    // I stand, they run away.
    const b = emptyGame();
    const meB = addTank(b, { id: 0, x: 450, y: 350, speed: 0 });
    addTank(b, { id: 1, x: 650, y: 350, heading: 0, speed: 120 });
    expect(encodeObs(b, meB.id)[OFF.TANKS + 10]).toBeCloseTo(-0.5, 6);

    // They stand, I charge — must read the same as case 1, which is why the
    // feature uses the RELATIVE velocity rather than theirs alone.
    const c = emptyGame();
    const meC = addTank(c, { id: 0, x: 450, y: 350, heading: 0, speed: 120 });
    addTank(c, { id: 1, x: 650, y: 350 });
    expect(encodeObs(c, meC.id)[OFF.TANKS + 10]).toBeCloseTo(0.5, 6);
  });

  it("dead tanks occupy no slot (no chasing corpses)", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350 });
    addTank(game, { id: 1, x: 470, y: 350, alive: false }); // nearer, but dead
    addTank(game, { id: 2, x: 700, y: 350 }); // alive, 250 u
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.TANKS + 0]).toBe(1);
    expect(obs[OFF.TANKS + 1]).toBeCloseTo(250 / NORM_DIST, 6); // the live one
    expect(obs[OFF.TANKS + TANK_SLOT + 0]).toBe(0); // slot 1 stays a sentinel
  });

  it("isAlly follows teamMode and shared team, nothing else", () => {
    const allied = emptyGame("allied");
    const meA = addTank(allied, { id: 0, x: 450, y: 350, team: 0 });
    addTank(allied, { id: 1, x: 750, y: 350, team: 0 }); // ally, 300 u
    addTank(allied, { id: 2, x: 550, y: 350, team: 1 }); // enemy, 100 u → slot 0
    const oa = encodeObs(allied, meA.id);
    expect(oa[OFF.TANKS + 6]).toBe(0); // slot 0 = the enemy
    expect(oa[OFF.TANKS + TANK_SLOT + 6]).toBe(1); // slot 1 = the ally

    const ffa = emptyGame("ffa");
    const meF = addTank(ffa, { id: 0, x: 450, y: 350, team: 0 });
    addTank(ffa, { id: 1, x: 550, y: 350, team: 0 }); // same team means NOTHING in ffa
    expect(encodeObs(ffa, meF.id)[OFF.TANKS + 6]).toBe(0);
  });
});

// --- BULLETS --------------------------------------------------------------------

describe("BULLETS block", () => {
  it("sorts nearest-first and writes the full row", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350 });
    addBullet(game, { ownerId: 1, team: 1, x: 600, y: 350, dir: Math.PI, bouncesLeft: 3 });
    addBullet(game, { ownerId: 1, team: 1, x: 510, y: 350, dir: 0, bouncesLeft: 1 });
    const obs = encodeObs(game, me.id);
    // slot 0 = the 60 u bullet, flying my way
    expect(obs[OFF.BULLETS + 0]).toBe(1);
    expect(obs[OFF.BULLETS + 1]).toBeCloseTo(60 / NORM_DIST, 6);
    expect(obs[OFF.BULLETS + 2]).toBeCloseTo(0, 6); // bearing dead ahead
    expect(obs[OFF.BULLETS + 3]).toBeCloseTo(0, 6); // travelling along my heading
    expect(obs[OFF.BULLETS + 4]).toBeCloseTo(1, 6); // BULLET_SPEED normalized
    expect(obs[OFF.BULLETS + 5]).toBe(0); // enemy-owned
    expect(obs[OFF.BULLETS + 6]).toBeCloseTo(1 / BOUNCE_POWERUP, 6);
    // slot 1 = the 150 u bullet, coming back at me
    expect(obs[OFF.BULLETS + BULLET_SLOT + 1]).toBeCloseTo(150 / NORM_DIST, 6);
    expect(obs[OFF.BULLETS + BULLET_SLOT + 3]).toBeCloseTo(-1, 6); // dir π vs heading 0
    // slot 2 = sentinel
    for (let i = 0; i < BULLET_SLOT; i++) {
      expect(obs[OFF.BULLETS + 2 * BULLET_SLOT + i]).toBe(0);
    }
  });

  it("ownerCode: mine 1, ally 0.5, enemy 0", () => {
    const game = emptyGame("allied");
    const me = addTank(game, { id: 0, x: 450, y: 350, team: 0 });
    addTank(game, { id: 1, team: 0 }); // the ally (needed only as a live tank)
    addBullet(game, { ownerId: 0, team: 0, x: 460, y: 350 }); // 10 u
    addBullet(game, { ownerId: 1, team: 0, x: 470, y: 350 }); // 20 u
    addBullet(game, { ownerId: 2, team: 1, x: 480, y: 350 }); // 30 u, owner already gone
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.BULLETS + 5]).toBe(1);
    expect(obs[OFF.BULLETS + BULLET_SLOT + 5]).toBe(0.5);
    expect(obs[OFF.BULLETS + 2 * BULLET_SLOT + 5]).toBe(0);
  });

  it("keeps the 18 nearest bullets and flags the overflow in SELF", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 0, y: 0 });
    for (let i = 0; i < MAX_BULLETS_GLOBAL; i++) {
      addBullet(game, { ownerId: 1, team: 1, x: 10 + i * 10, y: 0 }); // 10..240 u
    }
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.BULLETS + 17 * BULLET_SLOT + 0]).toBe(1); // 18th nearest fits
    expect(obs[OFF.BULLETS + 18 * BULLET_SLOT + 0]).toBe(0); // the 19th is dropped
    for (let i = 0; i < BULLET_SLOT; i++) {
      expect(obs[OFF.BULLETS + 18 * BULLET_SLOT + i]).toBe(0);
    }
    expect(obs[OFF.SELF + 14]).toBeCloseTo((MAX_BULLETS_GLOBAL - 18) / MAX_BULLETS_GLOBAL, 6);
  });
});

// --- POWERUPS -------------------------------------------------------------------

describe("POWERUPS block", () => {
  it("sorts nearest-first and encodes kind, combat class and despawn pressure", () => {
    const game = emptyGame();
    game.tick = 300;
    const me = addTank(game, { id: 0, x: 0, y: 0 });
    addPowerup(game, { kind: "pierce", x: 100, y: 0, despawn: 300 + 5 * TICK_HZ });
    addPowerup(game, { kind: "shield", x: 200, y: 0, despawn: 300 + 10 * TICK_HZ });
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.POWERUPS + 0]).toBe(1);
    expect(obs[OFF.POWERUPS + 1]).toBeCloseTo(100 / NORM_DIST, 6);
    expect(obs[OFF.POWERUPS + 2]).toBeCloseTo(0, 6);
    expect(obs[OFF.POWERUPS + 3]).toBeCloseTo(5 / 5, 6); // pierce = index 5
    expect(obs[OFF.POWERUPS + 4]).toBe(1); // combat kind
    expect(obs[OFF.POWERUPS + 5]).toBeCloseTo(0.25, 6); // 5 s of 20 left
    expect(obs[OFF.POWERUPS + POWERUP_SLOT + 0]).toBe(1);
    expect(obs[OFF.POWERUPS + POWERUP_SLOT + 3]).toBeCloseTo(1 / 5, 6); // shield = 1
    expect(obs[OFF.POWERUPS + POWERUP_SLOT + 4]).toBe(0); // survivability kind
    expect(obs[OFF.POWERUPS + POWERUP_SLOT + 5]).toBeCloseTo(0.5, 6);
    for (let i = 0; i < POWERUP_SLOT; i++) {
      expect(obs[OFF.POWERUPS + 2 * POWERUP_SLOT + i]).toBe(0); // slot 2 sentinel
    }
  });

  it("maps all six kinds to the pinned indices and combat flags", () => {
    // One powerup per game, so each row's typeIdx/isCombat pair is attributable
    // to exactly one kind. The ordinal table is the contract a trained net
    // depends on — swapping shield/repair here would re-label pickups silently.
    const idx: Record<PowerUpKind, number> = {
      rapid: 0,
      shield: 1,
      speed: 2,
      repair: 3,
      bounce: 4,
      pierce: 5,
    };
    const combat: Record<PowerUpKind, number> = {
      rapid: 1,
      shield: 0,
      speed: 0,
      repair: 0,
      bounce: 1,
      pierce: 1,
    };
    for (const kind of Object.keys(idx) as PowerUpKind[]) {
      const game = emptyGame();
      const me = addTank(game, { id: 0, x: 0, y: 0 });
      addPowerup(game, { kind, x: 50, y: 0 });
      const obs = encodeObs(game, me.id);
      expect(obs[OFF.POWERUPS + 3], kind).toBeCloseTo(idx[kind] / 5, 6);
      expect(obs[OFF.POWERUPS + 4], kind).toBe(combat[kind]);
    }
  });

  it("keeps the 4 nearest when the field is overfull", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 0, y: 0 });
    for (let i = 0; i < 5; i++) addPowerup(game, { kind: "repair", x: 40 + i * 40, y: 0 });
    const obs = encodeObs(game, me.id);
    for (let s = 0; s < 4; s++) expect(obs[OFF.POWERUPS + s * POWERUP_SLOT + 0]).toBe(1);
    // The 200 u pickup must be the one dropped: only four rows exist, so the
    // farthest SURVIVING row holds 160 u. (There is deliberately no "slot 4" to
    // zero-check — index 202 is already the RAYS block, asserted in the layout
    // test, which is exactly why the block must not overflow.)
    expect(obs[OFF.POWERUPS + 3 * POWERUP_SLOT + 1]).toBeCloseTo(160 / NORM_DIST, 6);
  });
});

// --- RAYS -----------------------------------------------------------------------

describe("RAYS block", () => {
  it("sees a wall dead ahead at 100 u as dist 100/RAY_LEN + hitWall", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: 0 });
    addRect(game, "wall", 550, 340, 100, 20); // left edge 100 u ahead of the hull
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.RAYS + 0]).toBeCloseTo(100 / RAY_LEN, 6);
    expect(obs[OFF.RAYS + 1]).toBe(1); // hitWall
    expect(obs[OFF.RAYS + 2]).toBe(0);
    expect(obs[OFF.RAYS + 3]).toBe(0);
    expect(obs[OFF.RAYS + 4]).toBe(0);
  });

  it("labels crates, enemy tanks and allies; clear rays read full distance", () => {
    const game = emptyGame("allied");
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: 0, team: 0 });
    addRect(game, "wall", 550, 340, 100, 20); // k=0  (+x)      wall at 100
    addRect(game, "crate", 440, 450, 20, 20); // k=4  (+y)      crate at 100
    addTank(game, { id: 1, x: 350, y: 350, team: 1 }); // k=8 (−x) enemy at 80
    addTank(game, { id: 2, x: 450, y: 250, team: 0 }); // k=12 (−y) ally at 80
    const obs = encodeObs(game, me.id);

    // k=2, 45° off-bore, passes between every obstacle: clear space reads as
    // "visible to the end of the ray", with no hit flags and no team sign.
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 0]).toBeCloseTo(1, 6);
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 1]).toBe(0);
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 2]).toBe(0);
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 3]).toBe(0);
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 4]).toBe(0);

    expect(obs[OFF.RAYS + 4 * RAY_SLOT + 0]).toBeCloseTo(100 / RAY_LEN, 6);
    expect(obs[OFF.RAYS + 4 * RAY_SLOT + 2]).toBe(1); // hitCrate

    expect(obs[OFF.RAYS + 8 * RAY_SLOT + 0]).toBeCloseTo((100 - TANK_RADIUS) / RAY_LEN, 6);
    expect(obs[OFF.RAYS + 8 * RAY_SLOT + 3]).toBe(1); // hitTank
    expect(obs[OFF.RAYS + 8 * RAY_SLOT + 4]).toBe(-1); // enemy

    expect(obs[OFF.RAYS + 12 * RAY_SLOT + 0]).toBeCloseTo((100 - TANK_RADIUS) / RAY_LEN, 6);
    expect(obs[OFF.RAYS + 12 * RAY_SLOT + 3]).toBe(1);
    expect(obs[OFF.RAYS + 12 * RAY_SLOT + 4]).toBe(1); // ally
  });

  it("casts in FFA so every tank hit reads hostile", () => {
    const game = emptyGame(); // ffa: no allies exist
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: 0, team: 0 });
    addTank(game, { id: 1, x: 350, y: 350, team: 1 });
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.RAYS + 8 * RAY_SLOT + 3]).toBe(1);
    expect(obs[OFF.RAYS + 8 * RAY_SLOT + 4]).toBe(-1);
  });

  it("fans the rays body-relative: rotating the hull rotates the hits with it", () => {
    // The FFA wall rig turned 90°: heading π/2 points +y, so the wall moves
    // from "ahead along +x" to "ahead along +y" and k=0 must find it again —
    // the fan must never be world-anchored.
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350, heading: Math.PI / 2 });
    addRect(game, "wall", 440, 450, 20, 100);
    const obs = encodeObs(game, me.id);
    expect(obs[OFF.RAYS + 0]).toBeCloseTo(100 / RAY_LEN, 6);
    expect(obs[OFF.RAYS + 1]).toBe(1);
    expect(obs[OFF.RAYS + 2 * RAY_SLOT + 0]).toBeCloseTo(1, 6); // 45° left: clear
  });
});

// --- perspective ----------------------------------------------------------------

describe("perspective mirror", () => {
  it("two tanks agree on the gap and mirror each other's bearings", () => {
    const game = emptyGame();
    const a = addTank(game, { id: 0, x: 300, y: 350, heading: 0 });
    const b = addTank(game, { id: 1, x: 600, y: 350, heading: Math.PI }); // face-off
    const oa = encodeObs(game, a.id);
    const ob = encodeObs(game, b.id);
    // Exactly one other tank each ⇒ slot 0 on both sides is the same pair.
    expect(oa[OFF.TANKS + 0]).toBe(1);
    expect(ob[OFF.TANKS + 0]).toBe(1);
    expect(oa[OFF.TANKS + 1]).toBe(ob[OFF.TANKS + 1]); // one shared distance
    expect(oa[OFF.TANKS + 1]).toBeCloseTo(300 / NORM_DIST, 6);
    expect(oa[OFF.TANKS + 2]).toBeCloseTo(0, 6);
    expect(ob[OFF.TANKS + 2]).toBeCloseTo(0, 6);
    expect(oa[OFF.TANKS + 2] + ob[OFF.TANKS + 2]).toBeCloseTo(0, 6); // mirrored
    // And nobody encodes themselves: each side has exactly one valid row.
    expect(oa[OFF.TANKS + TANK_SLOT + 0]).toBe(0);
    expect(ob[OFF.TANKS + TANK_SLOT + 0]).toBe(0);
  });
});

// --- buffer discipline ----------------------------------------------------------

describe("buffer discipline", () => {
  it("allocates a full-length buffer when out is omitted", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0 });
    expect(encodeObs(game, me.id).length).toBe(OBS_DIM);
  });

  it("reuses out and clears rows the new world leaves empty", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0, x: 450, y: 350 });
    addTank(game, { id: 1, x: 650, y: 350 }); // one other ⇒ two empty TANKS slots
    const out = new Float32Array(OBS_DIM).fill(0.75); // stale data everywhere
    const obs = encodeObs(game, me.id, out);
    expect(obs).toBe(out); // same buffer, no reallocation
    for (let i = 0; i < TANK_SLOT; i++) {
      expect(obs[OFF.TANKS + 2 * TANK_SLOT + i]).toBe(0); // stale 0.75 must be gone
    }
    expect(obs[OFF.TANKS + 0]).toBe(1); // ...while the real row is intact
  });

  it("rejects a short out buffer instead of writing silently past its length", () => {
    const game = emptyGame();
    const me = addTank(game, { id: 0 });
    expect(() => encodeObs(game, me.id, new Float32Array(16))).toThrow(RangeError);
  });

  it("rejects a tank id that is not in the state", () => {
    const game = emptyGame();
    addTank(game, { id: 0 });
    expect(() => encodeObs(game, 99)).toThrow(RangeError);
  });
});

// --- randomized property --------------------------------------------------------

// The six kinds, in the observation's index order — the random builder picks
// from this so every kind reaches the encoder across the run.
const KINDS: PowerUpKind[] = ["rapid", "shield", "speed", "repair", "bounce", "pierce"];

/**
 * A random but structurally valid GameState: entities inside the arena, timers
 * inside their durations, bullet ownership drawn from the live tanks. Values
 * are deliberately NOT engine-reachable in every detail (stacked tanks, an
 * overfull powerup field) — the encoder is a total function over the state
 * type, and totality is exactly what this property test is for.
 */
function randomGame(rng: () => number): GameState {
  const game = emptyGame(rng() < 0.5 ? "ffa" : "allied");
  game.tick = Math.floor(rng() * 500);
  const tankCount = 1 + Math.floor(rng() * 4);
  for (let i = 0; i < tankCount; i++) {
    addTank(game, {
      id: i,
      x: 40 + rng() * (ARENA_W - 80),
      y: 40 + rng() * (ARENA_H - 80),
      heading: rng() * 2 * Math.PI,
      speed: (rng() * 2 - 1) * 240,
      hp: rng() * TANK_HP,
      alive: rng() > 0.2,
      reload: rng(),
      bulletsInFlight: Math.floor(rng() * 7),
      kills: Math.floor(rng() * 4),
      deaths: Math.floor(rng() * 4),
      team: game.teamMode === "allied" ? (rng() < 0.5 ? 0 : 1) : i,
      timers: {
        rapid: rng() * 8,
        shield: rng() * 3,
        speed: rng() * 8,
        bounce: rng() * 10,
        pierce: rng() * 10,
      },
    });
  }
  const rectCount = 2 + Math.floor(rng() * 6);
  for (let i = 0; i < rectCount; i++) {
    const w = 20 + rng() * 100;
    const h = 20 + rng() * 100;
    addRect(
      game,
      rng() < 0.5 ? "wall" : "crate",
      rng() * (ARENA_W - w),
      rng() * (ARENA_H - h),
      w,
      h,
    );
  }
  const bulletCount = Math.floor(rng() * (MAX_BULLETS_GLOBAL + 1));
  for (let i = 0; i < bulletCount; i++) {
    const owner = game.tanks[Math.floor(rng() * game.tanks.length)];
    addBullet(game, {
      ownerId: owner.id,
      team: owner.team,
      x: rng() * ARENA_W,
      y: rng() * ARENA_H,
      dir: rng() * 2 * Math.PI,
      bouncesLeft: Math.floor(rng() * (BOUNCE_POWERUP + 1)),
    });
  }
  const puCount = Math.floor(rng() * 6); // overfull on purpose: cap must hold
  for (let i = 0; i < puCount; i++) {
    addPowerup(game, {
      kind: KINDS[Math.floor(rng() * KINDS.length)],
      x: rng() * ARENA_W,
      y: rng() * ARENA_H,
      despawn: game.tick + Math.floor(rng() * POWERUP_LIFETIME * TICK_HZ),
    });
  }
  return game;
}

/** Sentinel invariant over a whole vector: any row whose valid flag is 0 must
 *  be zero THROUGHOUT — the one thing that lets a net (and a human debugging
 *  one) tell "no such entity" apart from "an entity at bearing 0, range 0". */
function expectSentinelRowsAllZero(obs: Float32Array, where: string): void {
  const blocks: [base: number, rows: number, stride: number][] = [
    [OFF.TANKS, 3, TANK_SLOT],
    [OFF.BULLETS, 18, BULLET_SLOT],
    [OFF.POWERUPS, 4, POWERUP_SLOT],
  ];
  for (const [base, rows, stride] of blocks) {
    for (let s = 0; s < rows; s++) {
      if (obs[base + s * stride] !== 0) continue; // a live row may hold zeros
      for (let i = 0; i < stride; i++) {
        if (obs[base + s * stride + i] !== 0) {
          throw new Error(`${where}: sentinel row ${s} of block @${base} has [${i}] ≠ 0`);
        }
      }
    }
  }
}

describe("10 000 seeded random states", () => {
  it("never emits NaN, never leaves [−1, 1], and keeps sentinels pure", () => {
    const rng = mulberry32(0x7e2519a7);
    for (let iter = 0; iter < 10_000; iter++) {
      const game = randomGame(rng);
      for (const tank of game.tanks) {
        const obs = encodeObs(game, tank.id);
        for (let i = 0; i < OBS_DIM; i++) {
          const v = obs[i];
          // 1.0001 (not 1.0) leaves room for exact-±1 bearings from the wrap;
          // the finite check is the NaN guard this suite exists to enforce.
          if (!Number.isFinite(v) || v < -1.0001 || v > 1.0001) {
            throw new Error(`iter ${iter} tick ${game.tick} tank ${tank.id}: obs[${i}] = ${v}`);
          }
        }
        expectSentinelRowsAllZero(obs, `iter ${iter} tank ${tank.id}`);
      }
    }
  });
});
