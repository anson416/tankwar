// rewards.spec.ts — the reward contract: every term's exact magnitude per the
// plan table, death charged exactly once, the timeout placement split, the
// potential functions' ranges, and the shaping property that makes the
// potential terms legal at all (shaped − unshaped is exactly the potential
// difference, and telescopes over a path). Event rewards are computed on
// hand-built RewardViews so a failure names the broken TERM, not a fight.
import { describe, expect, it } from "vitest";
import { mulberry32 } from "./util/rng";
import {
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
  REWARD_PICKUP,
  REWARD_PICKUP_HEAL_BONUS,
  REWARD_PLACEMENT_MAX,
  REWARD_SURVIVAL,
  REWARD_SURVIVAL_SD_MULT,
  REWARD_WALL_CONTACT,
  TANK_HP,
} from "../../src/game/constants";
import { defaultConfig } from "../../src/train-protocol";
import { createGame, isOver, step } from "../../src/game/engine";
import {
  captureRewardView,
  placementBonus,
  shapedStep,
  stepRewards,
  potentialApproach,
  potentialDanger,
} from "../../src/game/rewards";
import type { RewardBullet, RewardTank, RewardView } from "../../src/game/rewards";

/** A default (alive, healthy, FFA-team = id) tank row; `over` overrides. */
function row(id: number, x: number, y: number, over: Partial<RewardTank> = {}): RewardTank {
  return {
    id,
    x,
    y,
    hp: TANK_HP,
    alive: true,
    team: id,
    diedThisTick: false,
    wallContact: false,
    ...over,
  };
}

/** A minimal reward view: FFA, phase 0, no bullets, no events unless said. */
function view(tanks: RewardTank[], over: Partial<RewardView> = {}): RewardView {
  return { tanks, bullets: [], events: [], teamMode: "ffa", phase: 0, ...over };
}

/** A hostile bullet row (owner on team 1 by default). */
function shot(x: number, y: number, team = 1): RewardBullet {
  return { x, y, ownerId: 100 + team, team };
}

// --- literal drift anchors ----------------------------------------------------
//
// Every relative assertion below restates an imported constant, so a tuning
// edit to constants.ts would pass this whole file while silently breaking the
// plan's binding reward table (kill +50, death −60, "Kill ≈ +250, death
// ≈ −210 — shaping never dominates outcomes"). These LITERALS are the pin
// that turns a magnitude drift into a red test — the same trick
// encoding.spec plays on OBS_DIM = 282.
describe("literal magnitudes (plan-table drift anchors)", () => {
  it("pins every outcome/shaping term's exact plan value", () => {
    expect(REWARD_KILL).toBe(50);
    expect(REWARD_DEATH).toBe(-60);
    expect(REWARD_DAMAGE_DEALT).toBe(2);
    expect(REWARD_DAMAGE_TAKEN).toBe(-1.5);
    expect(REWARD_SURVIVAL).toBe(0.02);
    expect(REWARD_WALL_CONTACT).toBe(-0.05);
    expect(REWARD_BULLET_EXPIRED).toBe(-0.3);
    expect(REWARD_ALLIED_FIRE).toBe(-3);
    expect(REWARD_CRATE_BREAK).toBe(5);
    expect(REWARD_PICKUP).toBe(8);
    expect(REWARD_PICKUP_HEAL_BONUS).toBe(4);
  });

  it("pins the learner's default γ to the shaping discount (policy-neutrality coupling)", () => {
    // shapedStep must build γΦ(s′)−Φ(s) with the SAME γ the value function
    // bootstraps with (NetConfig.gamma): any other value silently voids the
    // Ng et al. (1999) policy-neutrality guarantee. GAMMA below (0.97) pins
    // rewards.ts's shaping default; THIS pin makes the learner's default the
    // same number, so the two can only drift apart deliberately.
    expect(defaultConfig().gamma).toBe(0.97);
  });
});

describe("event reward magnitudes", () => {
  it("pays DAMAGE_DEALT per HP to the nearest enemy and DAMAGE_TAKEN per HP to the victim", () => {
    const prev = view([row(0, 0, 0), row(1, 50, 0)]);
    const cur = view([row(0, 0, 0, { hp: TANK_HP - 25 }), row(1, 50, 0)]);
    const out = stepRewards(prev, cur);
    // victim: −1.5·25 damage taken; shooter: +2·25 dealt; both bank survival.
    expect(out[0]).toBeCloseTo(REWARD_DAMAGE_TAKEN * 25 + REWARD_SURVIVAL, 9);
    expect(out[1]).toBeCloseTo(REWARD_DAMAGE_DEALT * 25 + REWARD_SURVIVAL, 9);
  });

  it("credits the NEAREST living enemy — a farther tank gets survival only", () => {
    const prev = view([row(0, 0, 0), row(1, 30, 0), row(2, 500, 0)]);
    const cur = view([row(0, 0, 0, { hp: TANK_HP - 25 }), row(1, 30, 0), row(2, 500, 0)]);
    const out = stepRewards(prev, cur);
    expect(out[1]).toBeCloseTo(REWARD_DAMAGE_DEALT * 25 + REWARD_SURVIVAL, 9);
    expect(out[2]).toBeCloseTo(REWARD_SURVIVAL, 9);
  });

  it("charges DAMAGE_TAKEN on drain HP too, but never deals credit for it", () => {
    // Phase 2: the arena drains 3 HP/s = 0.1 HP/tick; nobody fired.
    const prev = view([row(0, 0, 0), row(1, 500, 0)], { phase: 2 });
    const drained = 3 * (1 / 30);
    const cur = view(
      [row(0, 0, 0, { hp: TANK_HP - drained }), row(1, 500, 0, { hp: TANK_HP - drained })],
      { phase: 2 },
    );
    const out = stepRewards(prev, cur);
    for (const r of out) {
      expect(r).toBeCloseTo(REWARD_DAMAGE_TAKEN * drained + REWARD_SURVIVAL * REWARD_SURVIVAL_SD_MULT, 9);
    }
  });

  it("scales survival by phase and skips tanks already dead at tick start", () => {
    const prev = view([row(0, 0, 0), row(1, 500, 0, { alive: false })], { phase: 2 });
    const cur = view([row(0, 0, 0), row(1, 500, 0, { alive: false })], { phase: 2 });
    const out = stepRewards(prev, cur);
    expect(out[0]).toBe(REWARD_SURVIVAL * REWARD_SURVIVAL_SD_MULT);
    expect(out[1]).toBe(0); // a corpse banks nothing
  });

  it("charges WALL_CONTACT while scraping and BULLET_EXPIRED on the owner's miss only", () => {
    const prev = view([row(0, 0, 0, { wallContact: true }), row(1, 500, 500)]);
    const cur = view(
      [row(0, 0, 0, { wallContact: true }), row(1, 500, 500)],
      { events: [{ kind: "explode", x: 10, y: 10, tank: 1 }, { kind: "explode", x: 20, y: 20 }] },
    );
    const out = stepRewards(prev, cur);
    expect(out[0]).toBe(REWARD_WALL_CONTACT + REWARD_SURVIVAL);
    expect(out[1]).toBe(REWARD_BULLET_EXPIRED + REWARD_SURVIVAL);
  });

  it("pays CRATE_BREAK to the tank nearest the broken crate", () => {
    const prev = view([row(0, 80, 0), row(1, 400, 0)]);
    const cur = view([row(0, 80, 0), row(1, 400, 0)], {
      events: [{ kind: "crate", x: 100, y: 0 }],
    });
    const out = stepRewards(prev, cur);
    expect(out[0]).toBe(REWARD_CRATE_BREAK + REWARD_SURVIVAL);
    expect(out[1]).toBe(REWARD_SURVIVAL);
  });
});

describe("death and kill attribution", () => {
  it("charges DEATH exactly once — the flag tick, never the ticks after", () => {
    const hit = view([row(0, 0, 0), row(1, 50, 0)]);
    const died = view([row(0, 0, 0, { hp: 0, alive: false, diedThisTick: true }), row(1, 50, 0)]);
    const out = stepRewards(hit, died);
    expect(out[0]).toBeCloseTo(REWARD_DEATH + REWARD_DAMAGE_TAKEN * TANK_HP + REWARD_SURVIVAL, 9);
    // the corpse's next tick: no death, no survival, nothing at all
    const after = stepRewards(died, view([row(0, 0, 0, { hp: 0, alive: false }), row(1, 50, 0)]));
    expect(after[0]).toBe(0);
  });

  it("pays KILL to the victim's nearest enemy", () => {
    // the killing blow takes the victim from full to 0 in one tick, so the
    // killer banks the damage credit AND the kill on top of it
    const hit = view([row(0, 0, 0), row(1, 30, 0), row(2, 500, 0)]);
    const died = view([
      row(0, 0, 0, { hp: 0, alive: false, diedThisTick: true }),
      row(1, 30, 0),
      row(2, 500, 0),
    ]);
    const out = stepRewards(hit, died);
    expect(out[1]).toBeCloseTo(
      REWARD_DAMAGE_DEALT * TANK_HP + REWARD_KILL + REWARD_SURVIVAL,
      9,
    );
    expect(out[2]).toBe(REWARD_SURVIVAL);
  });
});

describe("allied fire", () => {
  it("charges the nearest TEAMMATE for allied damage in allied mode only", () => {
    // allied seats: victim + teammate on team 0, the shooter on team 1
    const alliedPrev = view(
      [row(0, 0, 0, { team: 0 }), row(1, 30, 0, { team: 0 }), row(2, 60, 0, { team: 1 })],
      { teamMode: "allied" },
    );
    const alliedCur = view(
      [
        row(0, 0, 0, { team: 0, hp: TANK_HP - 25 }),
        row(1, 30, 0, { team: 0 }),
        row(2, 60, 0, { team: 1 }),
      ],
      { teamMode: "allied" },
    );
    // nearest enemy (tank 2) deals credit, nearest teammate (tank 1) eats the
    // allied-fire tax for the blast that hurt their own side.
    const allied = stepRewards(alliedPrev, alliedCur);
    expect(allied[2]).toBeCloseTo(REWARD_DAMAGE_DEALT * 25 + REWARD_SURVIVAL, 9);
    expect(allied[1]).toBeCloseTo(REWARD_ALLIED_FIRE * 25 + REWARD_SURVIVAL, 9);
    expect(allied[0]).toBeCloseTo(REWARD_DAMAGE_TAKEN * 25 + REWARD_SURVIVAL, 9);
    // the same geometry under FFA (unique team per tank) has no teammates, so
    // the nearest tank simply takes the damage credit and no tax exists
    const ffa = stepRewards(
      view([row(0, 0, 0), row(1, 30, 0), row(2, 60, 0)]),
      view([row(0, 0, 0, { hp: TANK_HP - 25 }), row(1, 30, 0), row(2, 60, 0)]),
    );
    expect(ffa[1]).toBeCloseTo(REWARD_DAMAGE_DEALT * 25 + REWARD_SURVIVAL, 9);
    expect(ffa[0]).toBeCloseTo(REWARD_DAMAGE_TAKEN * 25 + REWARD_SURVIVAL, 9);
  });
});

describe("timeout placement bonus", () => {
  it("ranks alive tanks by hp: full PLACEMENT_MAX for 1st of 2, 0 for last", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 1, seed: 11 });
    g.tanks[0].hp = 100;
    g.tanks[1].hp = 50;
    expect(placementBonus(g)).toEqual([REWARD_PLACEMENT_MAX, 0]);
  });

  it("splits the bonus linearly for 4 and ranks the dead after every living tank", () => {
    const g = createGame({ tankCount: 4, teamMode: "ffa", episodeSeconds: 1, seed: 12 });
    g.tanks[0].hp = 5;
    g.tanks[1].hp = 100;
    g.tanks[2].hp = 20;
    g.tanks[3].hp = 50;
    expect(placementBonus(g)).toEqual([
      0, // rank 3 of 4 (hp 5)
      REWARD_PLACEMENT_MAX, // rank 0 (hp 100)
      (REWARD_PLACEMENT_MAX * 1) / 3, // rank 2 (hp 20)
      (REWARD_PLACEMENT_MAX * 2) / 3, // rank 1 (hp 50)
    ]);
    // a corpse ranks after every survivor even with the highest hp
    g.tanks[1].alive = false;
    expect(placementBonus(g)).toEqual([
      (REWARD_PLACEMENT_MAX * 1) / 3, // hp 5 → rank 2 overall
      0, // dead → rank 3 despite hp 100
      (REWARD_PLACEMENT_MAX * 2) / 3, // hp 20 → rank 1 of the living
      REWARD_PLACEMENT_MAX, // hp 50 → rank 0 of the living
    ]);
  });

  it("is paid by the engine on a timeout: leader banks exactly PLACEMENT_MAX more", () => {
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 1, seed: 13 });
    g.tanks[0].hp = 100;
    g.tanks[1].hp = 50;
    while (!isOver(g)) step(g, [2, 2]); // nobody moves, nobody shoots
    expect(g.over).toBe(true);
    expect(g.tick).toBe(30);
    // The two tanks are symmetric on every other final-tick term (same
    // survival ×4 in sudden death, same 0.1 HP drain taken at −1.5/HP, same
    // Ψ distance apart, no bullets) — so the placement split is exactly the
    // gap between their final rewards. The absolute values sit slightly under
    // bonus + survival because the drain's damage-taken (−0.15) out-weighs it.
    expect(g.lastRewards[0] - g.lastRewards[1]).toBeCloseTo(REWARD_PLACEMENT_MAX, 6);
    expect(g.lastRewards[0]).toBeGreaterThan(39);
    expect(g.lastRewards[1]).toBeLessThan(REWARD_SURVIVAL);
  });
});

describe("potentials Φ and Ψ", () => {
  it("Φ is PHI_DANGER_WEIGHT at contact, 0 at the radius, and ignores ally shells", () => {
    const me = view([row(0, 0, 0)]);
    expect(potentialDanger(me, 0)).toBe(0); // no bullets, no danger
    const grazing = view([row(0, 0, 0)], { bullets: [shot(PHI_DANGER_RADIUS, 0)] });
    expect(potentialDanger(grazing, 0)).toBeCloseTo(0, 12); // exactly at the radius
    const onTop = view([row(0, 0, 0)], { bullets: [shot(0, 0)] });
    expect(potentialDanger(onTop, 0)).toBe(PHI_DANGER_WEIGHT);
    expect(potentialDanger(onTop, 0)).toBeLessThanOrEqual(0);
    const own = view([row(0, 0, 0, { team: 3 })], { bullets: [shot(0, 0, 3)] });
    expect(potentialDanger(own, 0)).toBe(0); // your own shell is not a threat
    const corpse = view([row(0, 0, 0, { alive: false })], { bullets: [shot(0, 0)] });
    expect(potentialDanger(corpse, 0)).toBe(0);
  });

  it("Ψ is PSI_APPROACH_WEIGHT at contact, 0 at the diagonal, and 0 with no enemy", () => {
    const alone = view([row(0, 0, 0), row(1, 10, 0, { alive: false })]);
    expect(potentialApproach(alone, 0)).toBe(0); // only a corpse left to close
    const contact = view([row(0, 0, 0), row(1, 0, 0)]);
    expect(potentialApproach(contact, 0)).toBe(PSI_APPROACH_WEIGHT);
    const far = view([row(0, 0, 0), row(1, NORM_DIST, 0)]);
    expect(potentialApproach(far, 0)).toBeCloseTo(0, 12);
    const allies = view([row(0, 0, 0, { team: 5 }), row(1, 5, 5, { team: 5 })]);
    expect(potentialApproach(allies, 0)).toBe(0); // allies are not targets
    const corpse = view([row(0, 0, 0, { alive: false }), row(1, 3, 3)]);
    expect(potentialApproach(corpse, 0)).toBe(0);
  });

  it("stays inside [PHI weight, 0] × [0, PSI weight] for arbitrary in-arena geometry", () => {
    for (let d = 0; d < 1000; d += 37) {
      // sample INSIDE the arena: two in-arena points are never more than
      // NORM_DIST apart, which is the precondition for Ψ ≥ 0
      const v = view(
        [row(0, d % 860 + 20, d % 660 + 20), row(1, (d * 7) % 860 + 20, (d * 13) % 660 + 20)],
        { bullets: [shot((d * 3) % 860 + 20, (d * 5) % 660 + 20)] },
      );
      const phi = potentialDanger(v, 0);
      const psi = potentialApproach(v, 0);
      expect(phi).toBeLessThanOrEqual(0);
      expect(phi).toBeGreaterThanOrEqual(PHI_DANGER_WEIGHT);
      expect(psi).toBeGreaterThanOrEqual(-1e-12); // float slop at exact corners
      expect(psi).toBeLessThanOrEqual(PSI_APPROACH_WEIGHT);
    }
  });
});

describe("potential-based shaping", () => {
  // Must equal REWARD_GAMMA in rewards.ts (the shapedStep default, and the
  // learner default pinned against defaultConfig().gamma above) — this
  // constant is the pin that catches drift in either direction.
  const GAMMA = 0.97;

  /** Combined potential P = Φ + Ψ of tank 0 on a view. */
  function pot(v: RewardView): number {
    return potentialDanger(v, 0) + potentialApproach(v, 0);
  }

  it("shaped − unshaped is exactly the potential difference, tick by tick", () => {
    const g = createGame({ tankCount: 3, teamMode: "ffa", episodeSeconds: 4, seed: 0x5eed });
    const roll = mulberry32(0xa07e5);
    let before = captureRewardView(g);
    for (let t = 0; t < 100 && !isOver(g); t++) {
      const actions = [0, 1, 2].map(() => (roll() * 18) | 0);
      step(g, actions);
      const after = captureRewardView(g);
      const unshaped = stepRewards(before, after);
      const shaped = shapedStep(before, after);
      for (let i = 0; i < shaped.length; i++) {
        const id = after.tanks[i].id;
        const want =
          GAMMA * (potentialDanger(after, id) + potentialApproach(after, id)) -
          (potentialDanger(before, id) + potentialApproach(before, id));
        expect(shaped[i] - unshaped[i]).toBeCloseTo(want, 9);
        // the engine's own lastRewards may add pickup/placement on top —
        // but never anything else
        expect(Math.abs(g.lastRewards[i] - shaped[i])).toBeLessThanOrEqual(8 + 4 + 40 + 1e-9);
      }
      before = after;
    }
  });

  it("telescopes: the summed shaping equals γP_end − P_0 + (γ−1)·Σ interior P", () => {
    // Σ_{i=1..n} (γP_i − P_{i−1}) = γP_n − P_0 + (γ−1)·Σ_{i=1..n−1} P_i —
    // the exact algebra of the shaping terms over a path (with γ < 1 the
    // interior potentials do not cancel, they shrink by (γ−1); both sides are
    // computed from the tick stream, so this pins shapedStep's CONTRACT, not
    // its arithmetic).
    const g = createGame({ tankCount: 2, teamMode: "ffa", episodeSeconds: 6, seed: 0xd00d });
    const roll = mulberry32(0x7e1e);
    const seq: number[] = [pot(captureRewardView(g))]; // P_0
    let sum = 0;
    for (let t = 0; t < 150 && !isOver(g); t++) {
      const before = captureRewardView(g);
      step(g, [0, 1, 2, 3].map(() => (roll() * 18) | 0));
      const after = captureRewardView(g);
      sum += shapedStep(before, after)[0] - stepRewards(before, after)[0];
      seq.push(pot(after));
    }
    const interior = seq.slice(1, -1).reduce((a, b) => a + b, 0);
    const want = GAMMA * seq[seq.length - 1] - seq[0] + (GAMMA - 1) * interior;
    expect(sum).toBeCloseTo(want, 9);
  });
});
