// actions.spec.ts — the 18-action contract: packing arithmetic, keymap, and
// total-for-any-input clamping. Every other module (Q heads, masks, play-mode
// keymap, τ-sampling) codes against this table, so the anchors here are the
// SPEC-level pins, not implementation details.
import { describe, expect, it } from "vitest";
import {
  ACTION_COUNT,
  actionFromKeys,
  decode,
  encodeAction,
  type Action,
} from "../../src/game/actions";

/** Keys-shaped helper so test rows read like the control scheme. */
function keys(
  up: boolean,
  down: boolean,
  left: boolean,
  right: boolean,
  fire: boolean,
): { up: boolean; down: boolean; left: boolean; right: boolean; fire: boolean } {
  return { up, down, left, right, fire };
}

describe("action space shape", () => {
  it("pins ACTION_COUNT = 18 (Q-head size and fire-mask width)", () => {
    expect(ACTION_COUNT).toBe(18);
  });

  it("decode covers exactly indices 0..17 with the plan's grid anchors", () => {
    // Anchor rows straight out of the plan's grid — these pin the bit order
    // (a = (throttle*3 + steer)*2 + fire, steer index order left/straight/
    // right), not just a bijection.
    const anchors: [number, Action][] = [
      [0, { throttle: 0, steer: 0, fire: false }], // coast left no
      [1, { throttle: 0, steer: 0, fire: true }], // coast left fire
      [2, { throttle: 0, steer: 1, fire: false }], // coast straight no
      [5, { throttle: 0, steer: 2, fire: true }], // coast right fire
      [6, { throttle: 1, steer: 0, fire: false }], // fwd left no
      [8, { throttle: 1, steer: 1, fire: false }], // fwd straight no
      [11, { throttle: 1, steer: 2, fire: true }], // fwd right fire
      [12, { throttle: 2, steer: 0, fire: false }], // rev left no
      [15, { throttle: 2, steer: 1, fire: true }], // rev straight fire
      [17, { throttle: 2, steer: 2, fire: true }], // rev right fire
    ];
    for (const [a, want] of anchors) expect(decode(a)).toEqual(want);
  });

  it("encode is structured: fire is the fast axis (+1), steer +2, throttle +6", () => {
    // The stride structure is what lets a fire mask walk the table in stride-2
    // steps; a permutation-correct but stride-broken packing would break it.
    for (const t of [0, 1, 2]) {
      for (const s of [0, 1, 2]) {
        const no = encodeAction(t, s, false);
        expect(encodeAction(t, s, true)).toBe(no + 1);
        if (s < 2) expect(encodeAction(t, s + 1, false)).toBe(no + 2);
        if (t < 2) expect(encodeAction(t + 1, s, false)).toBe(no + 6);
      }
    }
  });
});

describe("decode/encode round-trip", () => {
  it("round-trips all 18 actions in both directions", () => {
    for (let a = 0; a < ACTION_COUNT; a++) {
      const { throttle, steer, fire } = decode(a);
      expect(encodeAction(throttle, steer, fire)).toBe(a);
    }
    for (const t of [0, 1, 2]) {
      for (const s of [0, 1, 2]) {
        for (const fire of [false, true]) {
          const a = encodeAction(t, s, fire);
          expect(a).toBeGreaterThanOrEqual(0);
          expect(a).toBeLessThan(ACTION_COUNT);
          expect(decode(a)).toEqual({ throttle: t, steer: s, fire } as Action);
        }
      }
    }
  });

  it("enumerates 18 distinct indices — no two rows collide", () => {
    const seen = new Set<number>();
    for (const t of [0, 1, 2])
      for (const s of [0, 1, 2])
        for (const fire of [false, true]) seen.add(encodeAction(t, s, fire));
    expect(seen.size).toBe(ACTION_COUNT);
  });
});

describe("decode clamping (total for any numeric input)", () => {
  it("clamps out-of-range indices into the table instead of aliasing", () => {
    expect(decode(-1)).toEqual(decode(0));
    expect(decode(-999)).toEqual(decode(0));
    expect(decode(ACTION_COUNT)).toEqual(decode(ACTION_COUNT - 1));
    expect(decode(9999)).toEqual(decode(ACTION_COUNT - 1));
  });

  it("clamps fractional values by truncation toward the table", () => {
    expect(decode(17.9)).toEqual(decode(17));
    expect(decode(6.2)).toEqual(decode(6));
    expect(decode(-0.9)).toEqual(decode(0));
  });

  it("non-finite input collapses to action 0 (the table's first row)", () => {
    // NaN = "no signal" → action 0 = coast/left/hold. +∞ clamps to the top of
    // the table, −∞ to the bottom: the sign is still meaningful.
    expect(decode(Number.NaN)).toEqual(decode(0));
    expect(decode(Number.NEGATIVE_INFINITY)).toEqual(decode(0));
    expect(decode(Number.POSITIVE_INFINITY)).toEqual(decode(ACTION_COUNT - 1));
  });
});

describe("encodeAction clamping", () => {
  it("clamps component values to 0..2 (same boundary safety as decode)", () => {
    expect(encodeAction(-1, 5, false)).toBe(encodeAction(0, 2, false));
    expect(encodeAction(3, -2, true)).toBe(encodeAction(2, 0, true));
    expect(encodeAction(99, 99, false)).toBe(encodeAction(2, 2, false));
  });

  it("non-finite components collapse so NaN/NaN lands on action 0, like decode(NaN)", () => {
    // Both directions agree on the sentinel: non-finite anywhere → component 0.
    expect(encodeAction(Number.NaN, Number.NaN, false)).toBe(0);
    expect(encodeAction(Number.NEGATIVE_INFINITY, Number.NaN, false)).toBe(0);
    expect(decode(encodeAction(Number.NaN, Number.NaN, false))).toEqual(decode(0));
    // A single non-finite component collapses to 0 independently (component
    // clamping is by-value — only the INDEX path in decode keeps the sign).
    expect(encodeAction(0, Number.POSITIVE_INFINITY, false)).toBe(encodeAction(0, 0, false));
    expect(encodeAction(Number.POSITIVE_INFINITY, 1, true)).toBe(encodeAction(0, 1, true));
  });
});

describe("actionFromKeys (human keymap through the shared table)", () => {
  const throttleRows: [boolean, boolean, 0 | 1 | 2][] = [
    // [up, down, expected throttle] — W=forward(1), S=reverse(2), none=coast(0)
    [true, false, 1],
    [false, true, 2],
    [false, false, 0],
  ];
  const steerRows: [boolean, boolean, 0 | 1 | 2][] = [
    // [left, right, expected steer] — A=left(0), D=right(2), none=straight(1)
    [true, false, 0],
    [false, true, 2],
    [false, false, 1],
  ];

  it("maps all 3×3×2 key combinations onto decoded table rows", () => {
    for (const [up, down, wantThrottle] of throttleRows) {
      for (const [left, right, wantSteer] of steerRows) {
        for (const fire of [false, true]) {
          const a = actionFromKeys(keys(up, down, left, right, fire));
          expect(a).toBeGreaterThanOrEqual(0);
          expect(a).toBeLessThan(ACTION_COUNT);
          expect(decode(a)).toEqual({
            throttle: wantThrottle,
            steer: wantSteer,
            fire,
          } as Action);
        }
      }
    }
  });

  it("pins the plain-language anchors (W, S, A, D, Space)", () => {
    expect(actionFromKeys(keys(true, false, false, false, false))).toBe(8); // W  = fwd straight
    expect(actionFromKeys(keys(false, true, false, false, false))).toBe(14); // S = rev straight
    expect(actionFromKeys(keys(false, false, true, false, false))).toBe(0); // A  = coast left
    expect(actionFromKeys(keys(false, false, false, true, false))).toBe(4); // D  = coast right
    expect(actionFromKeys(keys(false, false, false, false, true))).toBe(3); // ␣ = coast straight fire
    expect(actionFromKeys(keys(false, false, false, false, false))).toBe(2); // — = coast straight
    expect(actionFromKeys(keys(true, false, false, false, true))).toBe(9); // W+␣
    expect(actionFromKeys(keys(true, false, true, false, false))).toBe(6); // W+A
  });

  it("both keys of a pair held: forward beats reverse, left beats right", () => {
    // Undefined by the plan, so it is pinned HERE and deterministic — the
    // keymap must never emit an index outside the table for a mashed keyboard.
    expect(decode(actionFromKeys(keys(true, true, false, false, false))).throttle).toBe(1);
    expect(decode(actionFromKeys(keys(false, false, true, true, false))).steer).toBe(0);
    // Full keyboard mash is still a legal action.
    const mashed = actionFromKeys(keys(true, true, true, true, true));
    expect(mashed).toBeGreaterThanOrEqual(0);
    expect(mashed).toBeLessThan(ACTION_COUNT);
  });
});
