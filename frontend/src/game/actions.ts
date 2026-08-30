// actions.ts — the 18-action space that drives the continuous physics.
// DOM-free, tfjs-free: the network head, the keymap, and the tests share this
// one table so a human key press and an argmax index mean exactly the same thing.
//
// WHY 18 discrete actions over a continuous/multi-binary head: DQN needs a
// small enumerable head for the target max, and the plan pins ONE shared head
// for every agent and both architectures (MLP + attention). Discretizing only
// the INPUTS (throttle/steer are still integrated as continuous values by the
// physics) keeps 30 Hz control smooth with far fewer decisions than per-tick
// key emulation would need.

// Encoding: a = (throttle*3 + steer)*2 + (fire?1:0) — `fire` is the fastest
// axis so a behaviour-policy fire mask can be applied by walking the table with
// stride 2, and `throttle` is the slowest so grouping actions by "how I drive"
// groups indices by 6.
//
//   throttle: 0 = coast (no input), 1 = forward, 2 = reverse
//   steer:    0 = left (CCW, heading−), 1 = straight, 2 = right (CW, heading+)
//   fire:     0 = hold, 1 = fire (masked in the behaviour policy ONLY when
//             unavailable — never in the target max, reload is observable)
//
// The full grid (throttle {coast,fwd,rev} × steer {left,straight,right} ×
// fire {no,yes}), exactly as in the plan — the steer axis is listed in index
// order (left=0, straight=1, right=2), so the table reads:
//
//    a  thr    steer     fire      a  thr    steer     fire
//   ---  ----  --------  ----     ---  ----  --------  ----
//    0  coast left      no        9  fwd   straight  yes
//    1  coast left      yes      10  fwd   right     no
//    2  coast straight  no       11  fwd   right     yes
//    3  coast straight  yes      12  rev   left      no
//    4  coast right     no       13  rev   left      yes
//    5  coast right     yes      14  rev   straight  no
//    6  fwd   left      no       15  rev   straight  yes
//    7  fwd   left      yes      16  rev   right     no
//    8  fwd   straight  no       17  rev   right     yes

/** Number of discrete actions — the size of every Q head and of the fire mask. */
export const ACTION_COUNT = 18;

/** A decoded action. The literal unions make it impossible to represent an
 *  out-of-table action after decode — clamping happens at the boundary. */
export interface Action {
  throttle: 0 | 1 | 2;
  steer: 0 | 1 | 2;
  fire: boolean;
}

/**
 * Decode an action index into its components. Total for ANY numeric input:
 * out-of-range values clamp into the table, non-finite (NaN/±∞) collapses to
 * action 0 — the table's first row (coast/left/hold).
 *
 * WHY clamp instead of throw: this sits on the hot path of every agent's
 * argmax every tick. A desynced buffer or a hand-rolled caller producing 18
 * must degrade to "reverse+right" — never alias (bit-fiddling −1 wraps into
 * nonsense) and never abort a training lane mid-episode.
 */
export function decode(a: number): Action {
  const i = !Number.isFinite(a)
    ? a > 0 ? ACTION_COUNT - 1 : 0 // NaN is not > 0 → falls to 0 too
    : Math.min(ACTION_COUNT - 1, Math.max(0, Math.trunc(a)));
  const fire = i % 2 === 1;
  const steer = (Math.trunc(i / 2) % 3) as 0 | 1 | 2;
  const throttle = Math.trunc(i / 6) as 0 | 1 | 2;
  return { throttle, steer, fire };
}

/**
 * Inverse of decode: pack components into the canonical index. throttle/steer
 * clamp to 0..2 (same boundary-safety argument as decode — encode is also fed
 * by the play mode's aim-assist/τ-sampling paths, not just by decode).
 */
export function encodeAction(throttle: number, steer: number, fire: boolean): number {
  const t = clamp3(throttle);
  const s = clamp3(steer);
  return (t * 3 + s) * 2 + (fire ? 1 : 0);
}

/** Clamp a component to 0..2; non-finite collapses to 0, which makes
 * encodeAction(NaN, NaN, ·) agree with decode(NaN) on action 0. */
function clamp3(v: number): 0 | 1 | 2 {
  if (!Number.isFinite(v)) return 0;
  const i = Math.trunc(v);
  return (i < 0 ? 0 : i > 2 ? 2 : i) as 0 | 1 | 2;
}

/**
 * Human keymap → action index, through the SAME table the network uses (the
 * plan pins one action space for both). W/S → throttle 1/2, none → 0;
 * A/D → steer 0/2, none → 1; Space → fire.
 *
 * Both keys of a pair held at once is a real (if rare) input state and must be
 * deterministic: forward beats reverse and left beats right, matching the
 * table's low-index-first reading. Key order is not observable from booleans,
 * so "last pressed wins" is not an option here.
 */
export function actionFromKeys(keys: {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  fire: boolean;
}): number {
  const throttle = keys.up ? 1 : keys.down ? 2 : 0;
  const steer = keys.left ? 0 : keys.right ? 2 : 1;
  return encodeAction(throttle, steer, keys.fire);
}
