// rng.ts — seeded, state-threading PRNG for the deterministic engine.
// DOM-free, tfjs-free.
//
// WHY a pure `(state) → value` API instead of a closure: the engine's random
// counter lives in GameState.rngState, so an episode can be checkpointed
// mid-flight, replayed bit-exactly, and reproduced by the 2000-game property
// tests. A closure hides its counter inside a function object; a plain integer
// threads through the state like any other field and survives postMessage.
//
// mulberry32 (Tommy Ettinger's 32-bit generator — the same one the reference
// projects' test util uses): one add + two imul mixes per draw, ~2^32 period.
// Statistical quality is far beyond what spawn picks, powerup scheduling and
// ε-greedy need, and it is identical everywhere JS runs because every integer
// step is a |0/imul/>>> op (no float platform variance, no Math.random).
//
// CALLERS thread the state explicitly:
//   const [kind, s1] = randInt(s.rngState, POWERUP_KINDS.length);
//   s.rngState = s1;
// A float draw that also needs the successor pairs `rand(state)` with
// `nextState(state)` — the two advance the same counter, so they are always a
// consistent (value, state) pair.

/** Advance the raw mulberry32 counter. The half of a `rand` draw that callers
 *  must assign back to GameState.rngState. */
export function nextState(state: number): number {
  return (state + 0x6d2b79f5) | 0;
}

/** Mix one counter value into 32 unsigned pseudorandom bits. */
function mix(a: number): number {
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return (t ^ (t >>> 14)) >>> 0;
}

/**
 * One pure mulberry32 step: a float in [0, 1) derived from `state`. Pure — the
 * input state is unchanged; pair with `nextState(state)` to thread the counter.
 */
export function rand(state: number): number {
  return mix(nextState(state)) / 4294967296;
}

/**
 * Uniform integer in [0, n) plus the advanced state — the workhorse for picking
 * spawn points, powerup kinds and ε-greedy exploration. Exactly
 * `floor(rand(state) * n)`, so a caller can derive the same integer from a float
 * draw without consuming a second one. n ≤ 0 yields 0 (total function; engine
 * call sites always pass ≥ 1) and the counter still advances, so no caller can
 * loop forever on a degenerate n.
 */
export function randInt(state: number, n: number): [value: number, state: number] {
  const next = nextState(state);
  // min() is float-rounding insurance for huge n; max() covers n ≤ 0.
  const v = Math.max(0, Math.min(n - 1, Math.floor((mix(next) / 4294967296) * n)));
  return [v, next];
}
