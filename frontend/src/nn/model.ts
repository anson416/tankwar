// model.ts — the Q-network factory (plan §Network): ONE interface, TWO
// architectures behind it. tfjs, worker-side ONLY — nn/** and the worker are
// the only places allowed to import @tensorflow/*, so the main bundle never
// ships it (a unit-test grep enforces this).
//
// WHY the 18-action head is LINEAR in both architectures (big-two precedent):
// Q(s, ·) must be free to go negative and unbounded. A ReLU head clamps every
// action value at 0, so as soon as the true value function dips negative the
// regression collapses onto the scalar bias — the TD target for the WORST
// actions (the ones bootstrapping most needs priced) carries no gradient.
// ReLU lives in the hidden layers; the head is affine with the tfjs dense
// defaults (glorot-uniform kernel, zero bias), which is the init contract.
//
// WHY the attention encoder is raw tf.ops math instead of
// tf.layers.multiHeadAttention: tfjs-layers 4.22 ships MultiHeadAttention only
// as an INTERNAL module (dist/layers/nlp/multihead_attention.d.ts) — it is NOT
// re-exported by the public `layers` namespace (checked: exports_layers.d.ts),
// so using it means a deep import of a dist path the version pin does not
// promise to keep. That fragility is exactly why the plan chose learned
// sentinel embeddings over attention masks. The block below is therefore
// plain matMul/softmax over projection kernels this file owns — every
// parameter is a visible tf.Variable, so variables()/weights()/setWeights()/
// dispose() are trivially exhaustive and optimizer.minimize trains it.
//
// Token layout (plan §Observation table, offsets imported from encoding.ts —
// never re-declared here, a second copy of the layout table is precisely the
// silent drift that would train a net on scrambled features): the flat
// [B, 282] obs is SLICED into 26 entity tokens
//   SELF 1×16 | TANKS 3×12 | BULLETS 18×7 | POWERUPS 4×6   (= 26 tokens)
// and the RAYS block (16×5 = 80 floats) stays flat, concatenated at the head.
// Token counts are DERIVED from OFF and the slot strides rather than
// hard-coded, so they cannot disagree with the encoder they slice.
//
// Memory rules (house discipline, big-two precedent):
//  - q() runs the forward pass inside tf.tidy and returns ONLY the [B,18]
//    result — the CALLER owns and disposes it.
//  - weights() returns fresh clones (a snapshot); the caller disposes them.
//  - setWeights() copies FROM its argument. tfjs Variable.assign() re-points
//    the variable at the source buffer (verified tfjs-core 4.22:
//    disposeTensor(old); this.dataId = newValue.dataId) — refcounted, not
//    copied — so assigning the caller's tensors directly would alias them.
//    We clone, assign, drop the clone: the net owns an exclusive buffer and
//    the caller may dispose its snapshot immediately after the call.
//  - Raw variables are created UNNAMED: tfjs's named-variable registry is NOT
//    cleaned by dispose(), so a fixed name throws "already registered" on
//    every rebuild. Layers auto-suffix duplicate names and stay named.
//  - Weights are initialized with tfjs-layers initializers (glorot-uniform
//    kernels, zero biases, LN γ=1/β=0) — the same contract tf.layers.dense
//    gives the MLP path. No Math.random anywhere.

import * as tf from "@tensorflow/tfjs-core";
import { initializers, layers, sequential } from "@tensorflow/tfjs-layers";
import type { LayersModel } from "@tensorflow/tfjs-layers";
import { ACTION_COUNT } from "../game/actions";
import { OBS_DIM, RAY_COUNT } from "../game/constants";
import { BULLET_SLOT, OFF, POWERUP_SLOT, RAY_SLOT, TANK_SLOT } from "../game/encoding";
import type { NetConfig } from "../train-protocol";

// --- observation layout (derived, single source of truth = encoding.ts) -----

/** SELF block width: everything before the first tank slot. */
const SELF_W = OFF.TANKS - OFF.SELF; // 16
/** Token counts, derived from the block offsets + slot strides. */
const N_TANK = (OFF.BULLETS - OFF.TANKS) / TANK_SLOT; // 3
const N_BULLET = (OFF.POWERUPS - OFF.BULLETS) / BULLET_SLOT; // 18
const N_POWER = (OFF.RAYS - OFF.POWERUPS) / POWERUP_SLOT; // 4
/** Flat ray fan width — RAYS block is NOT tokenized (a fixed 16-fan lidar is
 *  already positional; tokens would add attention cost for no structure). */
const RAYS_W = RAY_COUNT * RAY_SLOT; // 80
/** 1 self + 3 tanks + 18 bullets + 4 powerups. */
const N_TOKEN = 1 + N_TANK + N_BULLET + N_POWER; // 26

/** Type-embedding rows: self, tank, bullet, powerup, sentinel. */
const TYPE_SELF = 0;
const TYPE_TANK = 1;
const TYPE_BULLET = 2;
const TYPE_POWER = 3;
const TYPE_SENTINEL = 4;
const N_TYPE = 5;

/** The embedding row each token WOULD read if it were valid (SELF always is). */
const BASE_TYPE: readonly number[] = [
  TYPE_SELF,
  ...fill(N_TANK, TYPE_TANK),
  ...fill(N_BULLET, TYPE_BULLET),
  ...fill(N_POWER, TYPE_POWER),
];

function fill(n: number, v: number): number[] {
  return new Array<number>(n).fill(v);
}

/** LayerNorm epsilon — matches tf.layers.layerNormalization's default, so the
 *  hand-rolled LN and any layers-based path share one numeric contract. */
const LN_EPS = 1e-3;
/** Pool-denominator floor. SELF is always valid so the true denominator is
 *  ≥ 1; the floor only stops a hypothetical all-invalid batch (a caller bug)
 *  from producing NaN instead of garbage the loss would flag. */
const POOL_EPS = 1e-6;
/** Post-encoder head width — FIXED by the plan (cfg.ffn is the attention
 *  block's FFN width, a different knob). */
const HEAD_W = 128;

const GLOROT = initializers.glorotUniform({});
const ZEROS = initializers.zeros();
const ONES = initializers.ones();

// --- the contract ------------------------------------------------------------

/** A Q network over the flat 282-float observation: [B, 282] → [B, 18] linear
 *  action values. Built by buildQNetwork; every method is worker-side only. */
export interface QNetwork {
  /** Action values for a batch of observations. The returned tensor is
   *  caller-owned: dispose it (inside a tidy, or after an awaited read). */
  q(obs: tf.Tensor2D): tf.Tensor2D;
  /** Trainable variables, for optimizer.minimize(f, vars). */
  variables(): tf.Variable[];
  /** Cloned copies of every weight, fixed order — a snapshot the caller owns
   *  (feed to another net's setWeights, then dispose). */
  weights(): tf.Tensor[];
  /** Copy FROM `w` into the net's variables (no aliasing; the caller keeps
   *  ownership of `w` and may dispose it right after). */
  setWeights(w: tf.Tensor[]): void;
  /** Alias of q — kept for symmetry with the big-two tower naming. */
  qSync(obs: tf.Tensor2D): tf.Tensor2D;
  /** Dispose every weight tensor. The net is dead afterwards. */
  dispose(): void;
}

/** Build a fresh Q network. MUST be called only after `await tf.setBackend()`
 *  and `tf.ready()` — it allocates tensors immediately (the worker gates this
 *  behind its backend ladder). */
export function buildQNetwork(cfg: NetConfig): QNetwork {
  checkLayout();
  return cfg.architecture === "attention"
    ? new AttentionQNetwork(cfg)
    : new MlpQNetwork(cfg.hidden);
}

/** One-time guard that the layout arithmetic above is still exact: the counts
 *  are quotients of block offsets, so a stride change upstream would leave a
 *  fraction here or break the OBS_DIM sum — either way, loudly. */
let layoutChecked = false;
function checkLayout(): void {
  if (layoutChecked) return;
  for (const [n, what] of [
    [N_TANK, "tank"],
    [N_BULLET, "bullet"],
    [N_POWER, "powerup"],
  ] as const) {
    if (!Number.isInteger(n)) {
      throw new Error(`model: ${what} token count ${n} is not an integer — obs layout drifted`);
    }
  }
  const total =
    OFF.SELF + SELF_W +
    N_TANK * TANK_SLOT + N_BULLET * BULLET_SLOT + N_POWER * POWERUP_SLOT +
    RAYS_W;
  if (total !== OBS_DIM) {
    throw new Error(`model: token layout covers ${total} floats, OBS_DIM is ${OBS_DIM}`);
  }
  layoutChecked = true;
}

function assertObs(obs: tf.Tensor2D): void {
  if (obs.shape.length !== 2 || obs.shape[1] !== OBS_DIM) {
    throw new RangeError(`q: expected shape [B, ${OBS_DIM}], got [${obs.shape}]`);
  }
}

/** Length/shape validation shared by both setWeights paths — check EVERYTHING
 *  before assigning anything, so a bad list cannot leave the net half-updated
 *  (fatal mid-training: that net is the one producing targets). */
function assertCompatible(live: readonly tf.Tensor[], w: readonly tf.Tensor[], what: string): void {
  if (w.length !== live.length) {
    throw new RangeError(`${what}.setWeights: got ${w.length} tensors for ${live.length} variables`);
  }
  for (let i = 0; i < live.length; i++) {
    const a = w[i];
    const b = live[i];
    if (a.dtype !== b.dtype || !sameShape(a.shape, b.shape)) {
      throw new RangeError(
        `${what}.setWeights: tensor ${i} is ${a.dtype}${JSON.stringify(a.shape)}, ` +
        `expected ${b.dtype}${JSON.stringify(b.shape)}`,
      );
    }
  }
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

// --- architecture 1: MLP (default) --------------------------------------------

/** [282] → Dense hidden[0] ReLU → … → Dense 18 LINEAR (see header). */
class MlpQNetwork implements QNetwork {
  private readonly model: LayersModel;

  constructor(hidden: readonly number[]) {
    const m = sequential({ name: "q_mlp" });
    let first = true;
    for (let i = 0; i < hidden.length; i++) {
      m.add(
        first
          ? layers.dense({ units: hidden[i], activation: "relu", inputShape: [OBS_DIM], name: `q_h${i}` })
          : layers.dense({ units: hidden[i], activation: "relu", name: `q_h${i}` }),
      );
      first = false;
    }
    // Linear head — the layer whose activation choice this file exists to
    // document (see header). `first` still true for an empty hidden array:
    // the obs feeds the head directly.
    m.add(
      first
        ? layers.dense({ units: ACTION_COUNT, inputShape: [OBS_DIM], name: "q_head" })
        : layers.dense({ units: ACTION_COUNT, name: "q_head" }),
    );
    this.model = m;
  }

  q(obs: tf.Tensor2D): tf.Tensor2D {
    assertObs(obs);
    // tidy keeps ONLY the returned tensor: every hidden activation dies here.
    return tf.tidy(() => this.model.apply(obs) as tf.Tensor2D);
  }

  qSync(obs: tf.Tensor2D): tf.Tensor2D {
    return this.q(obs);
  }

  variables(): tf.Variable[] {
    // LayerVariable.val is protected in tfjs-layers' typings (read()/write()
    // are the public surface), but optimizer.minimize needs the underlying
    // Variable handles — one documented cast at the boundary.
    return this.model.trainableWeights.map((lw) => (lw as unknown as { val: tf.Variable }).val);
  }

  weights(): tf.Tensor[] {
    // getWeights() returns the layers' LIVE variable tensors (numTensors Δ0,
    // verified big-two) — clone to make the snapshot independent of training.
    return this.model.getWeights().map((t) => t.clone());
  }

  setWeights(w: tf.Tensor[]): void {
    // getWeights() allocates nothing, so using it as the shape oracle is free.
    assertCompatible(this.model.getWeights(), w, "q-mlp");
    const copies = w.map((t) => t.clone());
    this.model.setWeights(copies);
    for (const c of copies) c.dispose();
  }

  dispose(): void {
    void this.model.dispose();
  }
}

// --- architecture 2: entity-token attention encoder ----------------------------

/** y = reshape(flatten(x)·W + b, shape) — the encoder's ONLY matmul plumbing:
 *  tfjs matMul takes 2D×2D or batched SAME-rank inputs, so every projection
 *  flattens its token axis first. The `shape` arity fixes the result rank and
 *  the cast is that assertion. */
function dense<T extends tf.Tensor>(x: tf.Tensor, w: tf.Tensor2D, b: tf.Tensor1D, shape: number[]): T {
  const flat = tf.reshape(x, [-1, x.shape[x.shape.length - 1]]);
  return tf.reshape(tf.add(tf.matMul(flat as tf.Tensor2D, w), b), shape) as T;
}

/** Pre-LN: (x − μ)/√(σ² + ε) · γ + β over the last axis. Hand-rolled because
 *  the rest of the block is: same ops, same gradients, zero extra machinery. */
function layerNorm(x: tf.Tensor, gamma: tf.Tensor1D, beta: tf.Tensor1D): tf.Tensor {
  const centered = tf.sub(x, tf.mean(x, -1, true));
  const variance = tf.mean(tf.square(centered), -1, true);
  return tf.add(
    tf.mul(tf.div(centered, tf.sqrt(tf.add(variance, LN_EPS))), gamma),
    beta,
  );
}

/** Each entity token's FIRST raw feature is the encoder's valid flag (plan
 *  §Observation) — pull it out of a [B, n, slot] token block as [B, n]. */
function validFlags(block: tf.Tensor3D): tf.Tensor2D {
  return tf.reshape(tf.slice(block, [0, 0, 0], [-1, block.shape[1], 1]), [-1, block.shape[1]]);
}

interface BlockVars {
  // attention sub-block: LN → Q/K/V/out projections (residual outside)
  g1: tf.Variable<tf.Rank.R1>;
  b1: tf.Variable<tf.Rank.R1>;
  wq: tf.Variable<tf.Rank.R2>;
  bq: tf.Variable<tf.Rank.R1>;
  wk: tf.Variable<tf.Rank.R2>;
  bk: tf.Variable<tf.Rank.R1>;
  wv: tf.Variable<tf.Rank.R2>;
  bv: tf.Variable<tf.Rank.R1>;
  wo: tf.Variable<tf.Rank.R2>;
  bo: tf.Variable<tf.Rank.R1>;
  // FFN sub-block: LN → Dense ffn ReLU → Dense d (residual outside)
  g2: tf.Variable<tf.Rank.R1>;
  b2: tf.Variable<tf.Rank.R1>;
  f1w: tf.Variable<tf.Rank.R2>;
  f1b: tf.Variable<tf.Rank.R1>;
  f2w: tf.Variable<tf.Rank.R2>;
  f2b: tf.Variable<tf.Rank.R1>;
}

/**
 * 26-token entity encoder (plan §Network, attention option):
 *   per-type Dense → + learned type embeddings (5 rows, sentinel row 4 for
 *   invalid tokens) → `layers` pre-LN blocks of multi-head self-attention +
 *   FFN (both residual) → valid-weighted mean pool → concat [pooled, SELF 16,
 *   RAYS 80] → Dense 128 ReLU → Dense 18 linear.
 *
 * NO attention mask, deliberately (plan's explicit decision): invalid tokens
 * are not removed from the attention — they carry the SENTINEL embedding
 * (row 4), a learned, constant "nothing here" key/value, because masks are
 * where tfjs version fragility lives and a learned sentinel does the same job
 * with ordinary dense math. Only the pool drops invalid tokens.
 */
class AttentionQNetwork implements QNetwork {
  private readonly d: number;
  private readonly heads: number;
  private readonly hd: number; // d / heads = per-head key/value dim
  private readonly ffnW: number;
  private readonly scale: number; // 1/√hd
  private readonly vs: tf.Variable[] = []; // registry; FIXED order = weights() order
  private readonly selfW: tf.Variable<tf.Rank.R2>;
  private readonly selfB: tf.Variable<tf.Rank.R1>;
  private readonly tankW: tf.Variable<tf.Rank.R2>;
  private readonly tankB: tf.Variable<tf.Rank.R1>;
  private readonly bulletW: tf.Variable<tf.Rank.R2>;
  private readonly bulletB: tf.Variable<tf.Rank.R1>;
  private readonly powerW: tf.Variable<tf.Rank.R2>;
  private readonly powerB: tf.Variable<tf.Rank.R1>;
  private readonly typeEmb: tf.Variable<tf.Rank.R2>;
  private readonly blocks: BlockVars[] = [];
  private readonly h1w: tf.Variable<tf.Rank.R2>;
  private readonly h1b: tf.Variable<tf.Rank.R1>;
  private readonly h2w: tf.Variable<tf.Rank.R2>;
  private readonly h2b: tf.Variable<tf.Rank.R1>;
  /** BASE_TYPE as a float32 constant — float32 so the valid-flag arithmetic
   *  below never mixes dtypes; disposed in dispose(). */
  private readonly baseType: tf.Tensor1D;

  constructor(cfg: NetConfig) {
    if (cfg.dModel < 1 || cfg.heads < 1 || cfg.dModel % cfg.heads !== 0) {
      throw new RangeError(
        `attention: dModel ${cfg.dModel} must be a positive multiple of heads ${cfg.heads}`,
      );
    }
    if (cfg.layers < 1 || cfg.ffn < 1) {
      throw new RangeError(`attention: layers ${cfg.layers} and ffn ${cfg.ffn} must be ≥ 1`);
    }
    this.d = cfg.dModel;
    this.heads = cfg.heads;
    this.hd = cfg.dModel / cfg.heads;
    this.ffnW = cfg.ffn;
    this.scale = 1 / Math.sqrt(this.hd);

    const kernel = (inDim: number, outDim: number): tf.Variable<tf.Rank.R2> => this.regK(inDim, outDim);
    const vec = (n: number): tf.Variable<tf.Rank.R1> => this.regV(n, "zero");
    const gamma = (n: number): tf.Variable<tf.Rank.R1> => this.regV(n, "one");

    // Per-type projections: each token row is narrow (16/12/7/6 floats), so
    // four small kernels cost (16+12+7+6)·d weights instead of one 282·d
    // matrix — and a tank row can never be linearly confused with a bullet
    // row before attention even sees it.
    this.selfW = kernel(SELF_W, this.d);
    this.selfB = vec(this.d);
    this.tankW = kernel(TANK_SLOT, this.d);
    this.tankB = vec(this.d);
    this.bulletW = kernel(BULLET_SLOT, this.d);
    this.bulletB = vec(this.d);
    this.powerW = kernel(POWERUP_SLOT, this.d);
    this.powerB = vec(this.d);
    this.typeEmb = kernel(N_TYPE, this.d);

    for (let i = 0; i < cfg.layers; i++) {
      this.blocks.push({
        g1: gamma(this.d),
        b1: vec(this.d),
        wq: kernel(this.d, this.d),
        bq: vec(this.d),
        wk: kernel(this.d, this.d),
        bk: vec(this.d),
        wv: kernel(this.d, this.d),
        bv: vec(this.d),
        wo: kernel(this.d, this.d),
        bo: vec(this.d),
        g2: gamma(this.d),
        b2: vec(this.d),
        f1w: kernel(this.d, cfg.ffn),
        f1b: vec(cfg.ffn),
        f2w: kernel(cfg.ffn, this.d),
        f2b: vec(this.d),
      });
    }

    this.h1w = kernel(this.d + SELF_W + RAYS_W, HEAD_W);
    this.h1b = vec(HEAD_W);
    this.h2w = kernel(HEAD_W, ACTION_COUNT);
    this.h2b = vec(ACTION_COUNT);
    this.baseType = tf.tensor1d(Float32Array.from(BASE_TYPE));
  }

  /** Create one UNNAMED variable (see header for why unnamed), register it in
   *  the fixed-order list, and dispose the initializer tensor — tf.variable()
   *  gives the Variable its own backing, so the init would otherwise leak one
   *  tensor per parameter per build (i.e. per reconfigure). The rank
   *  assertions below are exact: the initializer shape IS the rank. */
  private regK(inDim: number, outDim: number): tf.Variable<tf.Rank.R2> {
    const init = GLOROT.apply([inDim, outDim]) as tf.Tensor2D;
    const v = tf.variable(init, true);
    init.dispose();
    this.vs.push(v);
    return v;
  }

  private regV(n: number, kind: "zero" | "one"): tf.Variable<tf.Rank.R1> {
    const init = (kind === "one" ? ONES : ZEROS).apply([n]) as tf.Tensor1D;
    const v = tf.variable(init, true);
    init.dispose();
    this.vs.push(v);
    return v;
  }

  q(obs: tf.Tensor2D): tf.Tensor2D {
    assertObs(obs);
    return tf.tidy(() => this.forward(obs));
  }

  qSync(obs: tf.Tensor2D): tf.Tensor2D {
    return this.q(obs);
  }

  variables(): tf.Variable[] {
    return [...this.vs]; // copy: the caller must not reorder our registry
  }

  weights(): tf.Tensor[] {
    return this.vs.map((v) => v.clone());
  }

  setWeights(w: tf.Tensor[]): void {
    assertCompatible(this.vs, w, "q-attention");
    const copies = w.map((t) => t.clone());
    for (let i = 0; i < this.vs.length; i++) this.vs[i].assign(copies[i]);
    for (const c of copies) c.dispose();
  }

  dispose(): void {
    for (const v of this.vs) v.dispose();
    this.vs.length = 0;
    this.blocks.length = 0;
    this.baseType.dispose();
  }

  // -- forward pass (caller wraps in tf.tidy) ---------------------------------

  private forward(obs: tf.Tensor2D): tf.Tensor2D {
    const b = obs.shape[0] as number;

    // Slice the flat obs into token groups — offsets/strides straight from
    // encoding.ts; size -1 = "every row of the batch". The `as tf.Tensor3D`
    // casts assert the arity each reshape hands back (tfjs' reshape overloads
    // only narrow for tuple shapes, and these hold computed counts).
    const selfF = tf.slice(obs, [0, OFF.SELF], [-1, SELF_W]); // [B, 16]
    const raysF = tf.slice(obs, [0, OFF.RAYS], [-1, RAYS_W]); // [B, 80]
    const tanks = tf.reshape(
      tf.slice(obs, [0, OFF.TANKS], [-1, N_TANK * TANK_SLOT]),
      [-1, N_TANK, TANK_SLOT],
    ) as tf.Tensor3D;
    const bullets = tf.reshape(
      tf.slice(obs, [0, OFF.BULLETS], [-1, N_BULLET * BULLET_SLOT]),
      [-1, N_BULLET, BULLET_SLOT],
    ) as tf.Tensor3D;
    const powerups = tf.reshape(
      tf.slice(obs, [0, OFF.POWERUPS], [-1, N_POWER * POWERUP_SLOT]),
      [-1, N_POWER, POWERUP_SLOT],
    ) as tf.Tensor3D;

    // valid ∈ [0, 1] per token (clamped: the encoder writes an exact 0|1, and
    // clamping keeps the type-id arithmetic inside the 5-row table even if a
    // bug upstream emits something else — an out-of-range gather index would
    // read garbage or crash instead of failing the loss). SELF is always
    // valid: it is the observer, never a sentinel.
    const valid = tf.clipByValue(
      tf.concat([tf.ones([b, 1]), validFlags(tanks), validFlags(bullets), validFlags(powerups)], 1),
      0,
      1,
    ); // [B, 26]

    // type id = SENTINEL + valid·(base − SENTINEL): the entity's own row when
    // valid, the sentinel row when not.
    const typeIds = tf.cast(
      tf.add(TYPE_SENTINEL, tf.mul(valid, tf.sub(this.baseType, TYPE_SENTINEL))),
      "int32",
    ); // [B, 26]

    // Embed + project. tf.gather with 2-D indices broadcasts the table over
    // the batch: [5, d] ⋉ [B, 26] → [B, 26, d].
    const toks: tf.Tensor = tf.add(
      tf.concat(
        [
          dense(tf.reshape(selfF, [-1, 1, SELF_W]), this.selfW, this.selfB, [-1, 1, this.d]),
          dense(tanks, this.tankW, this.tankB, [-1, N_TANK, this.d]),
          dense(bullets, this.bulletW, this.bulletB, [-1, N_BULLET, this.d]),
          dense(powerups, this.powerW, this.powerB, [-1, N_POWER, this.d]),
        ],
        1,
      ),
      tf.gather(this.typeEmb, typeIds),
    );

    let h: tf.Tensor = toks; // stays [B, 26, d] but generic rank keeps the ops happy
    for (const blk of this.blocks) {
      // Pre-LN (Xiong et al. 2020): the norm INSIDE the residual branch keeps
      // the identity path clean, so a deep stack trains without warmup —
      // the reason a 2-block toy encoder can be dropped into a DQN cold.
      h = tf.add(h, this.mha(layerNorm(h, blk.g1, blk.b1), blk));
      h = tf.add(h, this.ffn(layerNorm(h, blk.g2, blk.b2), blk));
    }

    // Valid-weighted mean pool: invalid tokens (already sentinel-embedded for
    // attention) contribute nothing to the summary — the plan's "masked mean".
    const wValid = tf.reshape(valid, [-1, N_TOKEN, 1]); // [B, 26, 1]
    const pooled = tf.div(
      tf.sum(tf.mul(h, wValid), 1),
      tf.add(tf.sum(wValid, 1), POOL_EPS),
    ); // [B, d]

    // Head: pool + raw SELF + raw ray fan → 128 ReLU → 18 linear.
    const cat = tf.concat([pooled, selfF, raysF], 1); // [B, d + 16 + 80]
    const hid = tf.relu(dense<tf.Tensor2D>(cat, this.h1w, this.h1b, [-1, HEAD_W]));
    return dense<tf.Tensor2D>(hid, this.h2w, this.h2b, [-1, ACTION_COUNT]);
  }

  /** Multi-head self-attention over the 26 tokens: softmax(QKᵀ/√hd)·V. Batch
   *  and head axes are FOLDED into one leading axis so the score/context
   *  matMuls stay batched 3-D (tfjs matMul overloads stop at rank 3) — this
   *  is the manual-math replacement for the unexported
   *  tf.layers.multiHeadAttention (see header). */
  private mha(h: tf.Tensor, blk: BlockVars): tf.Tensor {
    const qkv = (w: tf.Variable<tf.Rank.R2>, b: tf.Variable<tf.Rank.R1>): tf.Tensor3D =>
      // [B, T, d] → project → [B, T, heads, hd] → [B, heads, T, hd] → fold
      tf.reshape(
        tf.transpose(
          dense<tf.Tensor4D>(h, w as tf.Tensor2D, b as tf.Tensor1D, [-1, N_TOKEN, this.heads, this.hd]),
          [0, 2, 1, 3],
        ),
        [-1, N_TOKEN, this.hd],
      ) as tf.Tensor3D;
    const q = qkv(blk.wq, blk.bq);
    const k = qkv(blk.wk, blk.bk);
    const v = qkv(blk.wv, blk.bv);
    const probs = tf.softmax(tf.mul(tf.matMul(q, k, false, true), this.scale)); // [B·h, T, T]
    const ctx = tf.matMul(probs, v); // [B·h, T, hd]
    // Unfold heads → [B, T, heads, hd] → [B, T, d] → output projection.
    return dense<tf.Tensor3D>(
      tf.reshape(tf.transpose(tf.reshape(ctx, [-1, this.heads, N_TOKEN, this.hd]), [0, 2, 1, 3]), [
        -1,
        N_TOKEN,
        this.d,
      ]),
      blk.wo as tf.Tensor2D,
      blk.bo as tf.Tensor1D,
      [-1, N_TOKEN, this.d],
    );
  }

  /** FFN sub-block: Dense ffn ReLU → Dense d (residual added by the caller). */
  private ffn(h: tf.Tensor, blk: BlockVars): tf.Tensor {
    const hid = tf.relu(dense<tf.Tensor3D>(h, blk.f1w as tf.Tensor2D, blk.f1b as tf.Tensor1D, [-1, N_TOKEN, this.ffnW]));
    return dense<tf.Tensor3D>(hid, blk.f2w as tf.Tensor2D, blk.f2b as tf.Tensor1D, [-1, N_TOKEN, this.d]);
  }
}

