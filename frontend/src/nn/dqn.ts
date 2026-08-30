// dqn.ts — the learner: one online Q-net, one frozen target copy, the DDQN
// gradient step, and the ONE batched behaviour-policy forward the whole fleet
// shares per tick. tfjs, worker-side ONLY: nn/** and the train worker are the
// only modules allowed to import @tensorflow/*, and src/app/** + main.ts ship
// none of it (a unit-test grep enforces the split).
//
// MEMORY DISCIPLINE (house rule, dmc.spec-verified pattern): no dataSync
// anywhere — every value crossing back to JS goes through ONE `await .data()`
// per call, taken OUTSIDE the tf.tidy that built it, and every tensor the call
// created is disposed on every path (the `finally` blocks below exist because
// a thrown kernel must not leak the inputs it failed to consume). Adam's m/v
// slots are created lazily on the first minimize, so anything measuring a
// tensor baseline must do it AFTER the first train step.
//
// DDQN (plan §Rewards/bootstrap — the explicit rules, both pinned here):
//   selection  a* = argmax_a Q_online(s′) — UNMASKED. The behaviour policy
//              masks fire because an unavailable shot is a wasted action the
//              agent can observe (reload is a feature); the TARGET max must
//              not mask, or the value function would learn biased, episode-
//              phase-dependent bootstraps.
//   evaluation target = r + γ(1−done)·Q_target(s′)[a*] — the target NET prices
//              the selected action, which is what de-correlates the max
//              operator's overestimation from the regression target.
//   ablation   doubleQ=false collapses to plain DQN: target = r +
//              γ(1−done)·max_a Q_target(s′,a). Both arms must train finitely
//              (dqn.spec); DDQN is on by default.
// The whole TD target is computed OUTSIDE the gradient tape: argmax is not
// differentiable, the target net is frozen by construction, and the online
// net's next-state forward only SELECTS — so the tape carries only the
// online-Q(s) forward, roughly halving the backward's graph.

import * as tf from "@tensorflow/tfjs-core";
import { ACTION_COUNT } from "../game/actions";
import { OBS_DIM } from "../game/constants";
import { nextState, rand, randInt } from "../game/rng";
import { buildQNetwork } from "./model";
import type { QNetwork } from "./model";
import type { ReplayBatch } from "./replay";
import type { NetConfig } from "../train-protocol";

/**
 * How `act` turns masked Q values into actions.
 *  - explore: ε-greedy — per row, draw one uniform; below ε pick UNIFORMLY
 *    over ALL 18 actions (see act's header note on why exploration is NOT
 *    fire-masked), otherwise the masked argmax.
 *  - temperature: one categorical sample per row from softmax(maskedQ/τ) —
 *    the play mode's difficulty knob (plan: τ = (1−difficulty)+0.05); τ is
 *    clamped to a positive floor so a pathological 0 cannot reach the divide.
 */
export type ActMode = { kind: "explore"; epsilon: number } | { kind: "temperature"; tau: number };

/** τ below this is clamped up: ÷0 would make softmax output NaN fleet-wide. */
const TAU_FLOOR = 1e-3;

/**
 * The behaviour-policy + learner pair. `net` is trained and used for both the
 * fleet forward and the DDQN selection; `target` is a bit-exact copy (see the
 * constructor) that only ever moves through syncTarget. Both live and die
 * together — dispose() is the only teardown.
 */
export class Dqn {
  readonly net: QNetwork;
  readonly target: QNetwork;
  readonly optimizer: tf.Optimizer;

  private readonly gamma: number;
  private readonly doubleQ: boolean;
  /** Gradient steps applied — the worker's targetSync counter. */
  private stepCount = 0;

  constructor(cfg: NetConfig) {
    // Transactional build: the second buildQNetwork can throw (attention
    // config validation) — the half-built pair must not leak the first net.
    const online = buildQNetwork(cfg);
    let target: QNetwork;
    try {
      target = buildQNetwork(cfg);
    } catch (err) {
      online.dispose();
      throw err;
    }
    this.net = online;
    this.target = target;
    this.gamma = cfg.gamma;
    this.doubleQ = cfg.doubleQ;
    this.optimizer = tf.train.adam(cfg.learningRate);
    // The target starts bit-equal, not "another random init": the very first
    // TD target would otherwise compare against unrelated weights, and every
    // snapshot/restore invariant below rests on sync meaning exact copy.
    this.syncTarget();
  }

  /**
   * Act for a whole fleet slice in ONE forward: `obs` rows are packed into a
   * single [B, 282] batch, the graph runs inside one tf.tidy, and exactly one
   * awaited tensor read comes back out of it.
   *
   * Fire masking: `fireAvail[i] === false` pushes every ODD action index (the
   * fire axis has stride 2 in the action table) to −Infinity BEFORE the
   * argmax/softmax, so an unavailable shot can never be chosen greedily. The
   * mask is an additive −Inf, not an algebraic re-ordering: Q is finite by
   * construction (bounded inputs, glorot init), so −Inf can only ever meet a
   * finite value and −Inf + finite = −Inf exactly — no NaN route exists.
   *
   * Exploration is deliberately NOT masked: ε's job is unbiased coverage of
   * the action space (the agent must learn WHEN firing is worth its cooldown),
   * and coupling the explorer to the cooldown state would skew the replay's
   * action marginal by episode phase. An unavailable shot simply no-ops in the
   * engine (tryFire returns false), costing one wasted action occasionally —
   * the price of an unbiased explorer.
   *
   * No fallback chain lives here: a caller that needs chosen→first-legal→
   * random (the play mode's never-freeze rule) re-rolls on ITS side; this
   * method's contract is exactly "argmax / ε-mix / τ-sample over masked Q".
   *
   * Determinism: every draw threads the mulberry32 counter — the returned
   * `state` is the counter after this call's draws, so a lane checkpointed
   * mid-episode resumes the identical stream. A pure-greedy call (ε=0) draws
   * nothing and hands the counter back untouched; an exploring call consumes
   * exactly one threshold draw per row plus one uniform per explored row.
   */
  async act(
    obs: Float32Array[],
    fireAvail: boolean[],
    mode: ActMode,
    rngState: number,
  ): Promise<{ actions: Int32Array; state: number }> {
    const n = obs.length;
    if (n === 0) return { actions: new Int32Array(0), state: rngState };
    // Pack + mask on plain typed arrays first: one batch tensor and one mask
    // tensor per call, no per-row tensors. A caller scratch buffer longer than
    // OBS_DIM is truncated (set() would throw on the overflow).
    const flat = new Float32Array(n * OBS_DIM);
    const mask = new Float32Array(n * ACTION_COUNT);
    for (let i = 0; i < n; i++) {
      flat.set(obs[i].subarray(0, OBS_DIM), i * OBS_DIM);
      if (!fireAvail[i]) {
        for (let a = 1; a < ACTION_COUNT; a += 2) mask[i * ACTION_COUNT + a] = -Infinity;
      }
    }
    const obsT = tf.tensor2d(flat, [n, OBS_DIM]);
    const maskT = tf.tensor2d(mask, [n, ACTION_COUNT]);
    let picked: tf.Tensor | null = null;
    let out: Float32Array;
    let cols: number;
    try {
      // tidy keeps ONLY the returned tensor alive past the block — every
      // intermediate (q head, masked copy, softmax buffer) dies here.
      picked = tf.tidy(() => {
        const masked = tf.add(this.net.q(obsT), maskT);
        if (mode.kind === "temperature") {
          return tf.softmax(tf.div(masked, Math.max(mode.tau, TAU_FLOOR)));
        }
        return tf.argMax(masked, 1) as tf.Tensor1D;
      });
      // argMax returns [B] (1 column of picks); softmax returns [B, 18].
      cols = picked.shape.length === 2 ? (picked.shape[1] as number) : 1;
      out = (await picked.data()) as Float32Array; // the ONE read, outside the tidy
    } finally {
      // picked first (created last): a rejected read (webgl context loss /
      // download failure) must release it too — the OLD code disposed it only
      // on the success path, leaking one [B,·] tensor per in-flight act the
      // moment the backend died.
      picked?.dispose();
      obsT.dispose();
      maskT.dispose();
    }

    const actions = new Int32Array(n);
    let state = rngState;
    for (let i = 0; i < n; i++) {
      if (mode.kind === "temperature") {
        // Categorical over the masked softmax row: masked entries are exactly
        // 0 (exp(−Inf) = 0), so the walk never lands on an unavailable fire.
        // The last entry catches float drift so a choice is always made.
        let u = rand(state);
        state = nextState(state);
        let a = cols - 1;
        let acc = 0;
        for (let k = 0; k < cols - 1; k++) {
          acc += out[i * cols + k];
          if (u < acc) {
            a = k;
            break;
          }
        }
        actions[i] = a;
      } else if (mode.epsilon > 0) {
        // One threshold draw PER ROW, advanced whether or not it explores:
        // rand() is pure, so skipping the advance on a greedy row would
        // re-draw the SAME value for every later row — the first greedy row
        // would freeze the counter and one coin flip would govern the whole
        // fleet slice (exploration collapses to ~ε/B and becomes perfectly
        // correlated across lanes).
        const u = rand(state);
        state = nextState(state);
        if (u < mode.epsilon) {
          const [a, next] = randInt(state, ACTION_COUNT); // uniform over ALL 18
          state = next;
          actions[i] = a;
        } else {
          actions[i] = out[i]; // masked argmax, an exact integer in float32
        }
      } else {
        actions[i] = out[i]; // masked argmax, an exact integer in float32
      }
    }
    return { actions, state };
  }

  /**
   * One DDQN gradient step on a sampled batch; null when the batch is empty.
   * Returns the Huber loss scalar (read and disposed before returning).
   *
   * Huber (not MSE) is the plan's pinned loss: per-tick rewards are small but
   * the death term is −60 and a kill +50, so a plain squared loss lets one
   * mispredicted terminal tick dominate the whole batch's gradient. Huber
   * quadratics near 0 and linearises past δ=1 — outliers shape the gradient
   * without owning it.
   */
  async trainStep(batch: ReplayBatch): Promise<number | null> {
    const n = batch.size;
    if (n === 0) return null;
    const sT = tf.tensor2d(batch.s, [n, OBS_DIM]);
    const s2T = tf.tensor2d(batch.s2, [n, OBS_DIM]);
    const aT = tf.tensor1d(batch.a, "int32");
    const rT = tf.tensor1d(batch.r);
    const doneT = tf.tensor1d(Float32Array.from(batch.done));
    let y: tf.Tensor1D | null = null;
    let lossT: tf.Scalar | null = null;
    try {
      // TD target — constants for this step (see header), so built once
      // outside the tape and disposed after the step has consumed it.
      y = tf.tidy(() => {
        const nextOnline = this.net.q(s2T);
        // UNMASKED argmax (header): reload is observable, so masking here
        // would bias the bootstrap rather than fix the behaviour policy.
        const aStar = tf.argMax(nextOnline, 1) as tf.Tensor1D;
        const nextTarget = this.target.q(s2T);
        const boot = this.doubleQ
          ? (tf.sum(tf.mul(nextTarget, tf.oneHot(aStar, ACTION_COUNT)), 1) as tf.Tensor1D)
          : (tf.max(nextTarget, 1) as tf.Tensor1D);
        const keep = tf.sub(1, doneT); // (1 − done): a terminal s′ contributes nothing
        return tf.add(rT, tf.mul(tf.mul(this.gamma, keep), boot)) as tf.Tensor1D;
      });
      // Only the online-Q(s) forward is on the tape; `true` keeps the loss
      // scalar for the awaited read. minimize() applies the update before
      // returning (Adam's m/v slots are created on this first call — anything
      // measuring tensors must have stepped at least once already).
      lossT = this.optimizer.minimize(() => {
        const q = this.net.q(sT);
        const taken = tf.sum(tf.mul(q, tf.oneHot(aT, ACTION_COUNT)), 1);
        return tf.losses.huberLoss(y as tf.Tensor1D, taken);
      }, true, this.net.variables());
      if (lossT === null) return null; // no differentiable path — nothing applied
      const loss = (await lossT.data())[0] as number;
      this.stepCount++;
      return loss;
    } finally {
      lossT?.dispose();
      y?.dispose();
      sT.dispose();
      s2T.dispose();
      aT.dispose();
      rT.dispose();
      doneT.dispose();
    }
  }

  /** Gradient steps applied so far — the worker's targetSync trigger. */
  get steps(): number {
    return this.stepCount;
  }

  /**
   * Hard-copy the online weights into the target net. Cloned → assigned →
   * dropped (model.ts setWeights copies FROM its argument, so the clones can
   * be disposed immediately after the call and the target owns its buffers).
   */
  syncTarget(): void {
    const w = this.net.weights();
    this.target.setWeights(w);
    for (const t of w) t.dispose();
  }

  /** Cloned online weights — caller-owned; hand to restore/disposeSnapshot. */
  snapshot(): tf.Tensor[] {
    return this.net.weights();
  }

  /** Copy FROM the snapshot into the online net; the caller keeps ownership. */
  restore(snap: tf.Tensor[]): void {
    this.net.setWeights(snap);
  }

  disposeSnapshot(snap: tf.Tensor[]): void {
    for (const t of snap) t.dispose();
  }

  /**
   * Structural alias closing the tfjs-free TrainingNet contract (scheduler.ts
   * drives any learner through `train`; Dqn's own name is trainStep). The
   * Trainer constructor's `net: TrainingNet` parameter is the compile-time
   * proof this alias keeps the two signatures fused.
   */
  train(batch: ReplayBatch): Promise<number | null> {
    return this.trainStep(batch);
  }

  /** Full teardown: both nets and the optimizer's slot state. Dead afterwards. */
  dispose(): void {
    this.net.dispose();
    this.target.dispose();
    this.optimizer.dispose();
    this.stepCount = 0;
  }
}
