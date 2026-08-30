# TankWar frontend — engineering contract (SPEC)

This is the binding contract for `frontend/`: what the simulation is, what the agents
see and want, how the learner and the UI are wired, and what the test/CI gates enforce.
Every number here is cross-checked against the code it names (`src/game/constants.ts`
is the single home of the world's tunables). If code and SPEC disagree, one of them is
a bug — file it.

Architecture in one sentence: **one Web Worker owns everything tfjs** (training lanes,
watch inference, play inference); the main thread is a pure renderer + keyboard input,
and the two talk through an id-keyed, field-presence-discriminated postMessage protocol.

---

## 1. World model

| Constant | Value | Why |
|---|---|---|
| Arena | 900 × 700 u (2× the original 450×350 board) | `ARENA_W`, `ARENA_H` |
| Tick rate | 30 Hz fixed step, `dt = 1/30 s` | `TICK_HZ`, `DT` |
| Tank body | Uniform circle, radius 20 u (fixes the original's rotating sprite-box hitbox) | `TANK_RADIUS` |
| Speed | 240 u/s fwd / 140 u/s rev; ×1.4 with the speed power-up | `MAX_SPEED_*`, `SPEED_MULT` |
| Acceleration | 480 u/s² throttle · 360 coast · 700 brake | `ACCEL`, `COAST_DECEL`, `BRAKE_DECEL` |
| Turn rate | 3.6 rad/s (×1.15 boosted); heading is radians, y-down, clockwise-positive | `TURN_RATE`, `TURN_MULT` |
| Bullets | 840 u/s, radius 4, 2 substeps/tick (14 u/sweep — no tunneling), life 2.5 s | `BULLET_SPEED`, `BULLET_SUBSTEPS`, `BULLET_LIFETIME` |
| Ricochet | ×1 from spawn; ×3 while the bounce-ammo power-up is active | `BULLET_BOUNCES`, `BOUNCE_POWERUP` |
| Damage | 25/hit; tank HP 100 (4 hits to kill); crate HP 60 (3 hits, 2 don't) | `BULLET_DAMAGE`, `TANK_HP`, `CRATE_HP` |
| Crate blast | 20 dmg within 60 u when a crate breaks | `CRATE_BLAST_*` |
| Fire control | 1.0 s cooldown (0.35 rapid); ≤6 shells/tank, ≤24 in the arena | `FIRE_COOLDOWN`, `RAPID_COOLDOWN`, `MAX_BULLETS_*` |
| Power-ups | ≤4 on the field; spawn gap uniform 8–12 s; 20 s floor life; weights rapid .20 / shield .15 / speed .20 / repair .20 / bounce .15 / pierce .10 | `POWERUP_*`, `MAX_POWERUPS_ON_FIELD` |
| Power-up effects | rapid 8 s · shield 3 s · speed 8 s · repair +40 HP (once) · bounce 10 s · pierce 10 s | `RAPID_TIME` … |
| Episode | 45 s default (play is always 45 s), hard cap 90 s = 2700 ticks | `EPISODE_SECONDS_DEFAULT`, `MAX_TICKS_CAP` |
| Escalation | phase 1 at 70% (survival ×2), sudden death at 85% (3 HP/s drain) | `ESCALATION_FRAC`, `SUDDEN_DEATH_*` |

**Step order** (fixed, engine.spec-pinned): capture reward view → tick++ → clear
events/lastRewards/per-tick flags → tanks (turn, accelerate, move, resolve + two
`dt = 0` re-resolves) → timers + fire → bullets (2 substeps) → power-ups (pickup →
spawn → despawn) → crate blasts → deaths → escalation/drain → terminal check
(placement bonus on timeout) → shaped rewards. After `over` is set, `step` is a no-op.

`wallContact` semantics: true for THIS tick only when the tank's motion had to be
corrected (a real push-out, separation or border clamp). A tank parked exactly on a
face at rest does not flag; the engine clears the flag at the top of `step` and ORs it
across the three resolve passes.

**Arena generation** (per episode, from the seed; `src/game/arena.ts`):
18×14 grid of 50 u cells; 5–8 wall blob seeds in the north half, grown 1–3 cells by
random walk, mirrored as a unit; 8–14 crates in 180°-mirrored pairs, Chebyshev ≥ 2
from every wall, every other crate AND from their own mirror (no corner-fused
ramparts); ≥ 100 u clearance from all four quarter spawns (plus the spawn cell and its
4-neighbours kept clear); every spawn→centre line ≥ 60% free (blocking wall blobs are
deleted, worst first); ≥ 85% of free floor in spawn 0's 4-connected component with all
spawns reachable. Retry ladder: seed, seed+1, … (20 tries), then the hand-built
`SAFE_LAYOUT`. Deterministic: same (seed, tankCount) → byte-identical layout.

## 2. Actions

18 discrete actions driving continuous physics:
`a = (throttle·3 + steer)·2 + fire` — throttle {coast, fwd, rev} × steer {left,
straight, right} × fire {no, yes} (`src/game/actions.ts`). Human keys decode through
the same table (`W/S` throttle, `A/D` steer, `Space` fire). Fire-unavailability
(reload/mag) is masked in the behaviour policy only — never in the target max
(reload is observable; masking the target would bias it).

## 3. Observation — `OBS_DIM = 282`, own-heading reference frame

Relative vectors rotated into body coords; bearing = atan2/π; distances normalized by
`NORM_DIST` = √(900² + 700²). All values ∈ [−1, 1] (`src/game/encoding.ts`,
`OFF = { SELF: 0, TANKS: 16, BULLETS: 52, POWERUPS: 178, RAYS: 202 }`):

| Block | Offset | Size | Content |
|---|---|---|---|
| SELF | 0 | 16 | speed, ang.vel, hp, reloadFrac, bullets/6, 5 powerup timers, isTeamMode, alliesAlive, kills, deaths, bulletOverflow, spare |
| TANKS ×3 (dist-sorted, sentinel 0/valid=0) | 16 | 36 | valid, dist, bearing, relHeading, relSpeed, hp, isAlly, reloadFrac, bullets, shield, closingSpeed, kills |
| BULLETS ×18 nearest (sentinel) | 52 | 126 | valid, dist, bearing, velDirRel, speed, ownerCode (me/ally/enemy), bouncesLeft |
| POWERUPS ×4 | 178 | 24 | valid, dist, bearing, typeIdx, isCombat, despawnFrac |
| RAYS ×16 (22.5° apart, k = 0 fwd) | 202 | 80 | dist/300 + one-hot {wall, crate, tank} + hitTeam |
| **Total** | | **282** | |

18 bullet slots < 24 max shells: the block keeps the 18 NEAREST and SELF[14] reports
the overflow. Rays ignore the observer's own hull; corpses (dead tanks, hp ≤ 0 crates)
are transparent.

## 4. Rewards + bootstrap

Per agent, FFA (`src/game/rewards.ts`, constants in `src/game/constants.ts`):

| Term | Value |
|---|---|
| Damage dealt | +2/HP (credited to the NEAREST living enemy of the victim — documented unbiased heuristic) |
| Kill | +50 |
| Crate broken | +5 (nearest tank to the rubble) |
| Pickup | +8 (+4 if repair actually healed) |
| Survival | +0.02/tick alive at tick start (×2 escalation, ×4 sudden death) |
| Damage taken | −1.5/HP (sudden-death drain included) |
| Death | −60 exactly once (`diedThisTick`) |
| Wall contact | −0.05/tick while scraping |
| Expired bullet | −0.3 (owner of a shell that hit nothing) |
| Allied fire | −3/HP (allied mode only) |
| Timeout placement | +40·(N−1−rank)/(N−1), alive tanks ranked by HP, ties by id |

Kill ≈ +250 (4×2×25 + 50), death ≈ −210 (−60 + 100 HP taken), 45 s survival ≈ +27 —
shaping never dominates outcomes. The literal magnitudes are pinned by
`rewards.spec`'s drift-anchor block.

**Potential-based shaping** (the only terms allowed to read enemy bullet geometry):
`γΦ(s′)−Φ(s)` danger (Φ = −0.5 · max(0, 1 − dNearestHostileBullet/160)) and
`γΨ(s′)−Ψ(s)` approach (Ψ = +0.3 · (1 − dNearestEnemy/NORM_DIST)). Ng et al. (1999)
policy-neutrality holds for the γ the terms are built with, so the episode carries its
γ (`GameConfig.gamma` ← `NetConfig.gamma`; default 0.97) and `shapedStep` builds both
terms with it — the learner's discount and the shaping discount cannot drift apart.

**Bootstrap — standard DQN (no negamax; FFA is not pairwise zero-sum):** each agent
stores its own per-tick transitions `(s, a, r, s′, done)` from its own perspective
encoding; target = `r + γ(1−done)·maxQ_target(s′,·)`; DDQN selects a* = argmax
Q_online(s′) UNMASKED; dead agent's episode ends (done = 1), survivors continue.
Transitions complete one tick after their action and carry their own reward (captured
at staging time); a terminal decision (death/timeout) flushes immediately with done = 1
and s′ = s.

## 5. Networks

- **MLP (default)**: 282 → Dense 256 ReLU → Dense 256 ReLU → Dense 18 linear
  (~143k params).
- **Attention (opt-in)**: 26 entity tokens (self + 3 tanks + 18 bullets + 4 power-ups)
  with type embeddings, d_model 64, 2 pre-LN blocks of `multiHeadAttention(4 heads,
  keyDim 16)` + FFN(128) residual; masked mean pool → concat SELF(16) + RAYS(80) →
  Dense 128 → the same shared Dense 18 head. Sentinel embeddings instead of attention
  masks (tfjs version-fragility).
- Huber loss, Adam (lr 3e-4 default). DDQN on by default; target net hard-synced every
  1000 gradient steps (bit-equal copy at construction).

## 6. Scheduler + training loop

One worker, one `setTimeout(0)` tick chain, bounded slices per iteration, in this
order: (1) interactive lanes tick (watch/play, ≤4 catch-up ticks each, never skipped);
(2) ONE `trainer.fleetTick()` — 16 lockstep lanes, ONE fleet forward (≤64 rows × 18)
— followed by the hard target sync and the fire-and-forget eval; (3) ≤2 gradient
steps of the bounded drain; (4) a throttled metrics push (200 ms = the plan's ≤5 Hz).
Queued worker messages are served between iterations.

- Replay: flat Float32Array ring, row = 567 floats (282 + 1 + 1 + 282 + 1), default
  40 000 rows (~91 MB); batch 256; one gradient step per 128 decisions; sample-reuse
  ratio 2.
- ε per completed episode: `max(0.05, 1.0 · 0.995^ep)` — ~600 episodes to the floor;
  exploration is per-row (an ε-greedy call consumes exactly one threshold draw per row
  plus one uniform per explored row; the RNG is a threaded mulberry32 counter, so
  sessions are reproducible and Math.random never appears).
- γ default 0.97 (spans ~33 ticks; a bullet flight is 30).
- Eval every 100 episodes: 24 greedy-vs-random episodes on a weight snapshot with a
  generation guard; protagonist acts ε = 0-greedy, opponents uniform over all 18.
  The eval yields real macrotasks (between games and every 30 in-game ticks) so it can
  never wedge the lanes; a clean eval does NOT restore the snapshot (nothing wrote
  "eval weights"), a crashed one does, and the clone is disposed on every path.
- Loss EMA 0.95/0.05; transactional rebuild (build into locals → dispose old → reset →
  `rebuildGen++`); a rebuild re-deals the watch lane from its own knobs and posts a
  synthetic `episodeEnd` (`reason: "reconfigured"`) for a live play match.

## 7. Protocol (`src/train-protocol.ts`)

Field-presence discriminated (no `type` tag), id-keyed; unsolicited pushes use id 0.

Requests: `{start}` · `{stop}` · `{config}` · `{watch:{tankCount,speedHz}}` ·
`{watchStop}` · `{pause, lane?}` (lane-scoped; absent = both) · `{step}` ·
`{play:{tankCount,teamMode,difficulty,aimAssist}}` · `{input:{throttle,steer,fire,seq}}`
(latest-wins by `seq`, never acked).

Responses/pushes: `{metrics}` · `{arena, lane}` (walls, crates, spawns + seed; once per
episode, and re-pushed on a same-settings watch restart) · `{snapshot, lane}` (plain
objects, coalesced to ≤60/s, carrying the whole coalesced interval's events) ·
`{episodeEnd, lane}` (`winner`, `ranks`, per-rank `kills`/`deaths`, `ticks`,
`reason: elimination | timeout | reconfigured`) · `{error}`. Lane-scoped pushes are
tagged `watch` | `play` at the source and routed per-lane by the client — the human's
match and the spectator duel never cross.

Watch: speed 0.5×–8× is pure pacing (accumulator-banked ticks, ≤4 catch-up per
iteration), snapshot coalescing to the render rate, worker-owned auto-restart after a
1200 ms beat, watch τ = 0.05. Play: real-time 30 Hz, τ = (1−difficulty) + 0.05
softmax over Q (never freezes — masked mass is exactly 0 and decode clamps), aim
assist ≤ 6°/tick hull snap toward the two-iteration intercept point. Teams: `ffa`
(everyone hittable) vs `allied` (human + AIs share team 0, no allied bullet damage —
only crate blasts can hurt a teammate). Training is always FFA.

## 8. Renderer / audio

Main-thread canvas 2D, interpolating entity positions between the last two snapshots
(`drawn = lerp(prev, cur, a)`). Sprites drawn 34×46 (tank_01) / 28×50 (tank_02) facing
up — canvas rotation is heading + 90°; tread animation alternates the A/B frames at
120 ms/frame at full speed. Tank tint atlas built at boot: 4 colours × 2 sprites × 2
frames via multiply + destination-in offscreen canvases. Explosion sheet: 1024², 8×8
grid of 128 px cells, frames `[1,5,9,13,17,21,25,29,33,37]` blitted as sub-rects
(80 u tank blasts, 20 u bullet impacts). Crates draw 46×46 (2 u inset on the honest
50 u physics cell). Juice: screen shake, muzzle flash, trails, hit flash, particles,
ricochet pings.

Audio: WebAudio, gesture-gated `AudioContext` (autoplay policy) with a header sound
chip. Engine loop `playbackRate = 0.7 + 1.1·(|v|/MAX_SPEED_FWD)`; cannon/explosion
SFX; ricochet ping = cannon fire at 2.2× rate through a lowpass; synthesized pickup
blip; music gain 0.15.

## 9. i18n / UI

Bilingual EN + 繁中: `STRINGS = { en, zh } as const` with compiler-enforced key parity
(`StringKey = keyof typeof STRINGS.en`); `{param}` interpolation; `data-i18n` +
document-wide rebind + per-controller `setLang` for dynamic text. Dark/light theme.
`localStorage` only for `tankwar.lang` / `tankwar.theme` — gameplay/training data is
never persisted (session-only by design). Runtime-built DOM: every id/testid is owned
by its controller's section builder; smoke hooks `#app-status`, `window.__engineReady`,
`window.__appReady`, `window.__tickCount`, `window.__inputState`.

## 10. Tests

- `engine.spec` — 2000 seeded random games: containment, no tunneling, caps,
  monotonicity, seed-determinism, termination ≤ 2700 ticks.
- `physics.spec` — reflection law about the AABB face normal (checked against an
  independent v′ = v − 2(v·n)n), accel/coast/brake integration, cooldowns, magazine
  caps, pierce-once, crate 3-hit invariant, raycast exactness.
- `arena.spec` — 500-seed battery: symmetry, structure, flood-fill ≥ 85%, spawn
  clearance, corridor, determinism, adjacent-seed diversity, SAFE_LAYOUT pins.
- `encoding.spec` — OBS_DIM/offsets, bounds, sentinels, rotation math, perspective
  mirror, JSON round-trip.
- `actions` / `rewards` / `powerups` / `protocol` / `scheduler` / `replay` / `model`
  specs — per-design contracts (scheduler drives a scriptable fake net: zero tfjs in
  the learner's test graph).
- `dqn.spec` (the one tfjs spec) — tiny net on CPU: finite loss, numTensors stable,
  target sync bit-equal, DDQN argmax/eval split, ε = 0 / ε = 1 draw accounting.
- `imports.spec` — the bundle-boundary gate: a whitelist walk of `src/**` proving
  `@tensorflow/*` appears ONLY in `workers/train.worker.ts` and `src/nn/**`, and that
  main-side code never imports from `src/nn` at all.
- `learn-sanity.spec` — headless learning check on the CPU backend (opt-in via
  `LEARN_SANITY=1`; auto-skipped in the fast suite).
- Playwright smoke (`test/smoke/`) against the BUILT `dist/` via `vite preview`:
  boot (status-ready ≤ 30 s, zero pageerrors, tabs), play (HUD, synthetic keydown
  advances `__tickCount`, aim-assist toggle), watch (scoreboard, pause freezes,
  speed changes rate).

## 11. CI + deploy

`.github/workflows/deploy.yml` (big-two verbatim): push to `main` (paths
`frontend/**`) → npm ci → unit tests → build (typecheck + wasm copy + bundle) →
static artifact check → Playwright cache + chromium → smoke → upload-pages-artifact →
deploy-pages. Published to `https://anson416.github.io/tankwar/` with a relative base,
so the wasm binaries and worker chunk resolve under the Pages sub-path.

## 12. Attribution

All art/audio is copied byte-for-byte from the original AIST1110 project; the verbatim
per-asset source/license records live in
[`frontend/public/assets/ATTRIBUTION.md`](frontend/public/assets/ATTRIBUTION.md)
(craftpix freebies; opengameart CC0 / CC-BY 3.0; explosion.wav CC-BY-SA 3.0 by Viktor
Hahn). The repo-level summary is in the README. The original Python env, scripts,
`.h5` models and result PNGs under `TankWar/` are untouched upstream work.
