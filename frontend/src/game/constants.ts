// constants.ts — every tunable number in the game, named and cited. DOM-free,
// tfjs-free (this module is imported by the worker engine AND node tests).
//
// WHY one flat module of consts instead of config objects or inline literals:
// the engine, the encoder, the rewards, the renderer, and the property tests
// all quote the same physics, and two copies of a number drift silently — a
// mismatch either breaks seed determinism or makes trained rewards meaningless.
// Units are part of the comment: `*_TIME`/`*_COOLDOWN`/`*_LIFETIME` are seconds,
// `*_TICKS` are ticks, speeds are u/s, decels u/s², rates rad/s, distances u
// (world units ≈ px), HP is health points, rewards are unitless return terms.
//
// Scale origin: the 2020 AIST1110 gym env (TankWar/gym-tankwar/gym_tankwar/
// envs/tank_war.py) ran a 450×350 pygame window at 30 fps with 4 px/frame
// (120 u/s) tanks. This rebuild is that world ×2 — arena 900×700, tank speed
// 240 u/s — so obstacle layouts, sprites and blast radii stay proportional to
// the original while continuous momentum physics gets room to work. Anything
// not derived from the original cites its own reason inline.

// --- world & timing ---------------------------------------------------------

/** Arena width u. 2× the original 450 px window (tank_war.py `window_width`). */
export const ARENA_W = 900;
/** Arena height u. 2× the original 350 px window (`window_height`). */
export const ARENA_H = 700;
/** Fixed simulation rate in ticks/s — the same 30 fps the original loop ran. */
export const TICK_HZ = 30;
/** Seconds per tick. Derived from TICK_HZ, never tuned independently: a dt that
 *  disagrees with the tick counter's wall-clock assumption desyncs every timer
 *  and breaks the fixed-step determinism the replay/tests rely on. */
export const DT = 1 / TICK_HZ;

// --- tank body & motion -----------------------------------------------------

/** Tank collision radius u — a UNIFORM circle for every tank. Deliberately
 *  replaces the original's axis-aligned sprite box that rotated with the hull,
 *  whose corners made bullets "miss" a visually-hit tank (the bug this rebuild
 *  exists to fix). Sprites are drawn larger than this; hitboxes are honest. */
export const TANK_RADIUS = 20;
/** Max forward speed u/s. Original 4 px/frame @30 fps = 120 u/s, ×2. */
export const MAX_SPEED_FWD = 240;
/** Max reverse speed u/s — slower than forward so retreating is strictly
 *  weaker than advancing (keeps episodes moving and gives FFA fights a face). */
export const MAX_SPEED_REV = 140;
/** Throttle acceleration u/s² — 0→240 in 0.5 s: momentum you can feel, no ice rink. */
export const ACCEL = 480;
/** Coast deceleration u/s² (no throttle input): ~0.67 s to a stop from max. */
export const COAST_DECEL = 360;
/** Brake deceleration u/s² (opposite throttle held): hard stop in ~0.34 s, so
 *  braking is always worth an action but cannot stop on a bullet-frame. */
export const BRAKE_DECEL = 700;
/** Hull turn rate rad/s at full steer — a full circle takes ~1.75 s, so aim is
 *  a real commitment and bullet dodging is possible at 30 Hz granularity. */
export const TURN_RATE = 3.6;

// --- bullets ----------------------------------------------------------------

/** Bullet speed u/s. With BULLET_SUBSTEPS=2 a bullet advances 840·DT/2 = 14 u
 *  per sweep — well under a tank's 40 u diameter and any wall thickness, so a
 *  full-speed shot can never tunnel through geometry between substeps. */
export const BULLET_SPEED = 840;
/** Collision substeps per tick for bullets (the only fast-moving bodies). */
export const BULLET_SUBSTEPS = 2;
/** Bullet collision radius u (circle-vs-circle vs tanks, circle-vs-AABB vs cover). */
export const BULLET_RADIUS = 4;
/** Spawn distance from tank centre to muzzle u — just outside TANK_RADIUS so a
 *  freshly fired bullet cannot collide with its own tank on tick 0. */
export const MUZZLE_OFFSET = 24;

// --- fire control & bullet lifetime ----------------------------------------

/** Base seconds between shots — keeps the original's deliberate one-shot pacing. */
export const FIRE_COOLDOWN = 1.0;
/** Seconds between shots while the rapid-fire powerup is active. */
export const RAPID_COOLDOWN = 0.35;
/** Max live bullets per tank (the "magazine"). Caps how much a tank can queue
 *  up behind its own cooldown, which is what makes rapid fire a real powerup. */
export const MAX_BULLETS_PER_TANK = 6;
/** Max live bullets in the whole arena, all tanks combined — bounds the
 *  encoder's BULLETS block (18 slots ≥ 4 tanks × 6) and the O(bullets²)-ish
 *  collision sweep. */
export const MAX_BULLETS_GLOBAL = 24;
/** Seconds a bullet survives before expiring (and docking REWARD_BULLET_EXPIRED):
 *  2.5 s of flight ≈ 2100 u ≈ 1.9 arena diagonals — long enough to matter, short
 *  enough that ricochet clutter clears itself. */
export const BULLET_LIFETIME = 2.5;
/** Ricochets allowed per bullet from spawn (ricochet ×1 is on from game start). */
export const BULLET_BOUNCES = 1;
/** Ricochets allowed while the bounce ("ammo") powerup is active. */
export const BOUNCE_POWERUP = 3;

// --- combat & destructible cover -------------------------------------------

/** Damage per bullet hit. 25 into TANK_HP=100 ⇒ exactly four hits to kill, so
 *  a fight's length is legible from the HP bar. */
export const BULLET_DAMAGE = 25;
/** Starting (and max) tank HP — the original's `starting_hp`. */
export const TANK_HP = 100;
/** Crate HP: three bullet hits (3×25=75 > 60) to break a crate, two don't. */
export const CRATE_HP = 60;
/** Damage to every tank inside a crate's blast (the crate's own punishment for
 *  using it as cover). */
export const CRATE_BLAST_DAMAGE = 20;
/** Crate blast radius u — 3 tank radii: punishes hugging the crate, not shooting it. */
export const CRATE_BLAST_RADIUS = 60;

// --- power-ups ---------------------------------------------------------------

/** Max powerups on the field at once. Matches the encoder's 4 POWERUPS slots:
 *  a fifth spawn is skipped rather than evicting an existing one (deterministic,
 *  no hidden priority). */
export const MAX_POWERUPS_ON_FIELD = 4;
/** Min ticks between powerup spawns (240 = 8 s): scarcity makes pickups worth
 *  the detour. */
export const POWERUP_SPAWN_MIN_TICKS = 240;
/** Max ticks between powerup spawns (360 = 12 s) — the actual gap is uniform in
 *  [min, max] from the episode seed. */
export const POWERUP_SPAWN_MAX_TICKS = 360;
/** Seconds a powerup sits on the floor before despawning (drives despawnFrac in
 *  the observation — urgency the agent can see). */
export const POWERUP_LIFETIME = 20;
/** Rapid-fire duration s (RAPID_COOLDOWN for that long). */
export const RAPID_TIME = 8;
/** Shield duration s — short on purpose: invulnerability is the strongest effect. */
export const SHIELD_TIME = 3;
/** Speed boost duration s (MAX_SPEED × SPEED_MULT). */
export const SPEED_TIME = 8;
/** Bounce-ammo duration s (BULLET_BOUNCES → BOUNCE_POWERUP). */
export const BOUNCE_TIME = 10;
/** Pierce duration s (bullets pass through tanks, once each). */
export const PIERCE_TIME = 10;
/** HP restored by the repair pickup, clamped to TANK_HP. */
export const REPAIR_AMOUNT = 40;
/** Speed multiplier while the speed powerup is active. */
export const SPEED_MULT = 1.4;
/** Turn-rate multiplier while the speed powerup is active — raised too, or the
 *  extra speed would make steering uselessly wide and the pickup a trap. */
export const TURN_MULT = 1.15;

// --- perception (observation geometry) --------------------------------------

/** Ray length u for the 16-ray lidar block: ~1/3 of the arena diagonal, far
 *  enough to see an approaching tank, short enough that rays stay discriminative. */
export const RAY_LEN = 300;
/** Ray count, cast every 22.5° with k=0 pointing along the hull's forward — the
 *  ray fan is body-relative so the policy is heading-invariant by construction. */
export const RAY_COUNT = 16;
/** Observation vector length: SELF 16 + TANKS 3×12 + BULLETS 18×7 + POWERUPS 4×6
 *  + RAYS 16×5 = 282, all values ∈ [−1, 1]. The offset table (which block starts
 *  where) lives in encoding.ts — this constant only pins the total, which is what
 *  the network input layer and every encoding test assert against. */
export const OBS_DIM = 282;

// --- episode structure & escalation ----------------------------------------

/** Escalation begins at this fraction of the episode: SURVIVAL reward doubles
 *  (REWARD_SURVIVAL_ESC_MULT) — nudges passive agents to fight before the
 *  endgame. Powerup spawn cadence deliberately does NOT change here: it stays
 *  on the same [POWERUP_SPAWN_MIN_TICKS, POWERUP_SPAWN_MAX_TICKS] band for the
 *  whole episode, which powerups.spec's seeded cadence run pins, and an
 *  escalation-phase spawn flood would hand the endgame to whoever is already
 *  nearest an icon. */
export const ESCALATION_FRAC = 0.70;
/** Sudden death begins at this fraction of the episode: the arena starts draining
 *  everyone, so a maxed-out staller still loses. */
export const SUDDEN_DEATH_FRAC = 0.85;
/** HP/s drained from every living tank during sudden death. 3 HP/s over the last
 *  15% of a 45 s episode ≈ 20 HP — enough to force a resolution, not enough to
 *  overwrite the fight that happened before it. */
export const SUDDEN_DEATH_DRAIN = 3;
/** Hard episode cap in ticks = 2700 = 90 s @30 Hz, i.e. 2× the default length.
 *  The engine MUST terminate by here (engine.spec asserts it) so a training lane
 *  can never wedge on one episode. */
export const MAX_TICKS_CAP = 2700;
/** Default episode length in seconds. The train config form may shorten it, but
 *  never past MAX_TICKS_CAP. */
export const EPISODE_SECONDS_DEFAULT = 45;

// --- rewards ----------------------------------------------------------------
//
// Descendants of the original gym shaping (tank_war.py): ±1000 kill/death,
// +10/−10 shoot/miss, −0.1·√steps border penalty, 0.1·√steps survival, plus
// 100/distance approach shaping. The rebuild rescales everything so shaping
// NEVER dominates outcomes — a kill totals ≈ +250 (4 hits × 2 dmg × 25 HP + 50),
// death ≈ −210 (−60 + 100 HP of damage taken at −1.5), a full 45 s of survival
// ≈ +27 — while staying per-tick additive (the original's √terms made returns
// depend on episode position in a way replay batches could not average).

/** Reward per HP of damage dealt to an enemy tank. */
export const REWARD_DAMAGE_DEALT = 2;
/** Reward for the finishing blow on a tank (stacks with the damage dealt on that hit). */
export const REWARD_KILL = 50;
/** Reward for the breaking hit on a crate (cover control is worth something, but
 *  far less than hurting a tank). */
export const REWARD_CRATE_BREAK = 5;
/** Reward for collecting any powerup. */
export const REWARD_PICKUP = 8;
/** Extra reward when the repair pickup actually heals missing HP (a full-HP tank
 *  grabbing repair gets the base pickup only, so hoarding hearts is not free). */
export const REWARD_PICKUP_HEAL_BONUS = 4;
/** Survival reward per tick alive. */
export const REWARD_SURVIVAL = 0.02;
/** Survival multiplier during escalation (phase 1): stalling gets relatively
 *  safer to do NOW, fighting gets relatively more urgent. */
export const REWARD_SURVIVAL_ESC_MULT = 2;
/** Survival multiplier during sudden death (phase 2): the drain already punishes
 *  idling in HP; this keeps the return signal consistent with it. */
export const REWARD_SURVIVAL_SD_MULT = 4;
/** Reward per HP of damage taken (negative). */
export const REWARD_DAMAGE_TAKEN = -1.5;
/** Reward for dying (negative). With damage taken it lands ≈ −210 — the original's
 *  −1000 cliff, rescaled so one death cannot wipe out a batch's signal. */
export const REWARD_DEATH = -60;
/** Per-tick penalty while in wall contact. Descendant of the original's border
 *  penalty (−0.1·√steps), now linear and small because it fires every contact
 *  tick — the old one spiked superlinearly with episode length. */
export const REWARD_WALL_CONTACT = -0.05;
/** Penalty when one of this tank's bullets expires without hitting anything.
 *  Descendant of the original's −10 miss penalty: spraying is taxed, but bounded
 *  (a full 6-bullet mag dumped into a wall costs −1.8, less than 1 HP of damage). */
export const REWARD_BULLET_EXPIRED = -0.3;
/** Per-HP penalty for damaging a teammate, allied mode only (FFA has no allies so
 *  the term never fires there). Taxed HARDER than the same damage rewards (−3 vs
 *  +2) so team-killing is never profitable even net of the kill bonus. */
export const REWARD_ALLIED_FIRE = -3;
/** Timeout placement bonus coefficient: on a tick-out the survivor ranked `rank`
 *  of N (by HP) gets +40·(N−rank)/(N−1) — full 40 for the leader, 0 for last.
 *  Without it a timeout is a 0-reward coin flip the value function cannot price. */
export const REWARD_PLACEMENT_MAX = 40;

// --- potential-based shaping (provably policy-neutral) ----------------------
//
// Φ/Ψ enter the return only as γΦ(s′)−Φ(s) / γΨ(s′)−Ψ(s), so they sum to zero
// over any episode and can reshape learning speed without changing the optimal
// policy (Ng et al. 1999 — the reason these are the ONLY shaping terms allowed
// to look at enemy bullet positions).

/** Distance u under which nearby enemy bullets contribute to Φ (danger potential).
 *  ~2× the original engagement distance — tight enough that only real threats
 *  count, wide enough to see a bullet one dodge away. */
export const PHI_DANGER_RADIUS = 160;
/** Weight of the danger potential (negative: standing in a bullet's path is
 *  shaped down, taking cover shaped up — but only differentially). */
export const PHI_DANGER_WEIGHT = -0.5;
/** Weight of the approach potential Ψ (positive: closing on the nearest enemy is
 *  shaped up) — the FFA replacement for the original's 100/distance term, which
 *  blew up near the target and rewarded hugging. */
export const PSI_APPROACH_WEIGHT = 0.3;
/** Arena diagonal u = √(900² + 700²). The single distance normalizer: every
 *  distance feature (encoding) and both potentials divide by it, so no distance
 *  input ever exceeds 1 regardless of arena size. */
export const NORM_DIST = Math.sqrt(ARENA_W * ARENA_W + ARENA_H * ARENA_H);
