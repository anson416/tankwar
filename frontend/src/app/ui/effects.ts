// effects.ts — the pooled juice layer: particles, muzzle flashes, explosion
// entries, bullet trails, screen shake, tread clocks. Canvas-2D only — it
// receives a context, never owns DOM nodes, and imports nothing from the game
// or tfjs (renderer.ts owns it; main.ts never touches this module directly).
//
// WHY a preallocated POOL (cap 600) instead of fresh objects per burst: a
// 4-tank firefight emits 8–24 particles per hit at 30 Hz; GC churn from
// short-lived objects shows up as frame hitches exactly when the screen is
// busiest. Particles live in one fixed array; spawn claims slot `count++`,
// death swaps the last active particle into the hole — no allocation, no
// compaction, O(1) per particle.
//
// WHY a local mulberry32 instead of ambient randomness: nothing here is
// gameplay, but seeding the generator from the episode's arena seed (reseed(),
// called by the renderer on every {arena} push) makes a replayed episode's
// sparks/shake jitter bit-stable — same episode, same fireworks. An unseeded
// source would make the playback differ from the live run for no benefit.
//
// All ages/ttls are MILLISECONDS accumulated by update(dtMs) — never read off
// the wall clock — so pause/step (the watch lane freezes frame()) freezes the
// juice with it.

export type ParticleKind = "spark" | "ember" | "splinter" | "smoke" | "ring";

/** One pooled particle. `active` is implied by slot order (see update), kept
 *  only so a recycled slot's stale fields can never be drawn. */
interface Particle {
  x: number;
  y: number;
  vx: number; // u/s
  vy: number;
  /** Age and total life, ms. */
  age: number;
  ttl: number;
  /** Half-size u (radius for round kinds, half-edge for rects). */
  size: number;
  /** Per-16.7ms velocity retention — 0.9 ≈ air drag, 1 = ballistic. */
  drag: number;
  rot: number;
  vrot: number; // rad/s
  color: string;
  kind: ParticleKind;
}

/** Data-only explosion request; renderer.ts blits the sheet sub-rects. */
export interface ExplosionFx {
  x: number;
  y: number;
  /** Drawn edge length u — 80 tank/death blasts, 20 bullet impacts (plan). */
  size: number;
  age: number;
  /** Per-animation-frame dwell ms — 60 for tank scale, 40 for bullet scale. */
  msPerFrame: number;
}

/** One muzzle flash: additive cone + glow, 60 ms total (plan §Juice). */
interface MuzzleFx {
  x: number;
  y: number;
  angle: number;
  age: number;
}

/** Trail ring entry for one live bullet: its last ≤3 snapshot positions,
 *  oldest first, plus the owner's team colour. */
interface TrailFx {
  pts: { x: number; y: number }[];
  color: string;
}

/** Trail ring length — three snapshot positions at 30 Hz ≈ 93 u behind a
 *  full-speed shell: enough to read trajectory, short enough not to smear. */
const TRAIL_LEN = 3;

/** Clamp helper (no Math.min/max chains at every call site). */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Local mulberry32 (same generator as game/rng.ts, closure form): one
 *  counter, one closure — the effects layer has no state object to thread. */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TAU = Math.PI * 2;

export class Effects {
  private readonly cap: number;
  private readonly pool: Particle[];
  private count = 0;
  private rng: () => number = mulberry32(0x9e3779b9);

  private explosions: ExplosionFx[] = [];
  private muzzles: MuzzleFx[] = [];
  private trails = new Map<number, TrailFx>();
  /** Per-tank tread animation clock: accumulated ms + current frame. */
  private treads = new Map<number, { acc: number; frame: 0 | 1 }>();

  private shakeMag = 0;
  shakeX = 0;
  shakeY = 0;

  constructor(cap = 600) {
    this.cap = cap;
    this.pool = new Array(cap);
    for (let i = 0; i < cap; i++) {
      this.pool[i] = {
        x: 0, y: 0, vx: 0, vy: 0, age: 0, ttl: 1, size: 1,
        drag: 1, rot: 0, vrot: 0, color: "#fff", kind: "spark",
      };
    }
  }

  /** Re-key the jitter RNG to an episode seed — deterministic replays. */
  reseed(seed: number): void {
    this.rng = mulberry32(seed | 0);
  }

  /** Drop per-entity state that cannot survive an episode boundary (new arena
   *  ⇒ new ids, and stale trails/treads would flash at episode 0 positions). */
  resetEntities(): void {
    this.trails.clear();
    this.treads.clear();
  }

  /**
   * Advance everything by `dtMs`. Ages, integrates, compacts the pool and
   * decays the shake. Frame-rate independence: drag/shake use per-16.7 ms
   * exponents so a 144 Hz monitor and a 30 fps tab decay identically.
   */
  update(dtMs: number): void {
    const step = dtMs / 16.7;
    for (let i = 0; i < this.count; i++) {
      const p = this.pool[i];
      p.age += dtMs;
      if (p.age >= p.ttl) {
        // Swap-with-last compaction — reprocess the imported slot.
        this.count--;
        const last = this.pool[this.count];
        this.pool[this.count] = p;
        this.pool[i] = last;
        i--;
        continue;
      }
      const drag = Math.pow(p.drag, step);
      p.vx *= drag;
      p.vy *= drag;
      p.x += (p.vx * dtMs) / 1000;
      p.y += (p.vy * dtMs) / 1000;
      p.rot += (p.vrot * dtMs) / 1000;
    }
    for (let i = this.explosions.length - 1; i >= 0; i--) {
      const e = this.explosions[i];
      e.age += dtMs;
      // 10 animation frames, then gone (length-independent dwell: msPerFrame).
      if (e.age >= e.msPerFrame * 10) this.explosions.splice(i, 1);
    }
    for (let i = this.muzzles.length - 1; i >= 0; i--) {
      const m = this.muzzles[i];
      m.age += dtMs;
      if (m.age >= 60) this.muzzles.splice(i, 1);
    }
    if (this.shakeMag > 0.05) {
      this.shakeX = (this.rng() * 2 - 1) * this.shakeMag;
      this.shakeY = (this.rng() * 2 - 1) * this.shakeMag;
      this.shakeMag *= Math.pow(0.85, step);
    } else {
      this.shakeMag = 0;
      this.shakeX = 0;
      this.shakeY = 0;
    }
  }

  // --- spawners ---------------------------------------------------------------

  /** Additive gun-flash + 2 smoke puffs at the muzzle. */
  muzzle(x: number, y: number, angle: number): void {
    if (this.muzzles.length < 32) this.muzzles.push({ x, y, angle, age: 0 });
    this.spawn("smoke", x, y, 260, 3, 0.92, "rgba(200, 200, 200, 0.5)", angle, 40);
    this.spawn("smoke", x, y, 220, 2, 0.9, "rgba(200, 200, 200, 0.35)", angle + Math.PI, 25);
  }

  /**
   * Radial burst of `n` additive sparks/embers. `speed` is the mean ejection
   * speed u/s; `spread` (radians) confines the cone around `baseAngle` when
   * finite (ricochet fans), or sprays isotropically when omitted.
   */
  burst(
    x: number, y: number, n: number, color: string, speed = 120, ttl = 420,
    baseAngle?: number, spread = Math.PI, size = 2,
  ): void {
    for (let i = 0; i < n; i++) {
      const a =
        baseAngle === undefined
          ? this.rng() * TAU
          : baseAngle + (this.rng() * 2 - 1) * spread;
      const v = speed * (0.4 + this.rng() * 0.9);
      const kind: ParticleKind = this.rng() < 0.35 ? "ember" : "spark";
      this.spawn(kind, x, y, ttl * (0.6 + this.rng() * 0.7), size, 0.88, color, a, v);
    }
  }

  /** Wooden crate debris — heavier, rotating, brown, longer-lived. */
  splinters(x: number, y: number, n: number): void {
    for (let i = 0; i < n; i++) {
      const a = this.rng() * TAU;
      const v = 60 + this.rng() * 140;
      this.spawn(
        "splinter", x, y, 520 + this.rng() * 300, 2 + this.rng() * 3, 0.92,
        this.rng() < 0.5 ? "#8a6a42" : "#a5824f", a, v,
      );
    }
  }

  /** Expanding stroke ring (powerup spawn/pickup, tank respawn). */
  ring(x: number, y: number, color: string, size = 10): void {
    this.spawn("ring", x, y, 380, size, 1, color, 0, 0);
  }

  /** Queue an explosion blit (see ExplosionFx). */
  explosion(x: number, y: number, size: number): void {
    // Bullet-scale impacts animate faster (40 ms) than tank-scale blasts (60).
    this.explosions.push({ x, y, size, age: 0, msPerFrame: size >= 40 ? 60 : 40 });
  }

  /** Record a bullet position into its trail ring. Called per FRAME with the
   *  RENDERED (interpolated) position — noting raw snapshot positions would let
   *  the ring lead the drawn shell by up to one interpolation interval (~28 u
   *  at full speed) and read backwards. */
  noteBullet(id: number, x: number, y: number, color: string): void {
    let t = this.trails.get(id);
    if (!t) {
      t = { pts: [], color };
      this.trails.set(id, t);
    }
    t.color = color;
    t.pts.push({ x, y });
    if (t.pts.length > TRAIL_LEN) t.pts.shift();
  }

  /** Screen-shake impulse (u). Magnitudes per plan §Juice: hit 3, crate 5,
   *  kill/death 8 — additive but capped so a three-tank pileup stays playable. */
  addShake(mag: number): void {
    this.shakeMag = Math.min(14, this.shakeMag + mag);
  }

  /**
   * Tread animation clock for one tank. Period = 120 ms × 240/|v| — the plan's
   * 120 ms/frame at full speed (240 u/s), proportionally slower when crawling,
   * clamped to [60, 400] ms so extremes stay readable. A stationary tank
   * (|v| < 8) freezes its current frame (idle tanks don't grind their treads).
   */
  treadFrame(id: number, speed: number, dtMs: number): 0 | 1 {
    let t = this.treads.get(id);
    if (!t) {
      t = { acc: 0, frame: 0 };
      this.treads.set(id, t);
    }
    const v = Math.abs(speed);
    if (v >= 8) {
      const period = clamp((120 * 240) / v, 60, 400);
      t.acc += dtMs;
      while (t.acc >= period) {
        t.acc -= period;
        t.frame = t.frame === 0 ? 1 : 0;
      }
    }
    return t.frame;
  }

  /** Drop trail/tread entries for ids no longer in the snapshot. */
  syncEntities(tankIds: number[], bulletIds: number[]): void {
    for (const id of this.trails.keys()) {
      if (!bulletIds.includes(id)) this.trails.delete(id);
    }
    for (const id of this.treads.keys()) {
      if (!tankIds.includes(id)) this.treads.delete(id);
    }
  }

  /** Active explosion entries (renderer blits sheet frames from them). */
  getExplosions(): readonly ExplosionFx[] {
    return this.explosions;
  }

  // --- drawing ------------------------------------------------------------------

  /** Trail rings: per-position dots at alpha .35/.2/.1 (newest → oldest) over
   *  a fading connector polyline. Drawn UNDER the bullet sprites. */
  drawTrails(ctx: CanvasRenderingContext2D): void {
    for (const t of this.trails.values()) {
      const n = t.pts.length;
      if (n === 0) continue;
      const alphas = [0.35, 0.2, 0.1];
      if (n >= 2) {
        ctx.strokeStyle = t.color;
        ctx.lineWidth = 2;
        for (let i = 1; i < n; i++) {
          // Segment i (pts[i-1]→pts[i]) fades like its NEWER endpoint — the
          // trail brightens toward the shell, never behind it.
          ctx.globalAlpha = alphas[Math.min(n - 1 - i, 2)] * 0.8;
          ctx.beginPath();
          ctx.moveTo(t.pts[i - 1].x, t.pts[i - 1].y);
          ctx.lineTo(t.pts[i].x, t.pts[i].y);
          ctx.stroke();
        }
      }
      ctx.fillStyle = t.color;
      for (let i = 0; i < n; i++) {
        ctx.globalAlpha = alphas[Math.min(n - 1 - i, 2)];
        ctx.beginPath();
        ctx.arc(t.pts[i].x, t.pts[i].y, 3, 0, TAU);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  /** Muzzle flashes — additive "lighter": a bright core glow plus a short cone
   *  along the shot direction; 60 ms fadeout. */
  drawMuzzles(ctx: CanvasRenderingContext2D): void {
    if (this.muzzles.length === 0) return;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const m of this.muzzles) {
      const k = 1 - m.age / 60; // 1 → 0
      ctx.globalAlpha = k;
      ctx.fillStyle = "#ffd9a0";
      ctx.beginPath();
      ctx.arc(m.x, m.y, 7 * k + 2, 0, TAU);
      ctx.fill();
      ctx.fillStyle = "#fff2d0";
      const len = 16 * k + 4;
      const w = 4 * k + 1.5;
      // Cone as a triangle: apex at the muzzle, base ahead — rotate-free via
      // the unit vector (cheap and exact for 2 points).
      const ux = Math.cos(m.angle);
      const uy = Math.sin(m.angle);
      ctx.beginPath();
      ctx.moveTo(m.x, m.y);
      ctx.lineTo(m.x + ux * len - uy * w, m.y + uy * len + ux * w);
      ctx.lineTo(m.x + ux * len + uy * w, m.y + uy * len - ux * w);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  /** All live particles. Sparks/embers draw additively ("lighter") in a second
   *  pass so their glow stacks without brightening splinters/smoke; state save/
   *  restore brackets the whole batch, not each particle. */
  drawParticles(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    for (let i = 0; i < this.count; i++) {
      const p = this.pool[i];
      const k = 1 - p.age / p.ttl;
      if (p.kind === "spark" || p.kind === "ember") continue; // additive pass below
      ctx.globalAlpha = k;
      if (p.kind === "splinter") {
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size, -p.size * 0.4, p.size * 2, p.size * 0.8);
        ctx.restore();
      } else if (p.kind === "smoke") {
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (2.5 - k * 1.5), 0, TAU);
        ctx.fill();
      } else if (p.kind === "ring") {
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (1 + (1 - k) * 3), 0, TAU);
        ctx.stroke();
      }
    }
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < this.count; i++) {
      const p = this.pool[i];
      if (p.kind !== "spark" && p.kind !== "ember") continue;
      const k = 1 - p.age / p.ttl;
      ctx.globalAlpha = k;
      ctx.strokeStyle = ctx.fillStyle = p.color;
      if (p.kind === "spark") {
        // Streak along the velocity vector — the spark IS its motion blur.
        ctx.lineWidth = p.size;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - (p.vx * 22) / 1000, p.y - (p.vy * 22) / 1000);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, TAU);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  /** Claim a pool slot (drop the particle when full — losing one spark of a
   *  24-burst is invisible; growing the pool past cap is a real hazard). */
  private spawn(
    kind: ParticleKind, x: number, y: number, ttlMs: number,
    size: number, drag: number, color: string, angle: number, speed: number,
  ): void {
    if (this.count >= this.cap) return;
    const p = this.pool[this.count++];
    p.kind = kind;
    p.x = x;
    p.y = y;
    p.vx = Math.cos(angle) * speed;
    p.vy = Math.sin(angle) * speed;
    p.age = 0;
    p.ttl = ttlMs;
    p.size = size;
    p.drag = drag;
    p.rot = this.rng() * TAU;
    p.vrot = (this.rng() * 2 - 1) * 12;
    p.color = color;
  }
}
