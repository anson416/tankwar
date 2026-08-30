// renderer.ts — the main-thread scene painter. Canvas 2D only; the whole
// src/app tree stays free of tensor-library imports (the worker owns all of
// that — a unit-test grep enforces the boundary) and DOM-free apart from the
// canvas handed to the constructor.
//
// Data flow: the worker pushes {arena} once per episode → setArena() caches
// merged geometry; then {snapshot} at 30 Hz (coalesced at 8× watch speed) →
// applySnapshot() stores it, drains the tick's juice events into the Effects
// pool and records bullet trail rings; frame(now) paints at rAF rate,
// interpolating entity positions between the last two snapshots.
//
// INTERPOLATION CONTRACT: drawn = lerp(prev, cur, a) with
// a = (now − curArrival) / arrivalInterval, clamped [0, 1]. The transition
// prev→cur therefore plays out during the interval AFTER `cur` arrives — the
// classic one-snapshot render delay (≈33 ms at 30 Hz, one tick of the sim
// itself). Deliberately NO extrapolation past the newest snapshot: a stalled
// or coalesced stream (pause, 0.5× watch speed) would overshoot into positions
// that were never simulated, and the artefacts would read as engine bugs.
//
// HEADING CONVENTION (verified before writing, plan §World + types.ts): the
// engine's forward = (cos h, sin h) in y-down screen space, so h = 0 points
// EAST and increasing h turns clockwise. Canvas rotate(θ) maps a local point
// (x, y) to (x·cosθ − y·sinθ, x·sinθ + y·cosθ); in y-down space positive θ
// appears CLOCKWISE. The sprite art faces UP, i.e. local forward (0, −1),
// which rotate(θ) maps to world (sinθ, −cosθ). Requiring (sinθ, −cosθ) =
// (cos h, sin h) gives θ = h + π/2. Sanity checks: h = 0 → art rotated 90°
// clockwise → faces east ✓; h = π/2 (south) → 180° → faces down ✓. This is
// the plan's "canvas rotation = heading + 90°, PNGs face up". Bullets reuse
// the identical convention (physics.ts ricochet reflects `dir` in the same
// frame), so one derivation covers both.

import { CELL, GRID_COLS } from "../../game/arena";
import {
  ARENA_H,
  ARENA_W,
  BULLET_BOUNCES,
  BULLET_DAMAGE,
  CRATE_HP,
  TANK_HP,
} from "../../game/constants";
import type { Phase, PowerUpKind } from "../../game/types";
import type { ArenaMsg, Snapshot, SnapshotBullet, SnapshotTank } from "../../game/snapshot";
import { Effects } from "./effects";
import { EXPLOSION_FRAMES, TANK_COLORS, bgOverlay, explosionSrc, loadSprites, type Sprite, type SpriteAtlas } from "./sprites";
import { currentTheme, type Theme } from "./theme";

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A crate the renderer tracks: ArenaMsg carries only the INITIAL hp, and
 *  snapshots don't ship crate state at all, so damage is accumulated here from
 *  juice events (see processEvents) — a render-side approximation, good enough
 *  for crack overlays, never read back by the sim. */
interface CrateView extends Rect {
  hp: number;
}

/** Plan §World: tank_01 drawn 34×46, tank_02 28×50 (PNGs face up). */
const TANK_DRAW: readonly [number, number][] = [
  [34, 46],
  [28, 50],
];
/** Shell draw sizes [big bullet_01, small bullet_02] u — elongated along the
 *  barrel like the art, small next to a 40 u hull. */
const BULLET_DRAW: readonly [number, number][] = [
  [10, 24],
  [7, 18],
];
/** Crates are full 50 u cells physically (arena.ts); the visual is inset 2 u
 *  per side (46×46) so cover reads as an object, not a wall segment. */
const CRATE_INSET = 2;
/** Powerup disc radius u. */
const POWERUP_R = 13;

/** Powerup accent colours — also the pickup/spawn particle tints. */
const POWERUP_COLORS: Record<PowerUpKind, string> = {
  rapid: "#e3b341",
  shield: "#58a6ff",
  speed: "#4bd1ca",
  repair: "#ff7b72",
  bounce: "#f2a65a",
  pierce: "#8957e5",
};

/** Obstacle palette per theme (mirrors the style.css tokens). */
const WALL_STYLE: Record<Theme, { fill: string; edge: string; grid: string }> = {
  dark: { fill: "#39424e", edge: "#4d5a6b", grid: "rgba(0, 0, 0, 0.28)" },
  light: { fill: "#c3cad4", edge: "#9aa4b1", grid: "rgba(31, 35, 40, 0.18)" },
};

const TAU = Math.PI * 2;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Shortest-arc angular lerp — headings wrap at ±π and a naive lerp would
 *  spin a tank the long way when it crosses the wrap. */
function lerpAngle(a: number, b: number, t: number): number {
  let d = (b - a) % TAU;
  if (d > Math.PI) d -= TAU;
  if (d < -Math.PI) d += TAU;
  return a + d * t;
}

/** Stateless 01-hash — per-crate crack jitter must be stable across frames and
 *  must NOT consume the effects rng stream (which drives shake/particles and
 *  would couple the two visuals). */
function hash01(n: number): number {
  let t = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
  t = Math.imul(t ^ (t >>> 13), 0xc2b2ae35);
  return ((t ^ (t >>> 16)) >>> 0) / 4294967296;
}

/**
 * Merge per-cell walls (arena.ts emits one 50×50 Obstacle per cell) into
 * maximal-ish rectangles: greedy width-then-height run merge over the cell
 * grid. Fewer draw calls, and — the real reason — interior seams between
 * adjacent cells stop rendering as grid lines through one wall. Non-cell
 * geometry (defensive: the fallback layout is cell-aligned today) passes
 * through untouched.
 */
function mergeWalls(walls: Rect[]): Rect[] {
  const occ = new Set<number>();
  const irregular: Rect[] = [];
  for (const w of walls) {
    if (w.w === CELL && w.h === CELL && w.x % CELL === 0 && w.y % CELL === 0) {
      occ.add((w.y / CELL) * GRID_COLS + w.x / CELL);
    } else {
      irregular.push(w);
    }
  }
  const has = (c: number, r: number): boolean => occ.has(r * GRID_COLS + c);
  const seen = new Set<number>();
  const slabs: Rect[] = [];
  for (const key of occ) {
    if (seen.has(key)) continue;
    const c0 = key % GRID_COLS;
    const r0 = (key / GRID_COLS) | 0;
    let c1 = c0;
    while (has(c1 + 1, r0) && !seen.has(r0 * GRID_COLS + c1 + 1)) c1++;
    let r1 = r0;
    grow: while (has(c0, r1 + 1)) {
      for (let c = c0; c <= c1; c++) {
        if (!has(c, r1 + 1) || seen.has((r1 + 1) * GRID_COLS + c)) break grow;
      }
      r1++;
    }
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) seen.add(r * GRID_COLS + c);
    }
    slabs.push({ x: c0 * CELL, y: r0 * CELL, w: (c1 - c0 + 1) * CELL, h: (r1 - r0 + 1) * CELL });
  }
  return [...slabs, ...irregular];
}

export class SceneRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private atlas: SpriteAtlas | null = null;
  private loadError: string | null = null;
  private readonly fx = new Effects();

  private arena: ArenaMsg | null = null;
  private slabs: Rect[] = [];
  private crates: CrateView[] = [];

  private prev: Snapshot | null = null;
  private cur: Snapshot | null = null;
  private prevAt = 0;
  private curAt = 0;

  /** Owner id → sprite/kind: snapshots don't type bullets, the shooter does. */
  private ownerInfo = new Map<number, { sprite: 0 | 1; colorIdx: number }>();
  /** Tank id → wall-clock ms the white hit flash ends. */
  private flashUntil = new Map<number, number>();
  private phase: Phase = 0;
  /** Wall-clock ms of the last "phase" event — drives the vignette flash. */
  private phaseFlashAt = -1e9;
  private lastFrameAt = 0;
  /** Visibility hint from the owning controller (ArenaRenderer seam): a hidden
   *  tab still receives snapshots (the lane runs headless), so skip painting —
   *  but keep the effects clock ticking so queued bursts age out instead of
   *  all popping the moment the tab is shown. */
  private visible = true;
  /** Visual clock (ms, advanced only by frame()) — pulse/blink phases freeze
   *  with the lane on pause instead of drifting on the wall clock. */
  private clock = 0;

  /** Pre-dimmed floor art, rebuilt on theme flip / new arena (1000² source
   *  resampled + 2 fills ONCE, not per frame). */
  private bg: HTMLCanvasElement | null = null;
  private bgTheme: Theme | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d context unavailable");
    this.ctx = ctx;
    void loadSprites().then(
      (atlas) => {
        this.atlas = atlas;
        this.bg = null; // rebuild now that the floor art exists
      },
      (err: unknown) => {
        this.loadError = err instanceof Error ? err.message : String(err);
      },
    );
  }

  /** Cache the episode's static world. Called once per episode, before the
   *  first snapshot. `seed` re-keys the effects rng so a replayed episode
   *  jitters/shakes identically. */
  setArena(a: ArenaMsg): void {
    this.arena = a;
    this.slabs = mergeWalls(a.walls);
    this.crates = a.crates.map((c) => ({ x: c.x, y: c.y, w: c.w, h: c.h, hp: c.hp }));
    this.fx.reseed(a.seed);
    this.fx.resetEntities();
    this.flashUntil.clear();
    this.prev = this.cur = null;
    this.bg = null;
  }

  /** Absorb one tick: shift the interpolation pair, refresh owner/trail state,
   *  and turn the tick's events into juice. Events are drained HERE (not in
   *  frame()) so a coalesced 8× stream still fires every burst exactly once. */
  applySnapshot(s: Snapshot): void {
    const now = performance.now();
    this.prev = this.cur;
    this.prevAt = this.curAt;
    this.cur = s;
    this.curAt = now;
    if (this.prev && this.prev.ep !== s.ep) {
      // New episode: ids are only unique WITHIN an episode (types.ts), so every
      // per-id structure must reset or stale entities flash at new spawns.
      this.prev = null;
      this.fx.resetEntities();
      this.flashUntil.clear();
    }
    this.phase = s.phase;

    this.ownerInfo.clear();
    for (const t of s.tanks) this.ownerInfo.set(t.i, { sprite: t.sprite, colorIdx: t.colorIdx });

    // Trails are noted from the RENDERED positions in drawBullets (so they can
    // never lead the interpolated sprite); only the pruning happens here.
    this.fx.syncEntities(
      s.tanks.map((t) => t.i),
      s.bullets.map((b) => b.i),
    );
    this.processEvents(s, now);
  }

  /** Visibility hint (ArenaRenderer seam): stop painting while the tab is
   *  hidden; the effects clock keeps running so nothing pops on return. */
  setVisible(v: boolean): void {
    this.visible = v;
  }

  /** Paint one frame at wall-clock `now` (performance.now ms). `alphaHint`
   *  overrides the derived interpolation fraction (controllers that know the
   *  lane's true cadence — e.g. a fixed 0.5× watch speed — may pin it). */
  frame(now: number, alphaHint?: number): void {
    const dt = clamp(now - this.lastFrameAt, 0, 100);
    this.lastFrameAt = now;
    this.clock += dt;
    this.fx.update(dt);
    if (!this.visible) return;

    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const theme = currentTheme();

    if (!this.atlas || this.loadError) {
      this.paintLetterbox(theme);
      if (this.loadError) this.paintCenterText(`asset error: ${this.loadError}`);
      return;
    }
    if (!this.resizeBackingStore()) return; // canvas display:none / zero-sized

    this.paintLetterbox(theme);
    const sc = Math.min(this.canvas.width / ARENA_W, this.canvas.height / ARENA_H);
    const ox = (this.canvas.width - ARENA_W * sc) / 2;
    const oy = (this.canvas.height - ARENA_H * sc) / 2;
    ctx.setTransform(sc, 0, 0, sc, ox, oy);

    // Screen-shake wrapper: the whole scene (floor included) translates; the
    // letterbox margin does not, so shake reads as camera motion, not resize.
    ctx.translate(this.fx.shakeX, this.fx.shakeY);

    this.drawBackground(theme);
    if (!this.arena) return;
    this.drawObstacles(theme);
    if (!this.cur) return; // arena pushed, first snapshot still in flight

    const alpha = alphaHint !== undefined ? clamp(alphaHint, 0, 1) : this.deriveAlpha(now);
    this.fx.drawTrails(ctx);
    this.drawPowerups();
    this.drawBullets(alpha);
    this.drawTanks(now, dt, alpha);
    this.drawExplosions();
    this.fx.drawMuzzles(ctx);
    this.fx.drawParticles(ctx);
    this.drawVignette(now);
  }

  // --- snapshot processing -----------------------------------------------------

  private deriveAlpha(now: number): number {
    if (!this.prev || this.prev === this.cur) return 1;
    return clamp((now - this.curAt) / Math.max(this.curAt - this.prevAt, 1), 0, 1);
  }

  private processEvents(s: Snapshot, now: number): void {
    // Kill positions first: an "explode" at a death spot is a tank blast (the
    // kill event already contributed shake 8), elsewhere it is a crate blast.
    const kills: { x: number; y: number }[] = [];
    for (const e of s.events) {
      if (e.kind === "kill") kills.push({ x: e.x, y: e.y });
    }
    const nearKill = (x: number, y: number): boolean =>
      kills.some((k) => Math.hypot(k.x - x, k.y - y) < 30);

    for (const e of s.events) {
      switch (e.kind) {
        case "fire":
          this.fx.muzzle(e.x, e.y, e.angle ?? 0);
          break;
        case "hit": {
          // Bullet impact: 20 u sheet frame + 8 victim-coloured sparks (plan
          // §Juice counts) + shake 3 + the victim's white flash window.
          const info = e.tank !== undefined ? this.ownerInfo.get(e.tank) : undefined;
          const color = TANK_COLORS[info?.colorIdx ?? 0];
          this.fx.explosion(e.x, e.y, 20);
          this.fx.burst(e.x, e.y, 8, color, 130, 380, undefined, Math.PI, 2);
          this.fx.addShake(3);
          if (e.tank !== undefined) this.flashUntil.set(e.tank, now + 90);
          break;
        }
        case "kill":
          // Death itself: shake 8 + a longer flash. The blast visuals belong to
          // the companion "explode" event (always emitted by killTank), so the
          // 80 u fireball draws exactly once.
          if (e.tank !== undefined) this.flashUntil.set(e.tank, now + 160);
          this.fx.addShake(8);
          break;
        case "explode":
          this.fx.explosion(e.x, e.y, 80);
          this.fx.burst(e.x, e.y, 24, "#ffb347", 180, 520, undefined, Math.PI, 3);
          if (!nearKill(e.x, e.y)) this.fx.addShake(5); // crate blast, not a death
          break;
        case "crate":
          // Crate BROKEN (physics.ts emits this only on the killing hit).
          this.fx.explosion(e.x, e.y, 80);
          this.fx.splinters(e.x, e.y, 14);
          this.fx.addShake(5);
          this.breakCrateAt(e.x, e.y);
          break;
        case "pickup": {
          const color = e.powerup ? POWERUP_COLORS[e.powerup] : "#ffffff";
          this.fx.ring(e.x, e.y, color, 12);
          this.fx.burst(e.x, e.y, 12, color, 110, 420, undefined, Math.PI, 2);
          break;
        }
        case "ricochet": {
          // Sparks fan along the REFLECTED direction (physics emits post-bounce
          // dir). Also the only crate-damage signal that exists renderer-side:
          // a crate hit that does NOT break emits no event of its own, but the
          // bullet still ricochets off it (or dies there when out of bounces).
          this.fx.burst(e.x, e.y, 6, "#ffe9a8", 150, 300, e.angle ?? 0, 0.9, 1.6);
          this.damageCrateAt(e.x, e.y);
          break;
        }
        case "spawn": {
          const color = e.powerup ? POWERUP_COLORS[e.powerup] : "#ffffff";
          this.fx.ring(e.x, e.y, color, 10);
          this.fx.burst(e.x, e.y, 5, color, 70, 350, undefined, Math.PI, 1.5);
          break;
        }
        case "phase":
          this.phaseFlashAt = now;
          break;
      }
    }
  }

  /** Mark the crate nearest a break event as gone (cosmetic only). */
  private breakCrateAt(x: number, y: number): void {
    let best: CrateView | null = null;
    let bestD = 40;
    for (const c of this.crates) {
      if (c.hp <= 0) continue;
      const d = Math.hypot(c.x + c.w / 2 - x, c.y + c.h / 2 - y);
      if (d < bestD) {
        best = c;
        bestD = d;
      }
    }
    if (best) best.hp = 0;
  }

  /** Credit BULLET_DAMAGE to a live crate whose padded bounds contain a
   *  ricochet point — drives the 2-stage crack overlay. */
  private damageCrateAt(x: number, y: number): void {
    for (const c of this.crates) {
      if (c.hp <= 0) continue;
      if (x >= c.x - 10 && x <= c.x + c.w + 10 && y >= c.y - 10 && y <= c.y + c.h + 10) {
        c.hp = Math.max(0, c.hp - BULLET_DAMAGE);
        return;
      }
    }
  }

  // --- layers --------------------------------------------------------------------

  private resizeBackingStore(): boolean {
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    if (cssW < 4 || cssH < 4) return false;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(cssW * dpr);
    const h = Math.round(cssH * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return true;
  }

  private paintLetterbox(theme: Theme): void {
    this.ctx.fillStyle = theme === "dark" ? "#0e1116" : "#f6f8fa";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private paintCenterText(msg: string): void {
    const { ctx } = this;
    ctx.save();
    ctx.fillStyle = "#8b949e";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(msg, this.canvas.width / 2, this.canvas.height / 2);
    ctx.restore();
  }

  /** Floor art: background.png cover-scaled, crushed to 25% brightness, then a
   *  theme tint — composed ONCE into an offscreen canvas (a per-frame 1000²
   *  resample + composite would dominate the frame budget). */
  private drawBackground(theme: Theme): void {
    if (!this.atlas) return;
    if (!this.bg || this.bgTheme !== theme) {
      const c = document.createElement("canvas");
      c.width = ARENA_W;
      c.height = ARENA_H;
      const g = c.getContext("2d");
      if (g) {
        // cover-fit the square 1000² source over the 900×700 arena
        const sc = Math.max(ARENA_W / this.atlas.background.width, ARENA_H / this.atlas.background.height);
        const dw = this.atlas.background.width * sc;
        const dh = this.atlas.background.height * sc;
        g.drawImage(this.atlas.background, (ARENA_W - dw) / 2, (ARENA_H - dh) / 2, dw, dh);
        g.fillStyle = "rgba(0, 0, 0, 0.75)"; // plan: 25% brightness under the art
        g.fillRect(0, 0, ARENA_W, ARENA_H);
        g.fillStyle = bgOverlay(theme);
        g.fillRect(0, 0, ARENA_W, ARENA_H);
      }
      this.bg = c;
      this.bgTheme = theme;
    }
    const ctx = this.ctx;
    ctx.drawImage(this.bg, 0, 0);
    ctx.strokeStyle = theme === "dark" ? "#2a313b" : "#d0d7de";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, ARENA_W - 2, ARENA_H - 2);
  }

  private drawObstacles(theme: Theme): void {
    const ctx = this.ctx;
    const style = WALL_STYLE[theme];

    // Walls: merged slabs + a 3 u top-edge highlight so the board reads lit
    // from above (matches the sprites' top-down lighting).
    for (const s of this.slabs) {
      ctx.fillStyle = style.fill;
      ctx.fillRect(s.x, s.y, s.w, s.h);
      ctx.fillStyle = style.edge;
      ctx.fillRect(s.x, s.y, s.w, 3);
      ctx.fillStyle = style.grid;
      ctx.fillRect(s.x, s.y + s.h - 3, s.w, 3);
    }

    // Crates: 46×46 inset wooden block; crack overlay in 2 damage stages
    // (hp > ⅔ clean, > ⅓ hairline cracks, ≤ ⅓ splintered).
    for (let i = 0; i < this.crates.length; i++) {
      const c = this.crates[i];
      if (c.hp <= 0) continue;
      const x = c.x + CRATE_INSET;
      const y = c.y + CRATE_INSET;
      const w = c.w - CRATE_INSET * 2;
      const h = c.h - CRATE_INSET * 2;
      ctx.fillStyle = "#8a6a42";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#6d5233";
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
      // plank cross-braces
      ctx.strokeStyle = "rgba(60, 42, 20, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x + 4, y + 4);
      ctx.lineTo(x + w - 4, y + h - 4);
      ctx.moveTo(x + w - 4, y + 4);
      ctx.lineTo(x + 4, y + h - 4);
      ctx.stroke();

      const frac = clamp(c.hp / CRATE_HP, 0, 1);
      const stage = frac > 2 / 3 ? 0 : frac > 1 / 3 ? 1 : 2;
      if (stage > 0) {
        const n = stage === 1 ? 2 : 5;
        ctx.strokeStyle = "rgba(35, 22, 10, 0.8)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let j = 0; j < n; j++) {
          // Jagged 3-point crack from a hash-picked perimeter point toward the
          // centre — hash01 (not the shared rng) keeps cracks frame-stable.
          const a = hash01(i * 31 + j) * TAU;
          let px = x + w / 2 + Math.cos(a) * (w / 2 - 3);
          let py = y + h / 2 + Math.sin(a) * (h / 2 - 3);
          ctx.moveTo(px, py);
          for (let k = 1; k <= 3; k++) {
            const t = k / 3;
            const ja = (hash01(i * 31 + j * 7 + k) - 0.5) * 1.4;
            px = lerp(px, x + w / 2, t) + Math.cos(ja) * 4;
            py = lerp(py, y + h / 2, t) + Math.sin(ja) * 4;
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
        if (stage === 2) {
          ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
          ctx.fillRect(x, y, w, h);
        }
      }
    }
  }

  /** Powerups: tinted disc + vector glyph (heart.png verbatim for repair).
   *  Pulsed on the visual clock — SnapshotPowerUp carries NO despawn field
   *  (snapshot.ts keeps the row minimal), so the plan's "blink in the last 3 s
   *  of lifetime" has no honest data to drive it; when the protocol grows a
   *  despawn field, gate the blink on it. */
  private drawPowerups(): void {
    if (!this.cur) return;
    const ctx = this.ctx;
    const pulse = 1 + 0.08 * Math.sin(this.clock / 300);
    for (const pu of this.cur.powerups) {
      const color = POWERUP_COLORS[pu.kind];
      const r = POWERUP_R * pulse;
      ctx.globalAlpha = 0.26;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pu.x, pu.y, r, 0, TAU);
      ctx.fill();
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pu.x, pu.y, r, 0, TAU);
      ctx.stroke();
      if (pu.kind === "repair" && this.atlas) {
        // Repair is the one glyph that is an asset: heart.png at 18 u.
        ctx.drawImage(this.atlas.heart, pu.x - 9, pu.y - 9, 18, 18);
      } else {
        this.drawGlyph(pu.kind, pu.x, pu.y);
      }
    }
    ctx.globalAlpha = 1;
  }

  /** White vector glyphs on the tinted disc, centred on (x, y). */
  private drawGlyph(kind: PowerUpKind, x: number, y: number): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(x, y);
    ctx.strokeStyle = "#f5f7fa";
    ctx.fillStyle = "#f5f7fa";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    switch (kind) {
      case "rapid": // double chevron — faster cycling
        ctx.moveTo(-6, -5);
        ctx.lineTo(-1, 0);
        ctx.lineTo(-6, 5);
        ctx.moveTo(0, -5);
        ctx.lineTo(5, 0);
        ctx.lineTo(0, 5);
        ctx.stroke();
        break;
      case "shield": // shield outline
        ctx.moveTo(0, -7);
        ctx.quadraticCurveTo(6, -6, 6, -1);
        ctx.quadraticCurveTo(6, 5, 0, 7);
        ctx.quadraticCurveTo(-6, 5, -6, -1);
        ctx.quadraticCurveTo(-6, -6, 0, -7);
        ctx.stroke();
        break;
      case "speed": // lightning bolt
        ctx.moveTo(2, -7);
        ctx.lineTo(-4, 1);
        ctx.lineTo(0, 1);
        ctx.lineTo(-2, 7);
        ctx.lineTo(4, -1);
        ctx.lineTo(0, -1);
        ctx.closePath();
        ctx.fill();
        break;
      case "bounce": // ricochet zig-zag with arrowhead
        ctx.moveTo(-6, 5);
        ctx.lineTo(-2, -1);
        ctx.lineTo(2, 3);
        ctx.lineTo(6, -4);
        ctx.moveTo(6, -4);
        ctx.lineTo(2.4, -3.4);
        ctx.moveTo(6, -4);
        ctx.lineTo(5, -0.4);
        ctx.stroke();
        break;
      case "pierce": // arrow through a hull ring
        ctx.arc(0, 0, 3.2, 0, TAU);
        ctx.moveTo(-8, 0);
        ctx.lineTo(8, 0);
        ctx.moveTo(8, 0);
        ctx.lineTo(4.6, -2.4);
        ctx.moveTo(8, 0);
        ctx.lineTo(4.6, 2.4);
        ctx.stroke();
        break;
      case "repair": // unreachable (heart.png branch above) — total switch
        break;
    }
    ctx.restore();
  }

  private drawBullets(alpha: number): void {
    if (!this.cur || !this.atlas) return;
    const prevById = new Map<number, SnapshotBullet>();
    if (this.prev) for (const b of this.prev.bullets) prevById.set(b.i, b);
    const ctx = this.ctx;
    for (const b of this.cur.bullets) {
      const p = prevById.get(b.i);
      const x = p ? lerp(p.x, b.x, alpha) : b.x;
      const y = p ? lerp(p.y, b.y, alpha) : b.y;
      const info = this.ownerInfo.get(b.owner);
      const big = (info?.sprite ?? 0) === 0;
      // Trail noted from the RENDERED position (see applySnapshot) so the ring
      // trails the drawn shell, never leads it.
      this.fx.noteBullet(b.i, x, y, TANK_COLORS[info?.colorIdx ?? 0]);
      // Buffed shells always draw the big tinted sprite (plan: bullet_01 is
      // the gold/violet tint target) so a powerup shot is legible at a glance;
      // plain shells match the shooter's sprite size.
      const img: Sprite = b.pierce
        ? this.atlas.tintedBullet.pierce
        : b.bounces > BULLET_BOUNCES
          ? this.atlas.tintedBullet.bounce
          : this.atlas.bullet[big ? 0 : 1];
      const boosted = b.pierce || b.bounces > BULLET_BOUNCES;
      const [bw, bh] = boosted ? BULLET_DRAW[0] : BULLET_DRAW[big ? 0 : 1];
      ctx.save();
      ctx.translate(x, y);
      // Same derivation as the tank hull (class header): sprite faces up,
      // world dir d points (cos d, sin d) ⇒ rotate(d + π/2).
      ctx.rotate(b.dir + Math.PI / 2);
      // A shell on its LAST bounce fades (snapshot.ts contract: consumers dim
      // a bullet with bounces 0) — the player sees it expire next wall hit.
      if (!boosted && b.bounces === 0) ctx.globalAlpha = 0.65;
      ctx.drawImage(img, -bw / 2, -bh / 2, bw, bh);
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  private drawTanks(now: number, dtMs: number, alpha: number): void {
    if (!this.cur || !this.atlas) return;
    const prevById = new Map<number, SnapshotTank>();
    if (this.prev) for (const t of this.prev.tanks) prevById.set(t.i, t);
    const ctx = this.ctx;
    for (const t of this.cur.tanks) {
      const p = prevById.get(t.i);
      const x = p ? lerp(p.x, t.x, alpha) : t.x;
      const y = p ? lerp(p.y, t.y, alpha) : t.y;
      const h = p ? lerpAngle(p.h, t.h, alpha) : t.h;
      const color = TANK_COLORS[t.colorIdx];
      const [w, hh] = TANK_DRAW[t.sprite];
      const frame = this.fx.treadFrame(t.i, t.v, dtMs);

      ctx.save();
      ctx.translate(x, y);
      if (!t.alive) ctx.globalAlpha = 0.22; // wreck: keep for spatial memory
      // Class-header derivation: rotate(h + π/2) turns the up-facing art to
      // the engine heading.
      ctx.rotate(h + Math.PI / 2);
      ctx.drawImage(this.atlas.tank[t.colorIdx][t.sprite][frame], -w / 2, -hh / 2, w, hh);
      const flashEnd = this.flashUntil.get(t.i);
      if (flashEnd !== undefined) {
        const rem = flashEnd - now;
        if (rem <= 0) this.flashUntil.delete(t.i);
        else {
          ctx.globalAlpha = Math.min(1, rem / 90) * 0.85;
          ctx.drawImage(this.atlas.flash[t.sprite][frame], -w / 2, -hh / 2, w, hh);
        }
      }
      ctx.restore();
      ctx.globalAlpha = 1;
      if (!t.alive) continue;

      // Team ring: hull-aligned ellipse, the colour key the scoreboard echoes.
      // Rotation h + π/2 puts the ellipse's LONG (46-based) axis along the
      // heading: a local axis at angle R points to world (cos R, sin R), and
      // R = h + π/2 is exactly the perpendicular of (cos h, sin h).
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(x, y, w / 2 + 3.5, hh / 2 + 3.5, h + Math.PI / 2, 0, TAU);
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (t.shield) {
        // Dashed invulnerability ring, gently breathing so it never reads as
        // static geometry.
        ctx.strokeStyle = "#58a6ff";
        ctx.globalAlpha = 0.55 + 0.3 * Math.sin(this.clock / 120);
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.arc(x, y, hh / 2 + 7, 0, TAU);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // HP bar above the hull: dark tray + 4-hit-legible fill (25 dmg = one
      // colour step), outlined in the team colour to tie it to the tank.
      const frac = clamp(t.hp / TANK_HP, 0, 1);
      const bw = 30;
      const bx = x - bw / 2;
      const by = y - hh / 2 - 10;
      ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
      ctx.fillRect(bx - 1, by - 1, bw + 2, 6);
      ctx.fillStyle = frac > 0.5 ? "#57c25a" : frac > 0.25 ? "#e3b341" : "#ff7b72";
      ctx.fillRect(bx, by, bw * frac, 4);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx - 1.5, by - 1.5, bw + 3, 7);
    }
  }

  /** Explosion sheet frames: 10 cells, 60 ms/frame at tank scale (80 u) and
   *  40 ms at bullet scale (20 u) — dwell lives on the Fx entry. */
  private drawExplosions(): void {
    if (!this.atlas) return;
    const ctx = this.ctx;
    for (const fx of this.fx.getExplosions()) {
      const i = Math.min(EXPLOSION_FRAMES.length - 1, Math.floor(fx.age / fx.msPerFrame));
      const { sx, sy, s } = explosionSrc(i);
      ctx.drawImage(
        this.atlas.explosion,
        sx, sy, s, s,
        fx.x - fx.size / 2, fx.y - fx.size / 2, fx.size, fx.size,
      );
    }
  }

  /** Escalation vignette. Phase 2 (sudden death): pulsing red edge gradient.
   *  Phase 1: a barely-there amber breath — escalation should be felt, not
   *  watched. A "phase" event boosts the edge for 400 ms so the transition is
   *  announced even at low ambient alpha. */
  private drawVignette(now: number): void {
    const flash = Math.max(0, 1 - (now - this.phaseFlashAt) / 400);
    if (this.phase === 0 && flash <= 0) return;
    const ctx = this.ctx;
    const cx = ARENA_W / 2;
    const cy = ARENA_H / 2;
    const outer = Math.hypot(cx, cy);
    const g = ctx.createRadialGradient(cx, cy, outer * 0.55, cx, cy, outer);
    if (this.phase === 2) {
      const a = 0.16 + 0.08 * Math.sin(this.clock / 160) + flash * 0.3;
      g.addColorStop(0, "rgba(232, 60, 50, 0)");
      g.addColorStop(1, `rgba(232, 60, 50, ${a.toFixed(3)})`);
    } else {
      const a = 0.05 + 0.04 * Math.sin(this.clock / 300) + flash * 0.18;
      g.addColorStop(0, "rgba(227, 179, 65, 0)");
      g.addColorStop(1, `rgba(227, 179, 65, ${a.toFixed(3)})`);
    }
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, ARENA_W, ARENA_H);
  }
}
