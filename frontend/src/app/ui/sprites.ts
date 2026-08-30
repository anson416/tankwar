// sprites.ts — asset loading + the runtime tint atlas. Main thread only (DOM
// canvas here is the point: the worker owns the simulation and its tensors).
//
// WHY a pre-multiplied tint ATLAS instead of per-frame filter/composite draws:
// plan §Assets pins 4 colours × 2 sprites × 2 frames = 16 tank canvases built
// ONCE at boot via "multiply" + "destination-in". Building them per frame
// would run three full-surface composites per tank per frame; building them
// once makes the per-frame cost a plain drawImage of a prerendered bitmap.
//
// Tint recipe (per variant): draw the greyscale source, multiply the team
// colour over it (colour × grey = tinted colour), then "destination-in" the
// source again — multiply is a blend mode composited source-over, so the fill
// would otherwise paint the transparent padding too; the destination-in pass
// re-applies the source alpha as a mask and restores the silhouette.
//
// All URLs are RELATIVE ("./assets/...") — vite.config.ts sets base "./", so
// they resolve against the Pages subpath (/tankwar/assets/...) as well as the
// dev server root.

import type { Theme } from "./theme";

/** Anything drawImage accepts that also exposes intrinsic pixel size. */
export type Sprite = CanvasImageSource & { width: number; height: number };

/** Team palette, index = SnapshotTank.colorIdx (0–3). Fixed: the CSS theme
 *  tokens are page chrome; tank colours must stay readable on BOTH themes, so
 *  they are theme-independent by design. */
export const TANK_COLORS = ["#e8543f", "#3f8ee8", "#57c25a", "#e8c13f"] as const;

/** Tint for bounce-ammo shells (gold) and pierce shells (violet) — plan
 *  §Assets. Boosted shots must be legible at a glance, so BOTH draw from the
 *  big bullet_01 sprite regardless of the owner's sprite size (see renderer). */
export const BOUNCE_TINT = "#ffcf5c";
export const PIERCE_TINT = "#a86bff";

/** The explosion sheet's 10 used cells, in animation order — the 8×8 grid of
 *  128 px cells holds 64 frames; the original game animated this subset. */
export const EXPLOSION_FRAMES = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37] as const;

/** Edge length of one explosion sheet cell (1024² sheet, 8×8 grid). */
export const EXPLOSION_CELL = 128;

/**
 * Sub-rect of the explosion sheet for animation step `f` (0–9, clamped).
 * Cell = EXPLOSION_FRAMES[f] laid out row-major on the 8-wide sheet.
 */
export function explosionSrc(f: number): { sx: number; sy: number; s: number } {
  const i = EXPLOSION_FRAMES[Math.max(0, Math.min(EXPLOSION_FRAMES.length - 1, f | 0))];
  return { sx: (i % 8) * EXPLOSION_CELL, sy: ((i / 8) | 0) * EXPLOSION_CELL, s: EXPLOSION_CELL };
}

/** Everything the renderer blits. `ready` is true once the promise resolves —
 *  the atlas object is built synchronously AFTER the loads, so consumers that
 *  hold the resolved value can rely on every field. */
export interface SpriteAtlas {
  /** [colorIdx][sprite 0|1][tread frame 0|1] — prerendered team tints. */
  tank: Sprite[][][];
  /** Pure-white silhouettes [sprite][frame] for the hit flash. */
  flash: Sprite[][];
  /** Base shells [0 = big bullet_01, 1 = small bullet_02]. */
  bullet: Sprite[];
  /** Powerup-ammo shells, tinted per plan. */
  tintedBullet: Record<"bounce" | "pierce", Sprite>;
  /** Repair icon, drawn verbatim (22×22 target, plan §Assets). */
  heart: Sprite;
  /** The 1024² explosion sheet (WEBP bytes under a .png name — browsers sniff
   *  content, not extensions, so the plain Image load just works). */
  explosion: HTMLImageElement;
  /** Arena floor art, pre-dimmed by the renderer into an offscreen canvas. */
  background: HTMLImageElement;
  ready: boolean;
}

/** Load one image; rejects with the URL so a missing asset names itself. */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`sprite load failed: ${src}`));
    img.src = src;
  });
}

/** Offscreen canvas of `img` tinted with `color` (multiply + alpha restore). */
function tinted(img: Sprite, color: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(img, 0, 0);
  // Multiply the fill INTO the art; transparent pixels would be painted by the
  // source-over compositing of the blend, hence the mask pass below.
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, c.width, c.height);
  // Re-clip to the source alpha: only pixels opaque in the original survive.
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(img, 0, 0);
  return c;
}

/** Flat `color` silhouette clipped to the sprite's alpha — the hit-flash frame
 *  ("multiply" with white would be a no-op, so this uses "source-in"). */
function silhouette(img: Sprite, color: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(img, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

// Source art, in sprite order (0 = heavy tank_01 185×254, 1 = light tank_02
// 123×234), each as its two tread frames A/B. The renderer scales to the plan's
// draw sizes (34×46 / 28×50 u) — the atlas stays at native resolution so the
// scale is one clean drawImage, never a re-encoded intermediate.
const TANK_SOURCES: readonly (readonly [string, string])[] = [
  ["./assets/images/tank_01/tank_01_A.png", "./assets/images/tank_01/tank_01_B.png"],
  ["./assets/images/tank_02/tank_02_A.png", "./assets/images/tank_02/tank_02_B.png"],
];

let cached: Promise<SpriteAtlas> | null = null;

/**
 * Load every image and build the tint atlas. Concurrent callers share ONE
 * load (module-level promise cache): the watch and play controllers boot
 * independently and must not double-fetch nine assets or rebuild 24 canvases.
 * Rejects if any asset is missing — a broken asset list is a boot bug, not a
 * render-time fallback.
 */
export function loadSprites(): Promise<SpriteAtlas> {
  if (cached) return cached;
  cached = (async (): Promise<SpriteAtlas> => {
    const [t0a, t0b, t1a, t1b, bulletBig, bulletSmall, heart, background, explosion] =
      await Promise.all(
        [
          TANK_SOURCES[0][0],
          TANK_SOURCES[0][1],
          TANK_SOURCES[1][0],
          TANK_SOURCES[1][1],
          "./assets/images/bullets/bullet_01.png",
          "./assets/images/bullets/bullet_02.png",
          "./assets/images/heart.png",
          "./assets/images/background.png",
          "./assets/images/explosion/explosion.png",
        ].map(loadImage),
      );

    const tankImgs: Sprite[][] = [
      [t0a, t0b],
      [t1a, t1b],
    ];
    // [colorIdx][sprite][frame] — outer colour so a single tank's two frames
    // stay adjacent in memory-iteration order during the per-frame draw.
    const tank: Sprite[][][] = TANK_COLORS.map((color) =>
      tankImgs.map((frames) => frames.map((img) => tinted(img, color))),
    );
    const flash: Sprite[][] = tankImgs.map((frames) => frames.map((img) => silhouette(img, "#ffffff")));

    return {
      tank,
      flash,
      bullet: [bulletBig, bulletSmall],
      tintedBullet: {
        bounce: tinted(bulletBig, BOUNCE_TINT),
        pierce: tinted(bulletBig, PIERCE_TINT),
      },
      heart,
      explosion,
      background,
      ready: true,
    };
  })();
  // A failed boot must not poison later retries with a rejected cache — the
  // controller surfaces the error and the next call re-attempts the load.
  cached.catch(() => {
    cached = null;
  });
  return cached;
}

/** Floor-tint layered over the 25%-brightness background, per theme (the hex
 *  values mirror the --bg tokens in style.css). */
export function bgOverlay(theme: Theme): string {
  return theme === "dark" ? "rgba(14, 17, 22, 0.30)" : "rgba(246, 248, 250, 0.38)";
}
