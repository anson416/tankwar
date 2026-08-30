// play-controller.ts — the Play tab: the human seat (tank 0) vs 1–3 trained
// AIs, FFA or allied, with a difficulty-scaled opponent softmax and optional
// aim assist. Real-time 30 Hz: keyboard state is shipped per tick, snapshots
// stream back, and the HUD is plain DOM on top of the canvas.
//
// CONTRACT with main.ts (same shape as train-controller): buildPlaySection()
// builds the whole <section id="panel-play"> — every id/testid listed below —
// and main appends it to the tab container; PlayController then binds into it
// by id and needs three calls from main: init() (after mount), setLang() on a
// language flip, and setVisible() from the tab switch handler (which is also
// what starts the first match).
//
// Concurrency discipline (big-two's table-controller adapted to a real-time
// lane): the stream is unsolicited id-0 pushes with no request id to pair, so
// every frame is gated on a LOCAL generation — `gen` bumps on each play
// request and `live` only turns true on that generation's {arena} message
// (sent once per episode before the lane's first snapshot). Hiding the tab
// pauses the lane gen-guarded, so a show/hide race around a rematch can never
// resume a dead generation, and stale input is dropped twice over: key
// tracking is disabled with the tab (enable()/disable()) while the forwarding
// gate adds "match live", and the per-packet `seq` lets the worker discard
// packets that overtook each other in the postMessage queue.
//
// Deliberate asymmetry vs watch: the WORKER auto-restarts watch lanes, but a
// play match always waits for the human to hit Rematch — the player owns the
// pacing of their own fights.
import { byId, el } from "../by-id";
import { t, type Lang, type StringKey } from "../i18n";
import type { GameAudio } from "../ui/audio";
import { SceneRenderer } from "../ui/renderer";
import { TankInput, type InputState } from "../ui/input";
import {
  isArena,
  isError,
  type ArenaMsg,
  type EpisodeEndMsg,
  type Metrics,
  type Snapshot,
  type TeamMode,
} from "../../train-protocol";
import type { SnapshotTank } from "../../game/snapshot";
import {
  ARENA_H,
  ARENA_W,
  BOUNCE_TIME,
  FIRE_COOLDOWN,
  PIERCE_TIME,
  RAPID_TIME,
  SHIELD_TIME,
  SPEED_TIME,
  TANK_HP,
  TICK_HZ,
} from "../../game/constants";
import type { WorkerBackendController } from "../worker-client";

/** powerup.* is a template-built key; the i18n dict is a literal union, so
 *  the family gets its one honest construction point (kinds are limited to
 *  what the snapshot can ship — repair is instantaneous and never listed). */
const puKey = (kind: string): StringKey => `powerup.${kind}` as StringKey;

/** Scoreboard refresh cadence while the tab is hidden — normally moot (the
 *  lane is paused on hide), but it bounds DOM work if a pause ever fails to
 *  land (the PauseReq is lane-scoped; this controller addresses "play"). */
const HIDDEN_RENDER_MS = 250;

/** Floor-display durations for active powerup chips. The snapshot ships WHICH
 *  timers are live, never how long is left, so the HUD estimates the remainder
 *  from the same constants the engine uses and re-arms on every re-pickup —
 *  display only, never read back by the game. */
const PU_SECONDS: Record<string, number> = {
  rapid: RAPID_TIME,
  shield: SHIELD_TIME,
  speed: SPEED_TIME,
  bounce: BOUNCE_TIME,
  pierce: PIERCE_TIME,
};

/**
 * Engine loops are reconciled against the snapshot's tank list (one loop per
 * alive tank, released when it disappears), so an empty tank list IS the
 * "all engines off" signal — pause/hide reuse it instead of a dedicated audio
 * API that would only duplicate syncEngines' own pruning.
 */
const ENGINES_OFF = { tanks: [] } as unknown as Snapshot;

/**
 * Build the Play panel: config row, canvas + HUD, keyboard hints. The section
 * is `hidden` like every tab panel; main.ts mounts it and its tab handler
 * unhides it. Styling leans on the shared primitives in style.css
 * (.panel/.btn/.field/.badge/.config-grid) plus inline layout, so the panel
 * needs no CSS of its own to be usable.
 */
export function buildPlaySection(): HTMLElement {
  const section = el("section", {
    class: "tab-section", id: "panel-play", "data-testid": "panel-play",
    role: "tabpanel", "aria-labelledby": "tab-play", hidden: true,
  });

  // --- config row: opponents · team mode · difficulty · assist · rematch ---
  const config = el("div", { class: "panel", "data-testid": "play-config" });
  const grid = el("div", { class: "config-grid" });

  const oppField = el("label", { class: "field" });
  oppField.appendChild(el("span", { class: "field-label", "data-i18n": "play.opponents", text: "Opponents" }));
  const opp = el("select", { class: "num", id: "play-opponents", "data-testid": "play-opponents" });
  for (let n = 1; n <= 3; n++) {
    opp.appendChild(el("option", { value: n, text: String(n), ...(n === 1 ? { selected: true } : {}) }));
  }
  oppField.appendChild(opp);
  grid.appendChild(oppField);

  const teamField = el("label", { class: "field" });
  teamField.appendChild(el("span", { class: "field-label", "data-i18n": "play.teamMode", text: "Team mode" }));
  const team = el("select", { class: "num", id: "play-team", "data-testid": "play-team" });
  team.appendChild(el("option", { value: "ffa", "data-i18n": "play.teamMode.ffa", text: "Free-for-all" }));
  team.appendChild(el("option", { value: "allied", "data-i18n": "play.teamMode.allied", text: "Allied AIs" }));
  teamField.appendChild(team);
  grid.appendChild(teamField);

  const diffField = el("label", { class: "field" });
  diffField.appendChild(el("span", { class: "field-label", "data-i18n": "play.difficulty", text: "Difficulty" }));
  const diffRow = el("span", { style: "display:flex;gap:8px;align-items:center" });
  // min/max before value: the value attribute is clamped to a range that must
  // already exist, and attribute order here is set-order.
  diffRow.appendChild(
    el("input", {
      type: "range", id: "play-difficulty", "data-testid": "play-difficulty",
      min: "0", max: "1", step: "0.05", value: "1",
    }),
  );
  diffRow.appendChild(el("output", { id: "play-diff-out", "data-testid": "play-diff-out", text: "τ=0.05" }));
  diffField.appendChild(diffRow);
  grid.appendChild(diffField);

  const assistField = el("label", { class: "field" });
  assistField.appendChild(el("span", { class: "field-label", "data-i18n": "play.aimAssist", text: "Aim assist" }));
  assistField.appendChild(
    el("input", { type: "checkbox", id: "play-assist", "data-testid": "play-assist", checked: true }),
  );
  grid.appendChild(assistField);

  config.appendChild(grid);
  config.appendChild(
    el("button", {
      class: "btn primary", id: "play-rematch", type: "button",
      "data-testid": "play-rematch", "data-i18n": "play.rematch", text: "Rematch",
    }),
  );
  section.appendChild(config);

  // --- result banner (also the surface for a failed lane start) ---
  section.appendChild(
    el("div", {
      id: "play-banner", "data-testid": "play-banner", role: "status", hidden: true,
      style:
        "padding:10px 14px;border:1px solid var(--accent);border-radius:var(--radius);" +
        "background:var(--accent-soft);font-weight:600",
    }),
  );

  // --- canvas + HUD ---
  section.appendChild(
    el("canvas", {
      id: "play-canvas", "data-testid": "play-canvas", width: ARENA_W, height: ARENA_H,
      style:
        "width:100%;max-width:var(--arena-max);display:block;background:var(--bg);" +
        "border:1px solid var(--border);border-radius:var(--radius-sm)",
    }),
  );

  const hud = el("div", { class: "panel", id: "play-hud", "data-testid": "play-hud" });
  const hudRows = el("div", { style: "display:flex;flex-direction:column;gap:6px" });
  // One labeled-bar builder for HP and reload — same geometry, different fill.
  const bar = (fillId: string, label: StringKey, color: string): HTMLElement => {
    const row = el("div", { style: "display:flex;gap:8px;align-items:center" });
    row.appendChild(el("span", { class: "field-label", "data-i18n": label, style: "width:64px;flex:none" }));
    const track = el("span", {
      style: "flex:1;height:10px;border-radius:999px;background:var(--bg-elev2);overflow:hidden",
    });
    track.appendChild(el("i", { id: fillId, style: `display:block;height:100%;width:100%;background:${color}` }));
    row.appendChild(track);
    return row;
  };
  hudRows.appendChild(bar("play-hp-fill", "hud.hp", "var(--good)"));
  hudRows.appendChild(bar("play-reload-fill", "hud.reload", "var(--accent)"));
  const kd = el("div", { style: "display:flex;gap:8px;align-items:center" });
  kd.appendChild(el("span", { class: "field-label", "data-i18n": "hud.kills", style: "width:64px;flex:none" }));
  kd.appendChild(
    el("span", {
      id: "play-kd", "data-testid": "play-kd", text: "0 / 0",
      style: "font-variant-numeric:tabular-nums",
    }),
  );
  hudRows.appendChild(kd);
  hudRows.appendChild(
    el("div", {
      id: "play-powerups", "data-testid": "play-powerups",
      style: "display:flex;gap:4px;flex-wrap:wrap;min-height:26px",
    }),
  );
  hud.appendChild(hudRows);
  hud.appendChild(
    el("div", {
      id: "play-esc-banner", "data-testid": "play-esc-banner", role: "status", hidden: true,
      style:
        "margin-top:8px;padding:8px 12px;border:1px solid var(--warn);border-radius:var(--radius-sm);" +
        "color:var(--warn);font-weight:600",
    }),
  );
  hud.appendChild(
    el("span", {
      class: "badge", id: "play-untrained", "data-testid": "play-untrained",
      "data-i18n": "play.untrained", text: "Untrained model (<10 episodes)", hidden: true,
      style: "margin-top:8px;border-color:var(--warn);color:var(--warn)",
    }),
  );
  section.appendChild(hud);

  // --- keyboard hints (train-config style title over the hint text) ---
  const keys = el("div", { class: "panel", id: "play-keys", "data-testid": "play-keys" });
  keys.appendChild(el("h2", { class: "panel-title", "data-i18n": "tab.play", text: "Play" }));
  keys.appendChild(
    el("p", { class: "hint", "data-i18n": "play.hint", text: "W/S throttle · A/D steer · Space fire" }),
  );
  section.appendChild(keys);

  return section;
}

/**
 * Play-lane lifecycle: idle until the first activation (the tab's first
 * setVisible(true) starts the match), then persistent across tab switches —
 * hidden pauses the lane, showing resumes it, or restarts it if the roster
 * config changed while it was hidden.
 */
export class PlayController {
  private lang: Lang = "en";
  /** Bumped on every play request; anything older is dropped. */
  private gen = 0;
  /** Gen at which the lane was paused by hiding — resuming must match it. */
  private hideGen = -1;
  /** True while the lane is frozen by a hide: audio must stay off even though
   *  a repaint fires on return, because a frozen lane streams no snapshots to
   *  re-silence the loops afterwards. */
  private pausedByHide = false;
  /** True once THIS generation's {arena} arrived (frames before it are stale). */
  private live = false;
  private started = false;
  private visible = false;
  /** Roster changed while the restart could not run (hidden tab). */
  private configDirty = false;
  private inputActive = false;
  /** Last result, kept so a language switch can re-render the banner text. */
  private ended: EpisodeEndMsg | null = null;
  private lastSnapshot: Snapshot | null = null;
  private lastTick = -1;
  private metrics: Metrics | null = null;
  private renderer: SceneRenderer | null = null;
  private input: TankInput | null = null;
  private puTimers = new Map<string, number>();
  private puChips = new Map<string, HTMLElement>();
  private lastHiddenPaint = 0;
  private audioTick = -1;
  private readonly seenEvents = new Set<string>();
  private raf = 0;
  private root: HTMLElement | null = null;
  private unsubs: (() => void)[] = [];

  constructor(
    private worker: WorkerBackendController,
    private audio: GameAudio,
  ) {}

  /** Paint pump (see watch-controller): runs continuously so the renderer's
   *  effects clock ages queued bursts even while the tab is hidden. */
  private readonly loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    this.renderer?.frame(now);
  };

  /** Attach listeners, build the renderer + keyboard input. MUST run after the
   *  section is mounted — byId throws on a missing id by design (wiring bug).
   *  Deliberately does NOT start a match: the first activation does that. */
  init(): void {
    this.root = byId<HTMLElement>("panel-play");
    this.renderer = new SceneRenderer(byId<HTMLCanvasElement>("play-canvas"));
    this.input = new TankInput((a: InputState) => this.onInput(a));

    byId("play-rematch").addEventListener("click", () => this.start());
    // Rostering changes reshape the arena, so they restart a running match;
    // the difficulty slider and aim assist are read at start() and only take
    // effect on the next one (τ shapes the opponent softmax at selection
    // time — a mid-episode restart would punish experimenting with it).
    byId("play-opponents").addEventListener("change", () => this.onConfigChange());
    byId("play-team").addEventListener("change", () => this.onConfigChange());
    const diff = byId<HTMLInputElement>("play-difficulty");
    diff.addEventListener("input", () => {
      byId("play-diff-out").textContent = this.tauLabel(diff.valueAsNumber);
    });
    // TankInput parks the assist flag so it survives re-renders; the checkbox
    // mirrors into it and start() reads it back from there.
    const assist = byId<HTMLInputElement>("play-assist");
    this.input?.setAimAssist(assist.checked);
    assist.addEventListener("change", () => this.input?.setAimAssist(assist.checked));

    this.unsubs = [
      this.worker.onSnapshot("play", (s) => this.onSnapshot(s)),
      this.worker.onArena("play", (a) => this.onArena(a)),
      this.worker.onEpisodeEnd("play", (e) => this.onEpisodeEnd(e)),
      this.worker.onMetrics((m) => this.onMetrics(m)),
    ];

    this.raf = requestAnimationFrame(this.loop);
  }

  // --- lane control -----------------------------------------------------------

  /** One tick of human control state, already seq-stamped by TankInput.
   *  enable()/disable() is the braces; this re-check is the belt — a keyup
   *  that lands between the two must not move a frozen lane. */
  private onInput(a: InputState): void {
    if (!this.inputActive) return;
    this.worker.input(a);
  }

  /** Roster change: restart now if a match is on screen, else defer to the
   *  next activation so a hidden tab doesn't spawn a match nobody watches. */
  private onConfigChange(): void {
    if (!this.started) return; // first activation reads the DOM anyway
    if (this.visible && this.live) {
      void this.start();
      return;
    }
    this.configDirty = true;
  }

  private async start(): Promise<void> {
    const gen = ++this.gen;
    this.started = true;
    this.configDirty = false;
    this.pausedByHide = false;
    this.live = false;
    this.ended = null;
    this.lastSnapshot = null;
    this.lastTick = -1;
    this.puTimers.clear();
    this.syncInput(); // drops the keyboard until the new arena arrives
    byId("play-banner").hidden = true;

    const opponents = Number(byId<HTMLSelectElement>("play-opponents").value);
    const teamMode = byId<HTMLSelectElement>("play-team").value as TeamMode;
    const difficulty = byId<HTMLInputElement>("play-difficulty").valueAsNumber;
    const aimAssist = this.input?.aimAssist ?? byId<HTMLInputElement>("play-assist").checked;
    try {
      const resp = await this.worker.playStart({
        tankCount: 1 + opponents, // the human seat plus the AIs
        teamMode,
        difficulty,
        aimAssist,
      });
      if (gen !== this.gen) return; // superseded while in flight
      if (isError(resp)) {
        console.warn("[play] start failed:", resp.error);
        this.showBanner(t(this.lang, "play.error"));
        return;
      }
      // The ack may itself carry the first {arena} (id-keyed response) or the
      // worker may push it on id 0 — both paths land in onArena().
      if (isArena(resp)) this.onArena(resp.arena);
    } catch {
      // A dead worker must not leave the player staring at a frozen canvas.
      if (gen === this.gen) this.showBanner(t(this.lang, "play.error"));
    }
  }

  // --- worker streams ---------------------------------------------------------

  private onArena(a: ArenaMsg): void {
    // No id to correlate: an {arena} arriving now is either this generation's
    // first layout or a straggler from the match the user just replaced — at
    // worst one frame of the old world, healed when the new match's own arena
    // lands a message later.
    this.live = true;
    this.hideGen = -1;
    this.pausedByHide = false;
    this.ended = null;
    this.puTimers.clear();
    this.syncInput();
    byId("play-banner").hidden = true;
    this.renderer?.setArena(a);
  }

  private onSnapshot(s: Snapshot): void {
    if (!this.live) return; // pre-arena, or a superseded generation's frame
    // Smoke hook: the Playwright play spec advances the lane with a synthetic
    // keydown and polls this. Watch shares the key — one counter per page is
    // enough for both specs, and they never run concurrently.
    (window as unknown as { __tickCount?: number }).__tickCount = s.tick;
    this.lastSnapshot = s;
    // The renderer absorbs every snapshot even while hidden — its event drain
    // is what fires each burst exactly once, and its visible gate skips only
    // the painting. DOM and audio are what this controller throttles.
    this.renderer?.applySnapshot(s);
    if (this.visible) {
      this.paint(s);
      return;
    }
    const now = performance.now();
    if (now - this.lastHiddenPaint < HIDDEN_RENDER_MS) return;
    this.lastHiddenPaint = now;
    this.paint(s);
  }

  /** Banner text for an episode end. "reconfigured" is the worker's synthetic
   *  end after a config Apply dropped the lane mid-match — nobody won, the
   *  human just needs their rematch button back. */
  private bannerText(e: EpisodeEndMsg): string {
    if (e.reason === "reconfigured") return t(this.lang, "play.banner.reconfigured");
    // `winner` is a tank id; the human seat is tank 0 (playStart seats the
    // human first), null is a mutual-destruction/timeout draw.
    return e.winner === 0
      ? t(this.lang, "play.banner.win")
      : e.winner === null
        ? t(this.lang, "play.banner.draw")
        : t(this.lang, "play.banner.lose");
  }

  private onEpisodeEnd(e: EpisodeEndMsg): void {
    if (!this.live) return; // a superseded match's result
    this.ended = e;
    this.live = false;
    this.syncInput();
    this.audio.syncEngines(ENGINES_OFF);
    this.showBanner(this.bannerText(e));
    // Dead tanks don't need the keyboard, and the button is the next move —
    // focus it so Enter rematches without a pointer.
    byId<HTMLButtonElement>("play-rematch").focus();
  }

  /** Fleet metrics push: drives the "untrained model" badge (<10 episodes).
   *  Unlike the stream subscriptions (armed in init, after mount) main may
   *  call this directly at any time, so it must tolerate a pre-mount call. */
  onMetrics(m: Metrics): void {
    this.metrics = m;
    if (this.root === null) return;
    byId("play-untrained").hidden = m.episodes >= 10;
  }
  /** Latest fleet metrics — exposed for tests/diagnostics. */
  get lastMetrics(): Metrics | null {
    return this.metrics;
  }

  // --- painting ---------------------------------------------------------------

  private paint(s: Snapshot): void {
    this.paintHud(s);
    // A lane frozen by a hide streams no snapshots to re-silence the engine
    // loops, so returning to it must not restart them either.
    if (!this.visible || this.pausedByHide) return;
    this.audio.syncEngines(s);
    this.pumpAudio(s);
  }

  private paintHud(s: Snapshot): void {
    // The human seat is tank 0 (see onEpisodeEnd) — an allied match still
    // numbers the human's seat first, so the id is the stable key either way.
    const me = s.tanks.find((tk) => tk.i === 0);
    if (me) {
      const hpFrac = Math.max(0, Math.min(1, me.hp / TANK_HP));
      const hpFill = byId("play-hp-fill");
      hpFill.style.width = `${(hpFrac * 100).toFixed(1)}%`;
      hpFill.style.background = hpFrac <= 0.25 ? "var(--red)" : "var(--good)";
      // Readiness: 0 reload = full bar. RAPID fire tops out below 100% — the
      // bar is honest about the shorter cycle, not pinned to the base one.
      const ready = me.reload <= 0 ? 1 : 1 - me.reload / FIRE_COOLDOWN;
      byId("play-reload-fill").style.width = `${(Math.max(0, Math.min(1, ready)) * 100).toFixed(1)}%`;
      byId("play-kd").textContent = `${me.kills} / ${me.deaths}`;
      this.paintPowerups(s, me);
    }
    const esc = byId("play-esc-banner");
    if (s.phase === 0) {
      esc.hidden = true;
    } else {
      esc.hidden = false;
      esc.textContent = t(this.lang, s.phase === 1 ? "esc.phase1" : "esc.phase2");
    }
  }

  private paintPowerups(s: Snapshot, me: SnapshotTank): void {
    // Estimated countdown (see PU_SECONDS): decay by elapsed ticks, re-arm on
    // re-pickup, drop when both the timer and the snapshot agree it's gone.
    const dt = this.lastTick < 0 ? 0 : Math.min(0.5, (s.tick - this.lastTick) / TICK_HZ);
    this.lastTick = s.tick;
    for (const [kind, left] of this.puTimers) this.puTimers.set(kind, left - dt);
    for (const kind of me.powerups) {
      if (!this.puTimers.has(kind)) this.puTimers.set(kind, PU_SECONDS[kind] ?? 0);
    }
    for (const [kind, left] of [...this.puTimers]) {
      if (left <= 0 && !me.powerups.includes(kind)) this.puTimers.delete(kind);
    }
    for (const [kind, chip] of this.puChips) {
      if (!this.puTimers.has(kind)) {
        chip.remove();
        this.puChips.delete(kind);
      }
    }
    for (const [kind, left] of this.puTimers) {
      let chip = this.puChips.get(kind);
      if (!chip) {
        chip = el("span", { class: "badge", "data-testid": `play-pu-${kind}` });
        this.puChips.set(kind, chip);
        byId("play-powerups").appendChild(chip);
      }
      chip.textContent = `${t(this.lang, puKey(kind))} ${Math.max(0, left).toFixed(1)}s`;
    }
  }

  /** Forward this tick's juice queue to the audio layer, deduped: snapshot
   *  coalescing can re-deliver a tick, and a cannon voiced twice is worse
   *  than no sound at all. */
  private pumpAudio(s: Snapshot): void {
    if (s.tick !== this.audioTick) {
      this.audioTick = s.tick;
      this.seenEvents.clear(); // events are per-tick; a new tick re-arms
    }
    for (const ev of s.events) {
      const key = `${s.tick}:${ev.kind}:${ev.tank ?? ""}`;
      if (this.seenEvents.has(key)) continue;
      this.seenEvents.add(key);
      this.audio.event(ev.kind);
    }
  }

  // --- chrome & input gating --------------------------------------------------

  private showBanner(text: string): void {
    const banner = byId("play-banner");
    banner.textContent = text;
    banner.hidden = false;
  }

  /** τ = (1 − s)·1.0 + 0.05: s=1 ⇒ 0.05 ⇒ near-greedy opponents, s=0 ⇒ 1.05 ⇒
   *  near-uniform. Same curve the protocol documents for the ranked pool. */
  private tauLabel(s: number): string {
    return `τ=${((1 - s) * 1.0 + 0.05).toFixed(2)}`;
  }

  /**
   * Two gates, two lifetimes. KEY TRACKING follows the TAB (enable on show,
   * disable on hide — a hidden Play tab must never swallow WASD from the rest
   * of the page, and the held-key state must be truthful the moment the tab is
   * on screen, not only once the worker's first arena push lands). FORWARDING
   * is stricter: visible AND a live match, so keys never leak into the worker
   * on a hidden, idle or finished lane.
   */
  private syncInput(): void {
    const active = this.visible && this.live;
    if (active === this.inputActive) return;
    this.inputActive = active;
  }

  private renderStatic(): void {
    byId<HTMLButtonElement>("play-rematch").textContent = t(this.lang, "play.rematch");
  }

  setLang(lang: Lang): void {
    this.lang = lang;
    this.renderStatic();
    // main.ts rebinds the whole document on toggle; this keeps the panel
    // correct standalone too (and re-renders after a late mount).
    this.root?.querySelectorAll<HTMLElement>("[data-i18n]").forEach((node) => {
      node.textContent = t(lang, node.dataset.i18n as StringKey);
    });
    if (this.ended !== null) {
      this.showBanner(this.bannerText(this.ended));
    }
    const s = this.lastSnapshot;
    if (s !== null) this.paintHud(s);
  }

  /** Tab-visibility hook from main.ts's switch handler. */
  setVisible(v: boolean): void {
    const first = v && !this.started;
    this.visible = v;
    this.renderer?.setVisible(v);
    // Key tracking follows the tab (see syncInput): held keys are live the
    // moment Play is on screen — before the match's arena push arrives.
    if (v) this.input?.enable();
    else this.input?.disable();
    if (first) {
      void this.start(); // idle until first activation
      return;
    }
    if (!this.started) return;
    if (!v) {
      // Freeze the lane while hidden so the player never returns to a corpse.
      // Gen-guarded: a hide racing a rematch must not resume a dead generation.
      // The pause is addressed to THIS lane — a worker-wide flag would freeze
      // the Watch duel just because the player switched tabs.
      if (this.live) {
        this.hideGen = this.gen;
        this.pausedByHide = true;
        this.worker.pause(true, "play");
      }
      this.audio.syncEngines(ENGINES_OFF);
      this.syncInput();
      return;
    }
    if (this.configDirty) {
      void this.start();
      return;
    }
    if (this.live && this.hideGen === this.gen) {
      this.pausedByHide = false;
      this.worker.pause(false, "play");
    }
    const s = this.lastSnapshot;
    if (s !== null && this.live) this.paint(s); // repaint now, not next frame
  }

  /** Drop worker subscriptions, the paint pump and the keyboard (teardown/HMR). */
  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs = [];
    cancelAnimationFrame(this.raf);
    this.input?.disable();
  }
}
