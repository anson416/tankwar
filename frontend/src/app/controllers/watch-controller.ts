// watch-controller.ts — the Watch tab: a spectator lane of 2–4 greedy AI
// tanks, simulated in the worker and streamed here as coalesced snapshots.
//
// CONTRACT with main.ts (same shape as train-controller): buildWatchSection()
// builds the whole <section id="panel-watch"> — every id/testid listed below —
// and main appends it to the tab container; WatchController then binds into it
// by id and needs three calls from main: init() (after mount), setLang() on a
// language flip, setVisible() from the tab switch handler, onMetrics() if main
// wants to feed it telemetry.
//
// Concurrency discipline (big-two's table-controller adapted from turn-based
// to a real-time lane): the snapshot/arena/episodeEnd pushes are unsolicited
// id-0 messages that carry no request id, so nothing can be paired — every
// frame is validated against a LOCAL generation instead. `gen` bumps on every
// (re)start request and `live` only turns true again on that generation's
// {arena} message, which is sent once per episode BEFORE the lane's first
// snapshot; a frame from a lane the user just reconfigured can therefore never
// paint into the fresh one. Belt two: audio events are deduped per tick
// (coalescing can re-deliver a tick) and a hidden tab skips canvas/audio work,
// because painting an invisible canvas is pure waste.
//
// Ownership split on episode end: the WORKER owns auto-restart (it re-arms the
// lane itself); this controller only re-renders the result banner and — when
// auto-restart is unchecked — tells the lane to stop.
import { byId, el } from "../by-id";
import { t, type Lang, type StringKey } from "../i18n";
import type { GameAudio } from "../ui/audio";
import { SceneRenderer } from "../ui/renderer";
import { TANK_COLORS } from "../ui/sprites";
import {
  isArena,
  isError,
  type ArenaMsg,
  type EpisodeEndMsg,
  type Metrics,
  type Snapshot,
} from "../../train-protocol";
import type { SnapshotTank } from "../../game/snapshot";
import {
  ARENA_H,
  ARENA_W,
  EPISODE_SECONDS_DEFAULT,
  ESCALATION_FRAC,
  SUDDEN_DEATH_FRAC,
  TANK_HP,
  TICK_HZ,
} from "../../game/constants";
import type { WorkerBackendController } from "../worker-client";

/** name.p1..p4 and powerup.* are template-built keys; the i18n dict is a
 *  literal union, so each family needs its one honest construction point
 *  (seat clamped, kind limited to what the snapshot can ship). */
const nameKey = (seat: number): StringKey => `name.p${Math.max(1, Math.min(4, seat + 1))}` as StringKey;
const puKey = (kind: string): StringKey => `powerup.${kind}` as StringKey;

/** Debounce for the speed slider: `input` fires on every notch while dragging
 *  and each notch would otherwise restart the lane (fresh arena) mid-drag. */
const SPEED_DEBOUNCE_MS = 250;
/** Scoreboard refresh cadence while the tab is hidden (4 Hz) — the numbers
 *  stay roughly current while canvas and audio work stops entirely. */
const HIDDEN_RENDER_MS = 250;

/**
 * Engine loops are reconciled against the snapshot's tank list (one loop per
 * alive tank, released when it disappears), so an empty tank list IS the
 * "all engines off" signal — pause/hide reuse it instead of a dedicated audio
 * API that would only duplicate syncEngines' own pruning.
 */
const ENGINES_OFF = { tanks: [] } as unknown as Snapshot;

/**
 * Build the Watch panel: toolbar, canvas, escalation bar, scoreboard. The
 * section is `hidden` like every tab panel; main.ts mounts it and its tab
 * handler unhides it. Styling leans on the shared primitives in style.css
 * (.panel/.btn/.field/.badge) plus inline layout, so the panel needs no CSS
 * of its own to be usable.
 */
export function buildWatchSection(): HTMLElement {
  const section = el("section", {
    class: "tab-section", id: "panel-watch", "data-testid": "panel-watch",
    role: "tabpanel", "aria-labelledby": "tab-watch", hidden: true,
  });

  // --- toolbar: rematch · tanks · speed · pause · step · auto-restart ---
  const toolbar = el("div", { class: "panel", "data-testid": "watch-toolbar" });
  toolbar.appendChild(
    el("button", {
      class: "btn primary", id: "watch-new", type: "button",
      "data-testid": "watch-new", "data-i18n": "play.rematch", text: "Rematch",
    }),
  );

  const tanksField = el("label", { class: "field" });
  tanksField.appendChild(el("span", { class: "field-label", "data-i18n": "watch.tanks", text: "Tanks" }));
  const tanks = el("select", { class: "num", id: "watch-tanks", "data-testid": "watch-tanks" });
  for (let n = 2; n <= 4; n++) {
    tanks.appendChild(el("option", { value: n, text: String(n), ...(n === 4 ? { selected: true } : {}) }));
  }
  tanksField.appendChild(tanks);
  toolbar.appendChild(tanksField);

  const speedField = el("label", { class: "field" });
  speedField.appendChild(el("span", { class: "field-label", "data-i18n": "watch.speed", text: "Speed" }));
  const speedRow = el("span", { style: "display:flex;gap:8px;align-items:center" });
  // min/max before value: the value attribute is clamped to a range that must
  // already exist, and attribute order here is set-order.
  speedRow.appendChild(
    el("input", {
      type: "range", id: "watch-speed", "data-testid": "watch-speed",
      min: "0.5", max: "8", step: "0.5", value: "1",
    }),
  );
  speedRow.appendChild(el("output", { id: "watch-speed-out", "data-testid": "watch-speed-out", text: "×1" }));
  speedField.appendChild(speedRow);
  toolbar.appendChild(speedField);

  // Label alternates with watch.resume on click, so it is NOT bound via
  // data-i18n — a document-wide rebind would pin it to "Pause" forever.
  toolbar.appendChild(
    el("button", { class: "btn", id: "watch-pause", type: "button", "data-testid": "watch-pause", text: "Pause" }),
  );
  // Step is only meaningful on a frozen lane — start disabled, armed by pause.
  toolbar.appendChild(
    el("button", {
      class: "btn", id: "watch-step", type: "button", "data-testid": "watch-step",
      "data-i18n": "watch.step", text: "Step", disabled: true,
    }),
  );

  const autoField = el("label", { class: "field" });
  autoField.appendChild(el("span", { class: "field-label", "data-i18n": "watch.autoRestart", text: "Auto-restart" }));
  autoField.appendChild(el("input", { type: "checkbox", id: "watch-auto", "data-testid": "watch-auto", checked: true }));
  toolbar.appendChild(autoField);
  section.appendChild(toolbar);

  // --- result banner (also the surface for a failed lane start) ---
  section.appendChild(
    el("div", {
      id: "watch-banner", "data-testid": "watch-banner", role: "status", hidden: true,
      style:
        "padding:10px 14px;border:1px solid var(--accent);border-radius:var(--radius);" +
        "background:var(--accent-soft);font-weight:600",
    }),
  );

  // --- canvas + escalation bar ---
  section.appendChild(
    el("canvas", {
      id: "watch-canvas", "data-testid": "watch-canvas", width: ARENA_W, height: ARENA_H,
      style:
        "width:100%;max-width:var(--arena-max);display:block;background:var(--bg);" +
        "border:1px solid var(--border);border-radius:var(--radius-sm)",
    }),
  );

  const esc = el("div", {
    id: "watch-esc", "data-testid": "watch-esc", "data-phase": "0",
    style:
      "position:relative;height:10px;border-radius:999px;background:var(--bg-elev2);" +
      "border:1px solid var(--border);overflow:hidden",
  });
  esc.appendChild(
    el("i", {
      id: "watch-esc-fill",
      style: "position:absolute;top:0;bottom:0;left:0;width:0%;background:var(--warn);transition:width 120ms linear",
    }),
  );
  // Phase markers read the engine's constants — re-typing 70/85 here would
  // silently drift the moment the engine re-tunes escalation.
  for (const frac of [ESCALATION_FRAC, SUDDEN_DEATH_FRAC]) {
    esc.appendChild(
      el("i", {
        style: `position:absolute;top:0;bottom:0;width:2px;background:var(--text-dim);left:${(frac * 100).toFixed(1)}%`,
      }),
    );
  }
  section.appendChild(esc);

  const meta = el("div", { style: "display:flex;gap:8px;align-items:center;flex-wrap:wrap" });
  const genBadge = el("span", { class: "badge", style: "gap:6px" });
  genBadge.appendChild(el("span", { "data-i18n": "watch.gen", text: "Model gen" }));
  genBadge.appendChild(el("span", { id: "watch-gen", "data-testid": "watch-gen", text: "—" }));
  meta.appendChild(genBadge);
  meta.appendChild(
    el("span", { class: "badge", id: "watch-esc-label", "data-testid": "watch-esc-label", hidden: true }),
  );
  section.appendChild(meta);

  // --- scoreboard: column headers + one row per tank (rebuilt per cast) ---
  const scorePanel = el("div", { class: "panel" });
  scorePanel.appendChild(el("h2", { class: "panel-title", "data-i18n": "watch.scoreboard", text: "Scoreboard" }));
  const head = el("div", {
    style: "display:flex;gap:8px;align-items:center;color:var(--text-dim);font-size:11px;padding-bottom:4px",
  });
  head.appendChild(el("span", { style: "width:14px;flex:none" })); // swatch column
  head.appendChild(el("span", { style: "min-width:56px" })); // name column (labelled by the row itself)
  head.appendChild(el("span", { "data-i18n": "hud.kills", text: "Kills", style: "width:44px" }));
  head.appendChild(el("span", { "data-i18n": "hud.deaths", text: "Deaths", style: "width:48px" }));
  head.appendChild(el("span", { "data-i18n": "hud.hp", text: "HP", style: "flex:1" }));
  scorePanel.appendChild(head);
  scorePanel.appendChild(el("div", { id: "watch-score", "data-testid": "watch-score" }));
  section.appendChild(scorePanel);

  return section;
}

/** One scoreboard row's live nodes — rebuilt per episode cast, updated in place. */
interface ScoreRow {
  root: HTMLDivElement;
  hpFill: HTMLElement;
  kills: HTMLElement;
  deaths: HTMLElement;
  chips: HTMLElement;
  chipKey: string;
}

/**
 * Watch-lane lifecycle: started on init (live from the start, like big-two's
 * spectator — the hidden tab just skips canvas/audio and throttles the DOM),
 * kept alive across tab switches, restarted on every toolbar change.
 */
export class WatchController {
  private lang: Lang = "en";
  /** Bumped on every (re)start request; anything older is dropped. */
  private gen = 0;
  /** True once THIS generation's {arena} arrived — the gate that keeps frames
   *  from a superseded lane, and pre-arena frames of the new one, off-screen. */
  private live = false;
  private visible = false;
  private paused = false;
  /** Last result, kept so a language switch can re-render the banner text. */
  private ended: EpisodeEndMsg | null = null;
  private lastSnapshot: Snapshot | null = null;
  private metrics: Metrics | null = null;
  private renderer: SceneRenderer | null = null;
  private rows = new Map<number, ScoreRow>();
  private rowIds = "";
  private lastHiddenPaint = 0;
  private audioTick = -1;
  private readonly seenEvents = new Set<string>();
  private speedTimer: number | null = null;
  private raf = 0;
  private root: HTMLElement | null = null;
  private unsubs: (() => void)[] = [];

  constructor(
    private worker: WorkerBackendController,
    private audio: GameAudio,
  ) {}

  /** Paint pump. Runs continuously (even hidden): frame() early-returns its
   *  PAINTING when the tab is hidden but still ages the effects clock, so
   *  queued bursts expire instead of all popping when the tab returns. */
  private readonly loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    this.renderer?.frame(now);
  };

  /** Attach listeners, build the renderer, start the lane. MUST run after the
   *  section is mounted — byId throws on a missing id by design (wiring bug). */
  init(): void {
    this.root = byId<HTMLElement>("panel-watch");
    this.renderer = new SceneRenderer(byId<HTMLCanvasElement>("watch-canvas"));

    byId("watch-new").addEventListener("click", () => this.start());

    const tanks = byId<HTMLSelectElement>("watch-tanks");
    tanks.addEventListener("change", () => this.start());

    const speed = byId<HTMLInputElement>("watch-speed");
    speed.addEventListener("input", () => {
      byId("watch-speed-out").textContent = `×${String(speed.valueAsNumber)}`;
      if (this.speedTimer !== null) clearTimeout(this.speedTimer);
      this.speedTimer = window.setTimeout(() => {
        this.speedTimer = null;
        this.start(); // same lane slot; the worker adjusts its pacing
      }, SPEED_DEBOUNCE_MS);
    });

    byId("watch-pause").addEventListener("click", () => {
      this.paused = !this.paused;
      this.worker.pause(this.paused, "watch");
      if (this.paused) this.audio.syncEngines(ENGINES_OFF); // loops follow the lane
      byId<HTMLButtonElement>("watch-step").disabled = !this.paused;
      this.renderStatic();
    });
    // Guard anyway: step is a pause-only affordance.
    byId("watch-step").addEventListener("click", () => {
      if (this.paused) this.worker.step();
    });

    this.unsubs = [
      this.worker.onSnapshot("watch", (s) => this.onSnapshot(s)),
      this.worker.onArena("watch", (a) => this.onArena(a)),
      this.worker.onEpisodeEnd("watch", (e) => this.onEpisodeEnd(e)),
      this.worker.onMetrics((m) => this.onMetrics(m)),
    ];

    this.raf = requestAnimationFrame(this.loop);
    // Live from the start: the lane runs behind the default Train tab with
    // canvas/audio skipped and is simply there when the user switches over.
    void this.start();
  }

  // --- lane control -----------------------------------------------------------

  /** (Re)start the spectator lane with the toolbar's current settings. Every
   *  path through here bumps `gen`, which is what invalidates in-flight
   *  frames — the same job big-two's newGame() does with its advice requests. */
  private async start(): Promise<void> {
    const gen = ++this.gen;
    this.live = false;
    this.ended = null;
    this.lastSnapshot = null;
    this.rows.clear();
    this.rowIds = "";
    this.seenEvents.clear();
    this.audioTick = -1;
    byId("watch-banner").hidden = true;

    const tankCount = Number(byId<HTMLSelectElement>("watch-tanks").value);
    const speedHz = byId<HTMLInputElement>("watch-speed").valueAsNumber;
    try {
      const resp = await this.worker.watchStart(tankCount, speedHz);
      if (gen !== this.gen) return; // superseded while in flight
      if (isError(resp)) {
        console.warn("[watch] start failed:", resp.error);
        this.showBanner(t(this.lang, "watch.error"));
        return;
      }
      // The ack may itself carry the first {arena} (id-keyed response) or the
      // worker may push it on id 0 — both paths land in onArena().
      if (isArena(resp)) this.onArena(resp.arena);
    } catch {
      // A dead worker must not leave the panel silently frozen.
      if (gen === this.gen) this.showBanner(t(this.lang, "watch.error"));
    }
    // Re-assert the pause a restarted lane must inherit, addressed to THIS
    // lane; FIFO message order puts it after the start request.
    if (gen === this.gen && this.paused) this.worker.pause(true, "watch");
  }

  // --- worker streams ---------------------------------------------------------

  private onArena(a: ArenaMsg): void {
    // No id to correlate: an {arena} arriving now is either this generation's
    // first layout or a straggler from the lane the user just replaced — at
    // worst one frame of the old world, healed when the new lane's own arena
    // lands a message later.
    this.live = true;
    this.ended = null;
    this.rowIds = ""; // the new cast rebuilds the scoreboard on first frame
    byId("watch-banner").hidden = true;
    this.renderer?.setArena(a);
  }

  private onSnapshot(s: Snapshot): void {
    if (!this.live) return; // pre-arena, or a superseded generation's frame
    // Smoke hook: the Playwright watch spec advances/polls the lane through it.
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

  private onEpisodeEnd(e: EpisodeEndMsg): void {
    if (!this.live) return; // a superseded lane's result
    this.ended = e;
    this.showBanner(this.bannerText(e));
    if (!byId<HTMLInputElement>("watch-auto").checked) {
      // Auto-restart belongs to the worker; unchecked, we stop the lane here
      // and drop frames until the user asks for another match.
      this.live = false;
      this.worker.watchStop();
      this.audio.syncEngines(ENGINES_OFF);
    }
  }

  /** Banner text for an episode end: a named winner, or the draw line. */
  private bannerText(e: EpisodeEndMsg): string {
    return e.winner === null
      ? t(this.lang, "watch.banner.draw")
      : t(this.lang, "watch.banner.winner", { name: this.tankName(e.winner) });
  }

  /** Fleet metrics push. Watch has no tile of its own; kept for API parity
   *  and so the panel could grow one without a re-wire. */
  onMetrics(m: Metrics): void {
    this.metrics = m;
  }
  /** Latest fleet metrics — exposed for tests/diagnostics. */
  get lastMetrics(): Metrics | null {
    return this.metrics;
  }

  // --- painting ---------------------------------------------------------------

  private paint(s: Snapshot): void {
    this.paintRows(s.tanks);
    this.paintEsc(s);
    byId("watch-gen").textContent = `G${this.gen}·E${s.ep}`;
    if (!this.visible) return;
    this.audio.syncEngines(s);
    this.pumpAudio(s);
  }

  private paintRows(tanks: SnapshotTank[]): void {
    const ids = tanks.map((tk) => tk.i).join(",");
    if (ids !== this.rowIds) {
      this.rowIds = ids;
      this.buildRows(tanks);
    }
    for (const tk of tanks) {
      const row = this.rows.get(tk.i);
      if (!row) continue;
      row.kills.textContent = String(tk.kills);
      row.deaths.textContent = String(tk.deaths);
      const frac = Math.max(0, Math.min(1, tk.hp / TANK_HP));
      row.hpFill.style.width = `${(frac * 100).toFixed(1)}%`;
      row.hpFill.style.background = frac <= 0.25 ? "var(--red)" : "var(--good)";
      row.root.classList.toggle("dead", !tk.alive);
      row.root.style.opacity = tk.alive ? "1" : "0.45";
      // Chips change rarely — diff on the joined kind list, not per frame.
      const chipKey = tk.powerups.join("+");
      if (chipKey !== row.chipKey) {
        row.chipKey = chipKey;
        row.chips.replaceChildren(
          ...tk.powerups.map((p) => el("span", { class: "badge", text: t(this.lang, puKey(p)) })),
        );
      }
    }
  }

  private buildRows(tanks: SnapshotTank[]): void {
    const host = byId("watch-score");
    host.replaceChildren();
    this.rows.clear();
    for (const tk of [...tanks].sort((a, b) => a.i - b.i)) {
      const row = el("div", {
        class: "score-row", "data-testid": `watch-tank-${tk.i}`,
        style: "display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)",
      });
      row.appendChild(
        el("i", {
          class: "swatch",
          style: `width:14px;height:14px;border-radius:3px;flex:none;background:${this.colorOf(tk.colorIdx)}`,
        }),
      );
      row.appendChild(el("span", { class: "name", text: this.tankName(tk.i), style: "min-width:56px;font-weight:600" }));
      const kills = el("span", { text: "0", style: "width:44px;font-variant-numeric:tabular-nums" });
      const deaths = el("span", { text: "0", style: "width:48px;font-variant-numeric:tabular-nums;color:var(--text-dim)" });
      const hpFill = el("i", { style: "display:block;height:100%;width:100%;background:var(--good)" });
      const hpBar = el("span", {
        style: "flex:1;height:8px;border-radius:999px;background:var(--bg-elev2);overflow:hidden",
      });
      hpBar.appendChild(hpFill);
      const chips = el("span", { style: "display:inline-flex;gap:4px;flex-wrap:wrap" });
      row.appendChild(kills);
      row.appendChild(deaths);
      row.appendChild(hpBar);
      row.appendChild(chips);
      host.appendChild(row);
      this.rows.set(tk.i, { root: row, hpFill, kills, deaths, chips, chipKey: "" });
    }
  }

  private paintEsc(s: Snapshot): void {
    // Nominal denominator: a shortened episodeSeconds config under-fills the
    // bar rather than overflowing it, while the phase still flips at the
    // engine's own fractions.
    const frac = Math.min(1, s.tick / (EPISODE_SECONDS_DEFAULT * TICK_HZ));
    byId("watch-esc-fill").style.width = `${(frac * 100).toFixed(1)}%`;
    byId("watch-esc").dataset.phase = String(s.phase);
    const label = byId("watch-esc-label");
    if (s.phase === 0) {
      label.hidden = true;
      return;
    }
    label.hidden = false;
    label.textContent = t(this.lang, s.phase === 1 ? "esc.phase1" : "esc.phase2");
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

  // --- chrome -----------------------------------------------------------------

  private showBanner(text: string): void {
    const banner = byId("watch-banner");
    banner.textContent = text;
    banner.hidden = false;
  }

  /** P1..P4 / 玩家一..四 by tank id (the human-less lane just numbers seats). */
  private tankName(i: number): string {
    return t(this.lang, nameKey(i));
  }

  private colorOf(idx: number): string {
    return TANK_COLORS[idx] ?? TANK_COLORS[0];
  }

  /** Text the controller owns outright: the pause label alternates two keys,
   *  which is exactly what a data-i18n binding cannot express. */
  private renderStatic(): void {
    byId<HTMLButtonElement>("watch-new").textContent = t(this.lang, "play.rematch");
    byId<HTMLButtonElement>("watch-pause").textContent = t(
      this.lang,
      this.paused ? "watch.resume" : "watch.pause",
    );
    byId<HTMLButtonElement>("watch-step").textContent = t(this.lang, "watch.step");
  }

  setLang(lang: Lang): void {
    this.lang = lang;
    this.renderStatic();
    // main.ts rebinds the whole document on toggle; this keeps the panel
    // correct standalone too (and re-renders after a late mount).
    this.root?.querySelectorAll<HTMLElement>("[data-i18n]").forEach((node) => {
      node.textContent = t(lang, node.dataset.i18n as StringKey);
    });
    for (const [id, row] of this.rows) {
      const name = row.root.querySelector<HTMLElement>(".name");
      if (name) name.textContent = this.tankName(id);
    }
    if (this.ended !== null) {
      this.showBanner(this.bannerText(this.ended));
    }
    const s = this.lastSnapshot;
    if (s !== null) this.paintEsc(s);
  }

  /** Tab-visibility hook from main.ts's switch handler. */
  setVisible(v: boolean): void {
    if (this.visible === v) return;
    this.visible = v;
    this.renderer?.setVisible(v);
    if (!v) {
      this.audio.syncEngines(ENGINES_OFF); // engine loops are silent for a hidden lane
      return;
    }
    // Activation re-asserts the lane's own pause state: nothing here ever
    // freezes the lane on hide, so a user-paused duel must still be paused
    // when its tab comes back — and no OTHER tab's hide/resume may have left
    // it in an unrequested state (the lane key on PauseReq makes that
    // impossible worker-side; this is the belt).
    if (this.paused) this.worker.pause(true, "watch");
    const s = this.lastSnapshot;
    // A paused lane must not come back humming: engine loops follow the
    // snapshot stream, and a paused lane streams none — so skip the repaint's
    // audio sync and let the next snapshot after Resume restart them.
    if (s !== null && this.live && !this.paused) this.paint(s);
  }

  /** Drop worker subscriptions, the paint pump and timers (teardown / HMR). */
  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs = [];
    cancelAnimationFrame(this.raf);
    if (this.speedTimer !== null) clearTimeout(this.speedTimer);
  }
}
