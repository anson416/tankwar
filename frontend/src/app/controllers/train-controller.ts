// train-controller.ts — the Train tab: start/stop toggle, twelve stat tiles,
// four SVG charts, and the NetConfig form.
//
// CONTRACT with main.ts (the shell agent): buildTrainSection() builds the whole
// <section id="panel-train"> — every id/testid listed below — and main appends
// it to the tab panel container; TrainController then binds into it by id.
// main.ts must not create duplicates of these ids, must not write into them
// after handing over, and gets live telemetry either by letting init()
// subscribe to the worker itself (the default) or by calling onMetrics() —
// never both, or chart samples would be pushed twice per push.
//
// Metric honesty (plan §UI): the win-rate chart carries a reference line at
// 1/N — the honest bar for an N-tank FFA — and loss is drawn only as a
// smoothed trace, never the headline.
import { byId, el } from "../by-id";
import { drawChart } from "../ui/chart-svg";
import { RingBuffer } from "../ui/ring-buffer";
import { t, type Lang, type StringKey } from "../i18n";
import type { WorkerBackendController } from "../worker-client";
import { defaultConfig, isError, type Metrics, type NetConfig } from "../../train-protocol";

const BUF_CAP = 240;

// ===== Validation ranges (plan §UI table) — the worker re-validates, but a
// config that cannot be built must never even reach the Apply click. =====
const R = {
  hiddenMin: 16,
  hiddenMax: 1024,
  hiddenLayersMax: 4,
  lr: [1e-5, 1e-2],
  batch: [32, 1024],
  buffer: [10_000, 80_000], // rows × 567 floats × 4 B ≈ 91 MB at the default 40k
  gamma: [0.9, 0.995],
  epsDecay: [0.9, 0.9999],
  epsMin: [0.01, 0.3],
  lanes: [1, 32],
  trainEvery: [32, 1024],
  targetSync: [100, 10_000],
  evalEvery: [20, 1000],
  evalEpisodes: [4, 100],
  episodeSeconds: [15, 90],
  tankCount: [2, 4],
} as const;

/** Every form control the Apply path reads — revalidate on any of them. */
const CONFIG_IDS = [
  "cfg-arch", "cfg-hidden", "cfg-lr", "cfg-batch", "cfg-buffer", "cfg-gamma",
  "cfg-epsdecay", "cfg-epsmin", "cfg-lanes", "cfg-tankcount", "cfg-episodeseconds",
  "cfg-trainevery", "cfg-targetsync", "cfg-evalevery", "cfg-evalepisodes", "cfg-doubleq",
] as const;

// ===== Section builder (called by main.ts before the controller exists) =====

/** Element whose text is bound to an i18n key (rebound wholesale by main's
 *  rebindI18n on a language switch; dynamic text has NO data-i18n). */
function i18nEl(
  tag: keyof HTMLElementTagNameMap,
  key: StringKey,
  attrs: Record<string, string | number | boolean> = {},
): HTMLElement {
  const node = el(tag, attrs) as HTMLElement;
  node.dataset.i18n = key;
  return node;
}

function chartSvg(id: string): SVGSVGElement {
  return el("svg", {
    class: "chart", id, "data-testid": id,
    viewBox: "0 0 200 60", preserveAspectRatio: "none",
  });
}

function chartBlock(titleKey: StringKey, id: string): HTMLElement {
  const block = el("div", { class: "chart-block" });
  block.appendChild(i18nEl("span", titleKey, { class: "chart-title" }));
  block.appendChild(chartSvg(id));
  return block;
}

function statTile(labelKey: StringKey, id: string, subId?: string): HTMLElement {
  const d = el("div", { class: "tile", "data-testid": `tile-${id}` });
  d.appendChild(i18nEl("span", labelKey, { class: "tile-label" }));
  d.appendChild(el("span", { class: "tile-value", id, "data-testid": id, text: "—" }));
  // Only the win-rate tile carries a sublabel (the "chance = 1/N" bar).
  if (subId) d.appendChild(el("span", { class: "tile-sub", id: subId, "data-testid": subId }));
  return d;
}

function numField(grid: HTMLElement, id: string, key: StringKey, step: string): void {
  const f = el("div", { class: "field" });
  f.appendChild(i18nEl("label", key, { class: "field-label", for: id }));
  f.appendChild(el("input", { class: "num", id, type: "number", step, "data-testid": id }));
  grid.appendChild(f);
}

function textField(grid: HTMLElement, id: string, key: StringKey, placeholder: string): void {
  const f = el("div", { class: "field" });
  f.appendChild(i18nEl("label", key, { class: "field-label", for: id }));
  f.appendChild(
    el("input", { class: "num", id, type: "text", inputmode: "numeric", placeholder, "data-testid": id }),
  );
  grid.appendChild(f);
}

/**
 * The whole Train tab DOM: control row → 12 stat tiles → 4 charts → config
 * form. Ids the controller binds: train-start, train-backend, stat-*,
 * stat-winrate-sub, chart-return/winrate/eps/gps, winrate-legend, cfg-*,
 * btn-apply-config, config-msg.
 */
export function buildTrainSection(): HTMLElement {
  const s = el("section", {
    class: "tab-section", id: "panel-train", "data-testid": "panel-train",
    role: "tabpanel", "aria-labelledby": "tab-train",
  });

  // --- control row: start/stop + the worker's tfjs backend badge ---
  const ctl = el("div", { class: "panel" });
  const row = el("div", { class: "action-row" });
  row.appendChild(
    el("button", { class: "btn primary", id: "train-start", "data-testid": "train-start", type: "button" }),
  );
  row.appendChild(
    el("span", { class: "badge", id: "train-backend", "data-testid": "train-backend", text: "…" }),
  );
  ctl.appendChild(row);
  s.appendChild(ctl);

  // --- stat tiles ---
  const stats = el("div", { class: "panel" });
  stats.appendChild(i18nEl("h2", "train.stats", { class: "panel-title" }));
  const tiles = el("div", { class: "tiles" });
  tiles.appendChild(statTile("train.backend", "stat-backend"));
  tiles.appendChild(statTile("train.episodes", "stat-episodes"));
  tiles.appendChild(statTile("train.decisions", "stat-decisions"));
  tiles.appendChild(statTile("train.trainSteps", "stat-trainsteps"));
  tiles.appendChild(statTile("train.epsilon", "stat-epsilon"));
  tiles.appendChild(statTile("train.loss", "stat-loss"));
  tiles.appendChild(statTile("train.avgReturn", "stat-avgreturn"));
  tiles.appendChild(statTile("train.winRate", "stat-winrate", "stat-winrate-sub"));
  tiles.appendChild(statTile("train.kd", "stat-kd"));
  tiles.appendChild(statTile("train.gamesPerSec", "stat-gps"));
  tiles.appendChild(statTile("train.aliveLanes", "stat-alive"));
  tiles.appendChild(statTile("train.bufferFill", "stat-buffer"));
  stats.appendChild(tiles);
  s.appendChild(stats);

  // --- charts (drawChart draws baseline + optional refline + one polyline) ---
  const charts = el("div", { class: "panel" });
  charts.appendChild(i18nEl("h2", "train.charts", { class: "panel-title" }));
  charts.appendChild(chartBlock("train.avgReturn", "chart-return"));
  // Win rate is the headline: its legend (not a plain title) names the trace
  // and the dashed 1/N chance line the controller keeps in sync with the
  // applied tank count.
  const winBlock = el("div", { class: "chart-block" });
  winBlock.appendChild(el("div", { class: "legend", id: "winrate-legend" }));
  winBlock.appendChild(chartSvg("chart-winrate"));
  charts.appendChild(winBlock);
  charts.appendChild(chartBlock("train.epsilon", "chart-eps"));
  charts.appendChild(chartBlock("train.gamesPerSec", "chart-gps"));
  s.appendChild(charts);

  // --- config form: defaults filled by the controller from defaultConfig() ---
  const cfg = el("div", { class: "panel" });
  cfg.appendChild(i18nEl("h2", "train.config.title", { class: "panel-title" }));
  const grid = el("div", { class: "config-grid" });

  const archField = el("div", { class: "field" });
  archField.appendChild(i18nEl("label", "train.config.arch", { class: "field-label", for: "cfg-arch" }));
  const arch = el("select", { class: "num", id: "cfg-arch", "data-testid": "cfg-arch" });
  arch.appendChild(i18nEl("option", "train.config.arch.mlp", { value: "mlp" }));
  arch.appendChild(i18nEl("option", "train.config.arch.attention", { value: "attention" }));
  archField.appendChild(arch);
  grid.appendChild(archField);

  // hidden is number[] → comma-separated text ("256,256"); parse + validate
  // client-side (each 16–1024, ≤4 layers) before anything reaches the worker.
  textField(grid, "cfg-hidden", "train.config.hidden", "256,256");
  numField(grid, "cfg-lr", "train.config.lr", "0.00001");
  numField(grid, "cfg-batch", "train.config.batch", "1");
  numField(grid, "cfg-buffer", "train.config.buffer", "1000");
  numField(grid, "cfg-gamma", "train.config.gamma", "0.005");
  numField(grid, "cfg-epsdecay", "train.config.epsDecay", "0.001");
  numField(grid, "cfg-epsmin", "train.config.epsMin", "0.01");
  numField(grid, "cfg-lanes", "train.config.lanes", "1");
  numField(grid, "cfg-tankcount", "train.config.tankCount", "1");
  numField(grid, "cfg-episodeseconds", "train.config.episodeSeconds", "1");
  numField(grid, "cfg-trainevery", "train.config.trainEvery", "1");
  numField(grid, "cfg-targetsync", "train.config.targetSync", "100");
  numField(grid, "cfg-evalevery", "train.config.evalEvery", "1");
  numField(grid, "cfg-evalepisodes", "train.config.evalEpisodes", "1");
  cfg.appendChild(grid);

  const dq = el("label", {
    class: "field-label", for: "cfg-doubleq",
    // Inline flex (the big-two pattern) — CSS owns no checkbox-row class.
    style: "display:flex;gap:6px;align-items:center",
  });
  dq.appendChild(el("input", { type: "checkbox", id: "cfg-doubleq", "data-testid": "cfg-doubleq" }));
  dq.appendChild(i18nEl("span", "train.config.doubleQ", {}));
  cfg.appendChild(dq);

  cfg.appendChild(i18nEl("p", "train.config.note", { class: "hint" }));
  cfg.appendChild(
    i18nEl("button", "train.config.apply", {
      class: "btn primary", id: "btn-apply-config", "data-testid": "apply-config", type: "button",
    }),
  );
  cfg.appendChild(el("p", { class: "setup-error", id: "config-msg", "data-testid": "config-msg" }));
  s.appendChild(cfg);
  return s;
}

// ===== Helpers shared by the redraw path =====

/** 1234567 → "1.23M", 12345 → "12.3k" — tiles stay one line on a phone. */
function fmtInt(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

/** Signed-min/max scale that always includes 0 (avg return swings negative
 *  early: death −60 vs survival +27), padded 5% so the trace never kisses the
 *  frame. Mirrors the reference loss-chart rescaling. */
function spreadScale(buf: RingBuffer<number>): { minY: number; maxY: number } {
  let lo = 0;
  let hi = 1e-6;
  for (const v of buf.values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const pad = Math.max(1e-6, (hi - lo) * 0.05);
  return { minY: lo - pad, maxY: hi + pad };
}

export class TrainController {
  private lang: Lang = "en";
  private running = false;
  /** Last config the worker ACCEPTED — drives the 1/N chance line and the
   *  win-rate sublabel (Metrics carries no tankCount of its own). */
  private applied: NetConfig = defaultConfig();
  /** What #config-msg currently shows, so a language switch can re-render it. */
  private msg: "none" | "invalid" | "note" = "none";

  private readonly bufReturn = new RingBuffer<number>(BUF_CAP);
  private readonly bufWin = new RingBuffer<number>(BUF_CAP);
  private readonly bufEps = new RingBuffer<number>(BUF_CAP);
  private readonly bufGps = new RingBuffer<number>(BUF_CAP);

  constructor(private worker: WorkerBackendController) {}

  /** Bind the section built by buildTrainSection(). Subscribes to the worker's
   *  metric pushes itself — main.ts does not need to forward them. */
  init(): void {
    byId<HTMLButtonElement>("train-start").addEventListener("click", () =>
      this.running ? this.stop() : this.start(),
    );
    byId<HTMLButtonElement>("btn-apply-config").addEventListener("click", () => this.applyConfig());

    this.fillConfigDefaults();
    // Both events: number/text inputs fire "input", the select and checkbox
    // fire "change" — listening to both keeps Apply's disabled state exact.
    for (const id of CONFIG_IDS) {
      byId<HTMLElement>(id).addEventListener("input", () => this.validate());
      byId<HTMLElement>(id).addEventListener("change", () => this.validate());
    }
    this.validate();
    this.renderDynamic();

    this.worker.onMetrics((m) => this.onMetrics(m));
  }

  setLang(lang: Lang): void {
    this.lang = lang;
    this.renderDynamic();
  }

  // --- start/stop ---------------------------------------------------------------

  /** Public so main.ts can auto-start on boot (SPEC §9 pattern) if it wants. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.worker.control({ start: true });
    this.renderDynamic();
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    this.worker.control({ stop: true });
    this.renderDynamic();
  }

  get isRunning(): boolean {
    return this.running;
  }

  // --- metrics ------------------------------------------------------------------

  /** One telemetry push → tiles + chart samples. Idempotent per push: call it
   *  either from init()'s subscription or from main.ts, never both. */
  onMetrics(m: Metrics): void {
    byId("stat-backend").textContent = m.backend;
    byId("stat-episodes").textContent = fmtInt(m.episodes);
    byId("stat-decisions").textContent = fmtInt(m.decisions);
    byId("stat-trainsteps").textContent = fmtInt(m.trainSteps);
    byId("stat-epsilon").textContent = m.epsilon.toFixed(3);
    byId("stat-loss").textContent = m.smoothedLoss.toFixed(3);
    byId("stat-avgreturn").textContent = m.avgReturn.toFixed(1);
    byId("stat-winrate").textContent = `${Math.round(m.winRateVsRandom * 100)}%`;
    byId("stat-kd").textContent = m.killDeathRatio.toFixed(2);
    byId("stat-gps").textContent = m.gamesPerSec.toFixed(2);
    byId("stat-alive").textContent = String(m.aliveLanes);
    byId("stat-buffer").textContent = `${Math.round(m.bufferFill * 100)}%`;

    this.bufReturn.push(m.avgReturn);
    this.bufWin.push(m.winRateVsRandom);
    this.bufEps.push(m.epsilon);
    this.bufGps.push(m.gamesPerSec);
    this.redrawCharts();
  }

  /** Drop every curve — called after a successful Apply (a fresh net's history
   *  is meaningless) and available to main for a session reset. */
  resetCharts(): void {
    this.bufReturn.clear();
    this.bufWin.clear();
    this.bufEps.clear();
    this.bufGps.clear();
    this.redrawCharts();
  }

  // --- config form ----------------------------------------------------------------

  private fillConfigDefaults(): void {
    const d = defaultConfig();
    byId<HTMLSelectElement>("cfg-arch").value = d.architecture;
    byId<HTMLInputElement>("cfg-hidden").value = d.hidden.join(",");
    byId<HTMLInputElement>("cfg-lr").value = String(d.learningRate);
    byId<HTMLInputElement>("cfg-batch").value = String(d.batchSize);
    byId<HTMLInputElement>("cfg-buffer").value = String(d.bufferSize);
    byId<HTMLInputElement>("cfg-gamma").value = String(d.gamma);
    byId<HTMLInputElement>("cfg-epsdecay").value = String(d.epsDecay);
    byId<HTMLInputElement>("cfg-epsmin").value = String(d.epsMin);
    byId<HTMLInputElement>("cfg-lanes").value = String(d.lanes);
    byId<HTMLInputElement>("cfg-tankcount").value = String(d.tankCount);
    byId<HTMLInputElement>("cfg-episodeseconds").value = String(d.episodeSeconds);
    byId<HTMLInputElement>("cfg-trainevery").value = String(d.trainEvery);
    byId<HTMLInputElement>("cfg-targetsync").value = String(d.targetSync);
    byId<HTMLInputElement>("cfg-evalevery").value = String(d.evalEvery);
    byId<HTMLInputElement>("cfg-evalepisodes").value = String(d.evalEpisodes);
    byId<HTMLInputElement>("cfg-doubleq").checked = d.doubleQ;
  }

  private readInt(id: string, min: number, max: number): number {
    const v = byId<HTMLInputElement>(id).valueAsNumber;
    if (!Number.isFinite(v) || v < min || v > max) throw new Error(id);
    return Math.floor(v);
  }

  private readFloat(id: string, min: number, max: number): number {
    const v = byId<HTMLInputElement>(id).valueAsNumber;
    if (!Number.isFinite(v) || v < min || v > max) throw new Error(id);
    return v;
  }

  /** "256,256" → [256, 256]; every layer 16–1024, at most 4 layers. Throws the
   *  offending input's id (the big-two idiom — the catch marks that field). */
  private readHidden(): number[] {
    const raw = byId<HTMLInputElement>("cfg-hidden").value.trim();
    if (raw === "") throw new Error("cfg-hidden");
    const out: number[] = [];
    for (const part of raw.split(",")) {
      const v = Number(part.trim());
      if (!Number.isInteger(v) || v < R.hiddenMin || v > R.hiddenMax) throw new Error("cfg-hidden");
      out.push(v);
    }
    if (out.length < 1 || out.length > R.hiddenLayersMax) throw new Error("cfg-hidden");
    return out;
  }

  private tryReadConfig(): { ok: true; cfg: NetConfig } | { ok: false; badId: string } {
    // Fields the form deliberately does not expose (attention geometry, ε
    // start) ride along from the defaults — the attention encoder fixes its
    // own shape and ε always starts at 1 per plan.
    const d = defaultConfig();
    try {
      const cfg: NetConfig = {
        architecture: byId<HTMLSelectElement>("cfg-arch").value === "attention" ? "attention" : "mlp",
        hidden: this.readHidden(),
        dModel: d.dModel,
        heads: d.heads,
        layers: d.layers,
        ffn: d.ffn,
        learningRate: this.readFloat("cfg-lr", R.lr[0], R.lr[1]),
        batchSize: this.readInt("cfg-batch", R.batch[0], R.batch[1]),
        bufferSize: this.readInt("cfg-buffer", R.buffer[0], R.buffer[1]),
        gamma: this.readFloat("cfg-gamma", R.gamma[0], R.gamma[1]),
        epsStart: d.epsStart,
        epsDecay: this.readFloat("cfg-epsdecay", R.epsDecay[0], R.epsDecay[1]),
        epsMin: this.readFloat("cfg-epsmin", R.epsMin[0], R.epsMin[1]),
        lanes: this.readInt("cfg-lanes", R.lanes[0], R.lanes[1]),
        trainEvery: this.readInt("cfg-trainevery", R.trainEvery[0], R.trainEvery[1]),
        targetSync: this.readInt("cfg-targetsync", R.targetSync[0], R.targetSync[1]),
        evalEvery: this.readInt("cfg-evalevery", R.evalEvery[0], R.evalEvery[1]),
        evalEpisodes: this.readInt("cfg-evalepisodes", R.evalEpisodes[0], R.evalEpisodes[1]),
        episodeSeconds: this.readInt("cfg-episodeseconds", R.episodeSeconds[0], R.episodeSeconds[1]),
        tankCount: this.readInt("cfg-tankcount", R.tankCount[0], R.tankCount[1]),
        doubleQ: byId<HTMLInputElement>("cfg-doubleq").checked,
      };
      return { ok: true, cfg };
    } catch (err) {
      return { ok: false, badId: err instanceof Error ? err.message : "cfg-hidden" };
    }
  }

  /** Revalidate everything; Apply is the ONLY path a config reaches the
   *  worker, and it stays disabled while any field is out of range. */
  private validate(): void {
    for (const id of CONFIG_IDS) byId<HTMLElement>(id).classList.remove("invalid");
    const res = this.tryReadConfig();
    if (res.ok) {
      byId<HTMLButtonElement>("btn-apply-config").disabled = false;
      if (this.msg === "invalid") this.msg = "none";
    } else {
      byId<HTMLElement>(res.badId).classList.add("invalid");
      byId<HTMLButtonElement>("btn-apply-config").disabled = true;
      this.msg = "invalid";
    }
    this.renderMsg();
  }

  private applyConfig(): void {
    const res = this.tryReadConfig();
    if (!res.ok) {
      this.validate();
      return;
    }
    // request(), not control(): the worker's rebuild is transactional and acks
    // with fresh metrics on the request id — or an ErrorResp if the new net
    // could not be built (previous run stays intact either way).
    void this.worker.applyConfig(res.cfg).then(
      (resp) => {
        if (isError(resp)) {
          this.msg = "invalid";
        } else {
          this.applied = res.cfg;
          this.msg = "note";
          this.resetCharts();
          this.renderDynamic(); // chance line + sublabel follow the new tankCount
        }
        this.renderMsg();
      },
      () => {
        // Rejected without a response = the worker died mid-rebuild. "invalid"
        // is the only error string the tab has; Apply stays available to retry.
        this.msg = "invalid";
        this.renderMsg();
      },
    );
  }

  // --- rendering ------------------------------------------------------------------

  private redrawCharts(): void {
    drawChart(byId<SVGSVGElement>("chart-return"), this.bufReturn, spreadScale(this.bufReturn));
    drawChart(byId<SVGSVGElement>("chart-winrate"), this.bufWin, {
      minY: 0,
      maxY: 1,
      refLine: 1 / this.applied.tankCount,
    });
    drawChart(byId<SVGSVGElement>("chart-eps"), this.bufEps, { minY: 0, maxY: 1 });
    const gps = this.bufGps.values;
    let maxGps = 1e-6;
    for (const v of gps) if (v > maxGps) maxGps = v;
    drawChart(byId<SVGSVGElement>("chart-gps"), this.bufGps, { minY: 0, maxY: maxGps * 1.1 });
  }

  /** Every piece of text this controller owns (i.e. NOT the [data-i18n] static
   *  labels — main's rebindI18n covers those). */
  private renderDynamic(): void {
    const btn = byId<HTMLButtonElement>("train-start");
    btn.textContent = t(this.lang, this.running ? "train.stop" : "train.start");
    btn.className = this.running ? "btn stop" : "btn primary";
    byId("stat-winrate-sub").textContent = t(this.lang, "train.winRateChance", {
      chance: this.applied.tankCount,
    });
    this.renderLegend();
    this.renderMsg();
  }

  private renderLegend(): void {
    const legend = byId("winrate-legend");
    legend.replaceChildren();
    const trace = el("span", {});
    trace.appendChild(el("i", { class: "swatch s-main" }));
    trace.appendChild(document.createTextNode(t(this.lang, "train.winRate")));
    legend.appendChild(trace);
    legend.appendChild(el("span", { text: `— — ${t(this.lang, "train.chanceLevel")}` }));
  }

  private renderMsg(): void {
    byId("config-msg").textContent =
      this.msg === "none" ? "" : t(this.lang, this.msg === "invalid" ? "train.config.invalid" : "train.config.note");
  }
}
