// header-controller.ts — owns the persistent header strip: the worker status
// chip (loading → ready → running, plus the tfjs backend name), the sound
// badge, the language toggle and the theme toggle.
//
// Contract with main.ts: main builds the header DOM (ids below are stable from
// the M1 shell) and passes the two decisions only it can make — flipping the
// app-wide language (saveLang + rebindI18n + every controller's setLang) and
// muting audio. This controller never touches localStorage itself; it renders
// whatever state the deps report and delegates the flips.
import { byId, el } from "../by-id";
import { t, type Lang } from "../i18n";
import { toggleTheme } from "../ui/theme";

/** Lifecycle of the worker, as the chip displays it. "running" implies the
 *  engine is alive (ready) AND a training run is in flight. */
export type HeaderState = "loading" | "ready" | "running";

/** The status text for each chip state, so setLang can re-render it without
 *  main.ts remembering which state it was in. */
const STATE_KEY: Record<HeaderState, "status.loading" | "status.ready" | "status.running"> = {
  loading: "status.loading",
  ready: "status.ready",
  running: "status.running",
};

export class HeaderController {
  private lang: Lang;
  private state: HeaderState = "loading";
  private backend: string | null = null;
  private soundOn: boolean;

  constructor(
    deps: {
      /** Called with the NEXT state after a sound-chip click (main owns the
       *  AudioContext; the chip only reports intent). */
      onToggleSound: (on: boolean) => void;
      initialSoundOn: boolean;
      /** Language flips belong to main (it must rebind every controller);
       *  the header only renders the resulting language. */
      onToggleLang: () => void;
    },
  ) {
    this.lang = "en";
    this.soundOn = deps.initialSoundOn;
    this.deps = deps;
  }

  private readonly deps: {
    onToggleSound: (on: boolean) => void;
    initialSoundOn: boolean;
    onToggleLang: () => void;
  };

  /**
   * Bind the header's four widgets. Call once after the shell DOM exists.
   * The chip's children are rebuilt here — this controller OWNS the chip's
   * content (dot · backend · text), so main.ts must not write into it.
   */
  init(): void {
    const chip = byId("status-chip");
    chip.replaceChildren(
      el("span", { class: "dot" }),
      el("span", { class: "backend", id: "chip-backend", "data-testid": "chip-backend" }),
      el("span", { id: "status-text", "data-testid": "status-text" }),
    );

    byId<HTMLButtonElement>("btn-lang").addEventListener("click", () => this.deps.onToggleLang());
    byId<HTMLButtonElement>("btn-theme").addEventListener("click", () => toggleTheme());

    // The sound widget is a span (badge), not a <button>, so Enter/Space are
    // wired by hand — keyboard users keep parity with the pointer click. Its
    // shell-era data-i18n is dropped: the label depends on sound state, which
    // a document-wide rebind cannot know.
    const sound = byId("chip-sound");
    sound.removeAttribute("data-i18n");
    sound.setAttribute("role", "button");
    sound.setAttribute("tabindex", "0");
    sound.addEventListener("click", () => this.toggleSound());
    sound.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.toggleSound();
      }
    });

    this.render();
  }

  setLang(lang: Lang): void {
    this.lang = lang;
    this.render();
  }

  /** Report the tfjs backend the worker selected ("webgpu"/"webgl"/…). */
  setBackend(name: string): void {
    this.backend = name;
    byId("chip-backend").textContent = name;
  }

  setState(state: HeaderState): void {
    this.state = state;
    const chip = byId("status-chip");
    // "running" keeps .ready too: the engine is obviously alive if it trains.
    chip.classList.toggle("ready", state !== "loading");
    chip.classList.toggle("running", state === "running");
    this.renderText();
  }

  // --- internals -------------------------------------------------------------

  private toggleSound(): void {
    this.soundOn = !this.soundOn;
    this.deps.onToggleSound(this.soundOn);
    this.renderSound();
  }

  private render(): void {
    byId<HTMLButtonElement>("btn-lang").textContent = this.lang === "en" ? "中文" : "EN";
    byId("btn-lang").setAttribute("aria-label", t(this.lang, "header.language"));
    byId("btn-theme").setAttribute("aria-label", t(this.lang, "header.theme"));
    this.renderText();
    this.renderSound();
  }

  /** Chip text = optional backend prefix + the state's localized label. */
  private renderText(): void {
    const label = t(this.lang, STATE_KEY[this.state]);
    byId("status-text").textContent = this.backend ? `${this.backend} · ${label}` : label;
  }

  private renderSound(): void {
    const sound = byId("chip-sound");
    sound.textContent = t(this.lang, this.soundOn ? "header.sound" : "header.soundOff");
    sound.classList.toggle("on", this.soundOn);
  }
}
