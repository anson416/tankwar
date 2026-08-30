// main.ts — bootstrap and the LAST wiring point: everything meets here. Builds
// the static shell (header + 3-tab nav) at runtime — index.html is a single
// #app mount — then mounts the three controllers' sections and connects the
// shared services (one train worker, one GameAudio graph, the header).
//
// Wiring map (who subscribes to what — the controllers document this contract
// in their headers, and it is load-bearing):
//   worker.onMetrics → Train/Watch/Play subscribe in their own init(); main
//     adds ONE extra subscription that drives the header chip and the
//     __engineReady smoke hook. Forwarding pushes to a controller's onMetrics
//     here as well would sample every push twice (the train chart would grow
//     two points per push).
//   worker.onArena/onSnapshot/onEpisodeEnd → Watch/Play subscribe in init();
//     main forwards nothing.
//   main → controllers: init() once after mount, setLang() on a language
//     flip, setVisible() from the tab switch handler.
//
// Boot order: theme → shell DOM → controllers → auto-start training (SPEC §9:
// the fleet trains from scratch on every visit so it improves while the user
// plays or watches; the Train tab's Stop button is the escape hatch).
// `window.__engineReady` flips on the first metrics push and `__appReady` once
// every controller is wired — the smoke tests poll both. `?test=1` changes
// nothing here: the WORKER self-configures its tiny smoke fixture, the shell
// proceeds exactly as in a normal session.
import "./style.css";
import { byId, el } from "./app/by-id";
import { loadLang, saveLang, t, type Lang, type StringKey } from "./app/i18n";
import { HeaderController } from "./app/controllers/header-controller";
import { buildPlaySection, PlayController } from "./app/controllers/play-controller";
import { buildTrainSection, TrainController } from "./app/controllers/train-controller";
import { buildWatchSection, WatchController } from "./app/controllers/watch-controller";
import { GameAudio } from "./app/ui/audio";
// toggleTheme is NOT bound here: HeaderController.init() owns the theme button
// (with the language button), so a second binding would flip the theme twice.
import { initTheme } from "./app/ui/theme";
import { WorkerBackendController } from "./app/worker-client";

declare global {
  interface Window {
    /** Smoke-test hook: true once the first worker metrics push arrives. */
    __engineReady?: boolean;
    /** Smoke-test hook: true once shell + all three controllers are wired. */
    __appReady?: boolean;
  }
}

type TabId = "train" | "play" | "watch";
/** Train first (default), matching the panel mount order below. */
const TABS: readonly { id: TabId; key: StringKey }[] = [
  { id: "train", key: "tab.train" },
  { id: "play", key: "tab.play" },
  { id: "watch", key: "tab.watch" },
];

/** Element whose text is bound to an i18n key (static labels only — re-bound
 *  wholesale on a language switch by rebindI18n). */
function i18nEl(
  tag: keyof HTMLElementTagNameMap,
  key: StringKey,
  attrs: Record<string, string | number | boolean> = {},
): HTMLElement {
  const node = el(tag, attrs) as HTMLElement;
  node.dataset.i18n = key;
  return node;
}

function buildShell(): void {
  const app = byId("app");

  // --- header: title · status chip · spacer · sound/lang/theme ---
  const header = el("header", { class: "app-header" });
  header.appendChild(i18nEl("h1", "app.title", { class: "app-title", "data-testid": "app-title" }));
  header.appendChild(i18nEl("p", "app.subtitle", { class: "app-subtitle" }));
  // Persistent status chip (worker liveness + tfjs backend). The placeholder
  // children here only cover the pre-init paint — HeaderController.init()
  // rebuilds the chip's content and owns it from then on.
  const chip = el("span", { class: "status-chip", id: "status-chip", "data-testid": "status-chip" });
  chip.appendChild(el("span", { class: "dot" }));
  chip.appendChild(i18nEl("span", "status.loading", { id: "status-text", "data-testid": "status-text" }));
  header.appendChild(chip);
  header.appendChild(el("span", { class: "header-spacer" }));
  // Sound chip: a badge acting as a button. Its ID is fixed by
  // HeaderController (it wires Enter/Space and owns the label/state class).
  header.appendChild(
    i18nEl("span", "header.soundOff", { class: "badge", id: "chip-sound", "data-testid": "sound-chip" }),
  );
  header.appendChild(
    el("button", {
      class: "icon-btn", id: "btn-lang", "data-testid": "lang-toggle",
      type: "button", "aria-label": "language", title: "Language / 語言", text: "中文",
    }),
  );
  header.appendChild(
    el("button", {
      class: "icon-btn", id: "btn-theme", "data-testid": "theme-toggle",
      type: "button", "aria-label": "theme", title: "Theme / 主題", text: "◐",
    }),
  );

  // --- tab bar + one controller-built panel per mode ---
  const tabbar = el("nav", { class: "tabbar", role: "tablist" });
  const main = el("main");
  for (const { id, key } of TABS) {
    tabbar.appendChild(
      i18nEl("button", key, {
        class: "tab", type: "button", role: "tab",
        id: `tab-${id}`, "data-testid": `tab-${id}`,
        "aria-controls": `panel-${id}`, "aria-selected": "false",
      }),
    );
  }
  // Panels come from the controllers' builders (single source of every id and
  // testid inside a mode). buildTrainSection leaves its section visible — the
  // default tab — while play/watch build hidden, so this order IS the state.
  main.appendChild(buildTrainSection());
  main.appendChild(buildPlaySection());
  main.appendChild(buildWatchSection());

  app.appendChild(header);
  app.appendChild(tabbar);
  app.appendChild(main);
}

/** Refresh every [data-i18n] node. Dynamic text (chip, legends, banners, HUD)
 *  is NOT data-i18n — each controller re-renders its own on setLang(). */
function rebindI18n(lang: Lang): void {
  document.querySelectorAll<HTMLElement>("[data-i18n]").forEach((node) => {
    node.textContent = t(lang, node.dataset.i18n as StringKey);
  });
}

function main(): void {
  initTheme();
  let lang = loadLang();
  buildShell();
  rebindI18n(lang);

  // --- shared services: one worker for the whole app, one audio graph ---
  const worker = new WorkerBackendController();
  const audio = new GameAudio();

  // --- header (the two app-wide flips delegate back to main) ---
  const header = new HeaderController({
    initialSoundOn: true, // GameAudio boots enabled; the chip only reports intent
    onToggleSound: (on) => {
      void audio.unlock(); // a chip click IS a user gesture; unlock is idempotent
      audio.setEnabled(on);
    },
    onToggleLang: () => flipLang(),
  });

  // --- one controller per tab, binding into the sections mounted above ---
  const train = new TrainController(worker);
  const watch = new WatchController(worker, audio);
  const play = new PlayController(worker, audio); // builds its own TankInput
  header.init();
  train.init();
  watch.init(); // lane starts now: it runs behind the train tab, canvas/audio gated
  play.init(); // deliberately starts no match — the tab's first activation does

  // --- tabs: aria state, panel visibility, controller visibility hooks ---
  const selectTab = (id: TabId): void => {
    for (const { id: tabId } of TABS) {
      byId(`tab-${tabId}`).setAttribute("aria-selected", String(tabId === id));
      byId(`panel-${tabId}`).hidden = tabId !== id;
    }
    watch.setVisible(id === "watch");
    play.setVisible(id === "play"); // first activation starts the human's match
  };
  for (const { id } of TABS) {
    byId(`tab-${id}`).addEventListener("click", () => selectTab(id));
  }
  selectTab("train");

  // --- worker telemetry → header chip + smoke hook ---
  worker.onMetrics((m) => {
    window.__engineReady = true;
    header.setBackend(m.backend);
    // Every push re-derives the chip from the source of truth, so the state
    // heals itself after Stop/Start without main tracking transitions.
    header.setState(train.isRunning ? "running" : "ready");
  });

  // --- language flip: persist → rebind static labels → every controller ---
  function flipLang(): void {
    lang = lang === "en" ? "zh" : "en";
    saveLang(lang);
    rebindI18n(lang);
    header.setLang(lang);
    train.setLang(lang);
    watch.setLang(lang);
    play.setLang(lang);
  }

  // AudioContext construction is gesture-gated (autoplay policy): build the
  // graph on the FIRST pointer/keypress wherever it lands. Once-listeners —
  // unlock() is internally idempotent, but the listeners should not outlive
  // their purpose.
  const unlock = (): void => {
    void audio.unlock();
  };
  window.addEventListener("pointerdown", unlock, { once: true });
  window.addEventListener("keydown", unlock, { once: true });

  // Auto-start (SPEC §9) — the fleet trains while the user plays or watches.
  train.start();
  header.setState("running");

  window.__appReady = true; // smoke hook: shell + every controller wired
}

main();
