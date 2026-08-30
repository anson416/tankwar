// input.ts — the human seat: window keyboard listeners → the SAME 18-action
// table the AI decodes through (actionFromKeys → decode), so a keypress and an
// argmax index are indistinguishable to the engine. The class only translates
// key state into {throttle, steer, fire, seq} payloads — the play controller
// forwards them as {input} requests, where `seq` lets the worker drop packets
// that overtook each other in the postMessage queue.
//
// Push model, no polling: onAction fires ONLY on a change (the spec forbids an
// interval — at 30 Hz the worker re-reads the last input every tick anyway, so
// re-pushing would just spam the queue). Held keys live in a Set with the
// original game's sticky-dir semantics: releasing W while S is still down
// falls back to reverse instead of to neutral, because direction derives from
// what REMAINS held, not from the key that just left.
//
// Aim assist is UI checkbox state parked here so it survives the play tab's
// re-renders; the controller reads it back when (re)starting a match.

import { actionFromKeys, decode } from "../../game/actions";

/** One tick of control state — the payload of the protocol's InputReq. */
export interface InputState {
  throttle: number;
  steer: number;
  fire: boolean;
  seq: number;
}

/** Drive axes. W/S (throttle) and A/D (steer) plus their arrow twins; both of
 *  a pair held at once resolves deterministically in actionFromKeys (forward
 *  beats reverse, left beats right). */
type DirKey = "up" | "down" | "left" | "right";

/** Browser KeyboardEvent.code → drive axis. `.code` (physical key), not
 *  `.key`, so the layout-independent WASD cluster works on AZERTY too. */
const DIR_KEYS: Readonly<Record<string, DirKey | undefined>> = {
  KeyW: "up",
  ArrowUp: "up",
  KeyS: "down",
  ArrowDown: "down",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
};

declare global {
  interface Window {
    /** ?test=1 smoke hook: the live TankInput, for key-synthesis assertions. */
    __inputState?: TankInput;
  }
}

/**
 * True when this key event's target is an interactive control (or editable
 * content) that owns Space/arrow semantics of its own. `e.target` may be the
 * document on unfocused keydowns, which is exactly the "driving" case — hence
 * the closest()/isContentEditable guards rather than a blind cast.
 */
function onInteractiveControl(e: KeyboardEvent): boolean {
  const node = e.target as HTMLElement | null;
  if (node === null || typeof node !== "object" || typeof node.closest !== "function") return false;
  return (
    node.closest("button, input, select, textarea, a[href]") !== null || node.isContentEditable
  );
}

export class TankInput {
  private readonly held = new Set<DirKey>();
  private fireHeld = false;
  /** Monotonic counter — every CHANGE produces seq+1; the worker drops stale. */
  private seq = 0;
  private active = false;
  private aimAssistOn = false;

  // Stable arrow fields so add/removeEventListener always pair up.
  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (e.metaKey || e.ctrlKey || e.altKey) return; // browser shortcuts win
    if (e.code === "Space" || e.code.startsWith("Arrow")) {
      // Swallow ONLY keys that belong to driving — and only outside the tab's
      // own interactive controls: with the listeners attached document-wide,
      // an unconditional preventDefault made Space unable to toggle a focused
      // checkbox or activate the focused Rematch button, and the selects
      // un-arrowable, locking keyboard-only users out of the Play tab. The
      // page still never scrolls behind a driving key (synthetic keydowns in
      // the smoke spec target <body>, which is not a control).
      if (!onInteractiveControl(e)) e.preventDefault();
    }
    if (e.repeat) return; // auto-repeat cannot change the held set
    if (e.code === "Space") {
      if (!this.fireHeld) {
        this.fireHeld = true;
        this.push();
      }
      return;
    }
    const dir = DIR_KEYS[e.code];
    if (dir && !this.held.has(dir)) {
      this.held.add(dir);
      this.push();
    }
  };

  private readonly onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === "Space") {
      if (this.fireHeld) {
        this.fireHeld = false;
        this.push();
      }
      return;
    }
    const dir = DIR_KEYS[e.code];
    if (dir && this.held.has(dir)) {
      this.held.delete(dir);
      this.push();
    }
  };

  /** Focus lost mid-hold (alt-tab, devtools) would otherwise leave the tank
   *  driving forever: drop every key and send one neutral tick. */
  private readonly onBlur = (): void => {
    this.releaseAll();
  };

  constructor(private readonly onAction: (action: InputState) => void) {
    window.__inputState = this; // ?test=1 inspection hook
  }

  /** Attach the window listeners. Idempotent. */
  enable(): void {
    if (this.active) return;
    this.active = true;
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.onBlur);
  }

  /** Detach the listeners and release any held keys (a neutral tick goes out
   *  so a stopped lane cannot inherit a stuck throttle on rematch). */
  disable(): void {
    if (!this.active) return;
    this.active = false;
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onBlur);
    this.releaseAll();
  }

  /** Aim-assist checkbox state (≤6° intercept snap) — read by the play
   *  controller when building its {play} request. */
  get aimAssist(): boolean {
    return this.aimAssistOn;
  }
  setAimAssist(on: boolean): void {
    this.aimAssistOn = on;
  }

  /** Current control state WITHOUT bumping seq — ?test=1 inspection only;
   *  real input always flows through onAction. */
  current(): InputState {
    return { ...this.sample(), seq: this.seq };
  }

  /** Decode the held-key set through the shared table. decode(actionFromKeys)
   *  is deliberately roundabout: the keymap's product IS an action index, and
   *  routing through it guarantees human and AI inputs stay the same space. */
  private sample(): { throttle: number; steer: number; fire: boolean } {
    const a = decode(
      actionFromKeys({
        up: this.held.has("up"),
        down: this.held.has("down"),
        left: this.held.has("left"),
        right: this.held.has("right"),
        fire: this.fireHeld,
      }),
    );
    return { throttle: a.throttle, steer: a.steer, fire: a.fire };
  }

  /** Fire-and-forget: fire:true is sent the whole time Space is held — the
   *  engine's FIRE_COOLDOWN dedupes it into one shot per cooldown window. */
  private push(): void {
    this.seq += 1;
    this.onAction({ ...this.sample(), seq: this.seq });
  }

  private releaseAll(): void {
    if (this.held.size === 0 && !this.fireHeld) return;
    this.held.clear();
    this.fireHeld = false;
    this.push();
  }
}
