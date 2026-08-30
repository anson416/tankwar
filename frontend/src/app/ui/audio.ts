// audio.ts — the WebAudio juice layer: music bed, per-tank engine loops,
// one-shot SFX and two synthesized cues (pickup blip, phase sweep). Built to
// the plan's audio table (SPEC §Renderer/audio): engine playbackRate tracks
// |v|, cannon fire is detuned, ricochet is a pitched-up cannon through a
// lowpass, and the whole graph is gated by one master gain.
//
// Autoplay policy: browsers refuse audio before a user gesture, so unlock() is
// the ONLY path that constructs the AudioContext (and it must be called from a
// click/keydown handler). Until then — and if the context or an asset fails —
// every method is a silent no-op: audio is juice, never a boot blocker.
//
// Node budget: one-shots create their nodes per call and let the GC reclaim
// them (at ≤6 shots/s/tank a pool would outlive its usefulness); the engine
// loops and the music bed are the only persistent nodes, tracked by tank id so
// deaths stop exactly the loop that tank owned.

import { MAX_SPEED_FWD } from "../../game/constants";
import type { EventKind } from "../../game/types";
import type { Snapshot } from "../../train-protocol";

/** Looping gain for the music bed — quiet on purpose (plan: 0.15). */
const MUSIC_GAIN = 0.15;
/** Explosion/kill SFX gain (plan: 0.8) — the loudest thing on the field. */
const EXPLOSION_GAIN = 0.8;
/** Cannon fire sits under the explosion so a volley never masks a kill. */
const FIRE_GAIN = 0.5;
/** Ricochet ping — a bright accent, quieter than the shot it bounced. */
const RICOCHET_GAIN = 0.35;
/** Lowpass corner for the ricochet ping: the 2.2× rate already shifts the
 *  sample's spectrum up; this keeps the metallic band, drops the boom. */
const RICOCHET_LP_HZ = 2400;

type SoundName = "music" | "engine" | "cannon" | "explosion";

/** Buffers fetched+decoded on first unlock, in parallel. Files live in
 *  public/assets/audios (ATTRIBUTION.md carries their licenses). */
const SOUND_FILES: readonly [SoundName, string][] = [
  ["music", "background_music.wav"],
  ["engine", "tank_engine.ogg"],
  ["cannon", "cannon_fire.ogg"],
  ["explosion", "explosion.wav"],
];

/** Resolve a public/ asset for BOTH dev ("/") and the Pages build ("./" base
 *  → the site lives under /tankwar/): document.baseURI carries the prefix the
 *  built index.html was served from, so a hard-coded path can 404 nowhere. */
function assetUrl(file: string): string {
  return new URL(`assets/audios/${file}`, document.baseURI).href;
}

/** One persistent engine loop: the sample source plus its mix gain. */
interface EngineNode {
  src: AudioBufferSourceNode;
  gain: GainNode;
}

export class GameAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private readonly buffers = new Map<SoundName, AudioBuffer>();
  /** Engine loops keyed by tank id — allocation is lazy, release on death. */
  private readonly engines = new Map<number, EngineNode>();
  private music: EngineNode | null = null;
  /** Guards unlock() against double init (two quick clicks must not spawn two
   *  contexts); nulled by dispose() so a disposed GameAudio can re-unlock. */
  private unlockPromise: Promise<void> | null = null;
  /** User preferences recorded even before the context exists — the master
   *  gain is initialized from them at unlock and updated live after. */
  private enabled = true;
  private muted = false;

  /**
   * Create/resume the AudioContext and start the music bed. MUST be called
   * from a user-gesture handler (autoplay policy) — typically the first click
   * anywhere. Idempotent; concurrent callers share one init.
   */
  async unlock(): Promise<void> {
    if (!this.unlockPromise) this.unlockPromise = this.init();
    return this.unlockPromise;
  }

  /** Master gain gate, driven by the header sound chip. Starts ON. */
  setEnabled(on: boolean): void {
    this.enabled = on;
    this.applyGate();
  }

  /** User mute preference, kept distinct from `enabled` (the ctx state) so the
   *  chip can dim the sound without tearing down the context. */
  setMuted(m: boolean): void {
    this.muted = m;
    this.applyGate();
  }

  /**
   * Reconcile the engine loops with the snapshot: one loop per ALIVE tank,
   * allocated on first sight, reused by id, stopped when the tank dies (or
   * simply leaves the snapshot — watch-mode callers may stream fewer tanks).
   * playbackRate = 0.7 + 1.1·(|v|/MAX_SPEED_FWD) per the plan, so idle tanks
   * rumble low and flat-out tanks whine. Mix is distance-free: the k-th alive
   * tank (0-based) plays at 1/(k+1) — a cheap headroom-safe sum over ≤4 loops.
   */
  syncEngines(snapshot: Snapshot): void {
    const ctx = this.ctx;
    if (!ctx || !this.master) return;
    const now = ctx.currentTime;
    const alive = new Set<number>();
    let idx = 0;
    for (const t of snapshot.tanks) {
      if (!t.alive) continue;
      alive.add(t.i);
      const speed = Math.min(Math.abs(t.v), MAX_SPEED_FWD);
      const rate = 0.7 + 1.1 * (speed / MAX_SPEED_FWD);
      const gain = 1 / (idx + 1);
      idx++;
      let node = this.engines.get(t.i);
      if (!node) {
        const buf = this.buffers.get("engine");
        if (!buf) continue; // asset still decoding / failed — stay silent
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.loop = true;
        src.playbackRate.value = rate;
        const gainNode = ctx.createGain();
        gainNode.gain.value = gain;
        src.connect(gainNode);
        gainNode.connect(this.master);
        src.start(now);
        node = { src, gain: gainNode };
        this.engines.set(t.i, node);
      } else {
        // setTargetAtTime, not .value=, so per-tick updates glide instead of
        // zipper-cracking when a tank changes speed every 33 ms.
        node.src.playbackRate.setTargetAtTime(rate, now, 0.05);
        node.gain.gain.setTargetAtTime(gain, now, 0.05);
      }
    }
    for (const [id, node] of this.engines) {
      if (!alive.has(id)) this.stopEngine(id, node);
    }
  }

  /**
   * One gameplay cue, by the event kind the engine emitted (see the plan's
   * audio table). Unmapped kinds ("hit"/"crate"/"spawn") are visual-only and
   * fall through silently — the effects layer owns them.
   */
  event(kind: EventKind, opts?: { rate?: number }): void {
    const ctx = this.ctx;
    if (!ctx || !this.master) return;
    switch (kind) {
      case "fire":
        // Detune 0.95–1.05 keeps volleys from phasing into one sound.
        // Math.random is fine here: audio flavour is off the deterministic
        // path — the engine never reads it, no replay ever diffs it.
        this.play("cannon", opts?.rate ?? 0.95 + Math.random() * 0.1, FIRE_GAIN);
        break;
      case "kill":
      case "explode":
        this.play("explosion", opts?.rate ?? 1, EXPLOSION_GAIN);
        break;
      case "ricochet":
        // Per-call BiquadFilter — a one-shot needs no persistent node.
        this.play("cannon", opts?.rate ?? 2.2, RICOCHET_GAIN, RICOCHET_LP_HZ);
        break;
      case "pickup":
        this.blip(ctx);
        break;
      case "phase":
        this.sweep(ctx);
        break;
      default:
        break; // hit/crate/spawn: visuals only
    }
  }

  /** Stop every node and close the context. Idempotent. */
  dispose(): void {
    for (const [id, node] of this.engines) this.stopEngine(id, node);
    if (this.music) {
      try {
        this.music.src.stop();
      } catch {
        // Already stopped — nothing to do.
      }
      this.music = null;
    }
    const ctx = this.ctx;
    this.ctx = null;
    this.master = null;
    this.buffers.clear();
    this.unlockPromise = null; // a disposed instance may be re-unlocked
    if (ctx) void ctx.close().catch(() => undefined);
  }

  // --- internals ---------------------------------------------------------

  /** The gesture-gated half of unlock(): build the graph, load the four
   *  buffers in parallel, then start the looping music bed. */
  private async init(): Promise<void> {
    const Ctor: typeof AudioContext | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return; // no WebAudio at all — every method stays a no-op
    const ctx = new Ctor();
    this.ctx = ctx;
    const master = ctx.createGain();
    master.gain.value = this.enabled && !this.muted ? 1 : 0;
    master.connect(ctx.destination);
    this.master = master;
    try {
      await ctx.resume(); // gesture-created contexts may still start suspended
    } catch {
      // Resume refused — sounds will start on the next gesture-driven call.
    }
    await Promise.all(
      SOUND_FILES.map(async ([name, file]) => {
        try {
          const res = await fetch(assetUrl(file));
          if (!res.ok) return;
          this.buffers.set(name, await ctx.decodeAudioData(await res.arrayBuffer()));
        } catch {
          // Missing/undecodable asset: that cue stays silent, the rest load.
        }
      }),
    );
    if (this.ctx !== ctx) return; // disposed mid-load — drop everything
    const music = this.buffers.get("music");
    if (music && this.master) {
      const src = ctx.createBufferSource();
      src.buffer = music;
      src.loop = true; // music starts with the session and never ends
      const gain = ctx.createGain();
      gain.gain.value = MUSIC_GAIN;
      src.connect(gain);
      gain.connect(this.master);
      src.start();
      this.music = { src, gain };
    }
  }

  /** Master gate = enabled && !muted; 10 ms ramp so toggling never clicks. */
  private applyGate(): void {
    const ctx = this.ctx;
    if (!ctx || !this.master) return;
    const on = this.enabled && !this.muted;
    this.master.gain.setTargetAtTime(on ? 1 : 0, ctx.currentTime, 0.01);
  }

  /** Fire one decoded sample: source → [lowpass] → gain → master, all
   *  per-call. A missing buffer (still decoding / failed fetch) is silent. */
  private play(name: SoundName, rate: number, gain: number, lowpassHz?: number): void {
    const ctx = this.ctx;
    const buf = this.buffers.get(name);
    if (!ctx || !this.master || !buf) return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = rate;
    const out = ctx.createGain();
    out.gain.value = gain;
    if (lowpassHz !== undefined) {
      const lp = ctx.createBiquadFilter();
      lp.type = "lowpass";
      lp.frequency.value = lowpassHz;
      src.connect(lp);
      lp.connect(out);
    } else {
      src.connect(out);
    }
    out.connect(this.master);
    src.start();
  }

  /** Pickup: synthesized two-note sine blip 880 → 1320 Hz, 120 ms total — no
   *  asset, so it exists even if the sample fetches all failed. */
  private blip(ctx: AudioContext): void {
    if (!this.master) return;
    const t0 = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, t0);
    osc.frequency.setValueAtTime(1320, t0 + 0.06);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(0.25, t0 + 0.01);
    gain.gain.setValueAtTime(0.25, t0 + 0.05);
    gain.gain.linearRampToValueAtTime(0, t0 + 0.12);
    osc.connect(gain);
    gain.connect(this.master);
    osc.start(t0);
    osc.stop(t0 + 0.12);
  }

  /** Phase transition: rising sweep — escalation announced, sudden death
   *  escalated by the same gesture (the event carries no phase payload). */
  private sweep(ctx: AudioContext): void {
    if (!this.master) return;
    const t0 = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(180, t0);
    osc.frequency.exponentialRampToValueAtTime(720, t0 + 0.5);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.22, t0 + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.5);
    osc.connect(gain);
    gain.connect(this.master);
    osc.start(t0);
    osc.stop(t0 + 0.5);
  }

  /** Stop + unlink one engine loop. Deleting during iteration of a Map is
   *  safe in JS (live iteration), so syncEngines may prune inline. */
  private stopEngine(id: number, node: EngineNode): void {
    try {
      node.src.stop();
    } catch {
      // Already stopped.
    }
    node.src.disconnect();
    node.gain.disconnect();
    this.engines.delete(id);
  }
}
