// worker-client.ts — main-thread handle on the training worker. Spawns it,
// pairs requests to responses by id, and routes the four unsolicited id-0
// streams (metrics progress, arena layout, watch/play snapshots, episode
// results) to fan-out listeners.
//
// ALL neural-network work lives in the worker (training + every inference for
// play/watch) — the main bundle imports no @tensorflow/*.
import {
  isArena,
  isEpisodeEnd,
  isMetrics,
  isSnapshot,
  type ArenaMsg,
  type EpisodeEndMsg,
  type LaneKind,
  type Metrics,
  type NetConfig,
  type Snapshot,
  type TeamMode,
  type WorkerReq,
  type WorkerResp,
} from "../train-protocol";

export type MetricsListener = (m: Metrics) => void;
/** The static world pushed once per episode (the {arena} message). */
export type ArenaListener = (a: ArenaMsg) => void;
export type SnapshotListener = (s: Snapshot) => void;
export type EpisodeEndListener = (e: EpisodeEndMsg) => void;

/**
 * `Omit` that distributes over the union. Plain `Omit<WorkerReq, "id">`
 * collapses the union to its COMMON keys (just `id`), which would make
 * request()/control() accept any object; distributed, each member keeps its
 * payload field so call sites are checked against the real wire shapes.
 */
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export class WorkerBackendController {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (r: WorkerResp) => void; reject: (e: unknown) => void }
  >();
  private metricsListeners = new Set<MetricsListener>();
  // Lane-scoped fan-out: BOTH lanes push on id 0 (watch duel + human's match),
  // and every push is tagged at the source — a controller subscribed to one
  // lane must never ingest the other's world (two interleaved streams would
  // paint two different games into one canvas/HUD and fire cross-lane episode
  // banners mid-match).
  private arenaListeners = new Map<LaneKind, Set<ArenaListener>>();
  private snapshotListeners = new Map<LaneKind, Set<SnapshotListener>>();
  private episodeEndListeners = new Map<LaneKind, Set<EpisodeEndListener>>();

  constructor() {
    // The `new URL(...)` MUST sit lexically inside `new Worker(...)`: Vite's
    // worker plugin pattern-matches exactly that shape to emit a worker chunk
    // (hoisting the URL into a variable makes the bundler emit the raw .ts as
    // a plain asset instead — measured build failure). The cost of the rigid
    // form: the page's query string cannot be forwarded, so the worker's
    // ?test=1 smoke fixture — keyed on the WORKER's own location.search — is
    // currently unreachable from a page URL and only the default config boots.
    this.worker = new Worker(new URL("../workers/train.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (e: MessageEvent<WorkerResp>) => this.onMessage(e.data);
    this.worker.onerror = (ev: ErrorEvent) => {
      // A dead worker can never answer — fail every waiter now rather than
      // leaving promises pending forever.
      const err = new Error(ev.message || "train worker error");
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    };
  }

  private onMessage(data: WorkerResp): void {
    // Pair by id FIRST. A request's metrics ack carries the REQUEST id, so
    // testing isMetrics() before the pending lookup (the naive order) would
    // fan the ack out to live tiles and leave applyConfig() pending forever.
    const p = data.id !== 0 ? this.pending.get(data.id) : undefined;
    if (p) {
      this.pending.delete(data.id);
      p.resolve(data); // ErrorResp resolves too; callers check isError()
      return;
    }
    // Everything reaching here is an unsolicited push (id 0) — or an ErrorResp
    // the worker could not pair with a request (version skew), which has no
    // waiter to reject and is dropped. Lane-scoped pushes route ONLY to the
    // listeners registered for their lane.
    if (isMetrics(data)) {
      for (const fn of this.metricsListeners) fn(data.metrics);
      return;
    }
    if (isArena(data)) {
      for (const fn of this.arenaListeners.get(data.lane) ?? []) fn(data.arena);
      return;
    }
    if (isSnapshot(data)) {
      for (const fn of this.snapshotListeners.get(data.lane) ?? []) fn(data.snapshot);
      return;
    }
    if (isEpisodeEnd(data)) {
      for (const fn of this.episodeEndListeners.get(data.lane) ?? []) fn(data.episodeEnd);
      return;
    }
  }

  /** Subscribe to metric pushes (fleet-wide — metrics are not lane-scoped).
   *  Returns an unsubscribe function. */
  onMetrics(fn: MetricsListener): () => void {
    this.metricsListeners.add(fn);
    return () => this.metricsListeners.delete(fn);
  }
  /** Subscribe to `lane`'s arena layout pushes. */
  onArena(lane: LaneKind, fn: ArenaListener): () => void {
    this.listenerSet(this.arenaListeners, lane, fn);
    return () => this.arenaListeners.get(lane)?.delete(fn);
  }
  /** Subscribe to `lane`'s snapshot stream (coalesced, id 0). */
  onSnapshot(lane: LaneKind, fn: SnapshotListener): () => void {
    this.listenerSet(this.snapshotListeners, lane, fn);
    return () => this.snapshotListeners.get(lane)?.delete(fn);
  }
  /** Subscribe to `lane`'s episode terminations. */
  onEpisodeEnd(lane: LaneKind, fn: EpisodeEndListener): () => void {
    this.listenerSet(this.episodeEndListeners, lane, fn);
    return () => this.episodeEndListeners.get(lane)?.delete(fn);
  }

  /** First subscription for a lane creates its set (a Map keeps the three
   *  lane-scoped fan-outs symmetric without seeding both keys up front). */
  private listenerSet<S>(map: Map<LaneKind, Set<S>>, lane: LaneKind, fn: S): void {
    let set = map.get(lane);
    if (!set) {
      set = new Set<S>();
      map.set(lane, set);
    }
    set.add(fn);
  }

  /** Send a request expecting the worker to answer on the same id. */
  request<T extends WorkerResp>(req: DistOmit<WorkerReq, "id">): Promise<T> {
    const id = this.nextId++;
    const payload = { ...req, id } as WorkerReq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (r) => resolve(r as T),
        reject,
      });
      this.worker.postMessage(payload);
    });
  }

  /** Fire-and-forget control message — anything the UI must not wait on
   *  (stop/pause/step/watchStop and the per-tick input stream). */
  control(req: DistOmit<WorkerReq, "id">): void {
    const id = this.nextId++;
    this.worker.postMessage({ ...req, id } as WorkerReq);
  }

  /**
   * Reconfigure the worker's nets/scheduler. Resolves with the post-rebuild
   * metrics ack (a MetricsResp carrying the request id) or an ErrorResp when
   * the rebuild failed — the worker's rebuild is transactional, so a failure
   * means the previous run is still intact.
   */
  applyConfig(config: NetConfig): Promise<WorkerResp> {
    return this.request({ config });
  }

  /** Start (or restart) the AI spectator lane. Awaited so the Watch tab can
   *  surface a failed start (e.g. no model trained yet) instead of a black canvas. */
  watchStart(tankCount: number, speedHz: number): Promise<WorkerResp> {
    return this.request({ watch: { tankCount, speedHz } });
  }
  watchStop(): void {
    this.control({ watchStop: true });
  }
  /** Pause (true) / resume (false) a lane. `lane` addresses ONE lane — the
   *  pause button and the tab-hide paths must never freeze the OTHER lane's
   *  duel through a shared flag; omitted means "both" (the wire default). */
  pause(pause: boolean, lane?: LaneKind): void {
    this.control(lane === undefined ? { pause } : { pause, lane });
  }
  /** Advance the paused watch lane one tick. */
  step(): void {
    this.control({ step: true });
  }

  /** Start (or restart) the human seat. Awaited for the same reason as
   *  watchStart; `difficulty` ∈ [0,1] → opponent temperature. */
  playStart(opts: {
    tankCount: number;
    teamMode: TeamMode;
    difficulty: number;
    aimAssist: boolean;
  }): Promise<WorkerResp> {
    return this.request({ play: opts });
  }
  /** One tick of human control state — 30/s and never awaited; `seq` lets the
   *  worker discard packets that arrive out of order. */
  input(input: { throttle: number; steer: number; fire: boolean; seq: number }): void {
    this.control({ input });
  }

  /** Stop the worker and drop every waiter/listener (page teardown / HMR). */
  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
    this.metricsListeners.clear();
    this.arenaListeners.clear();
    this.snapshotListeners.clear();
    this.episodeEndListeners.clear();
  }
}
