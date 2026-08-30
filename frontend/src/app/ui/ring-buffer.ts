// ring-buffer.ts — fixed-capacity FIFO used by the chart tiles: pushing past
// capacity drops the oldest sample, so memory stays bounded no matter how long
// a training session runs. `values` exposes the live backing array (read-only
// view) so drawChart can iterate it without copying on every frame.
export class RingBuffer<T> {
  private data: T[] = [];
  constructor(public capacity: number) {}
  push(v: T): void {
    this.data.push(v);
    if (this.data.length > this.capacity) this.data.shift();
  }
  get values(): readonly T[] {
    return this.data;
  }
  clear(): void {
    this.data = [];
  }
}
