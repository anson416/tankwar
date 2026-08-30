// chart-svg.ts — hand-rolled SVG line chart, no chart library. Redraws a
// single <polyline> from a RingBuffer with a fixed Y scale on demand.
// Clutter-free: baseline + polyline (+ optional reference line, e.g. the
// 1/N chance level under a win-rate trace), no in-chart text.
import type { RingBuffer } from "./ring-buffer";

interface ChartOpts {
  minY: number;
  maxY: number; // fixed max; pass dynamic value to rescale
  /** Optional horizontal reference line (e.g. 0.25 chance level). */
  refLine?: number;
  /** CSS class for the reference line (default "chart-refline"). */
  refClass?: string;
}

/** Redraw an SVG element from a numeric ring buffer. */
export function drawChart(
  svg: SVGSVGElement,
  buffer: RingBuffer<number>,
  opts: ChartOpts,
): void {
  const W = 200;
  const H = 60;
  const vals = buffer.values;
  // Clear previous content (we rebuild baseline + optional refline + line).
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const ns = "http://www.w3.org/2000/svg";

  // baseline at the bottom
  const baseline = document.createElementNS(ns, "line");
  baseline.setAttribute("x1", "0");
  baseline.setAttribute("y1", String(H - 0.5));
  baseline.setAttribute("x2", String(W));
  baseline.setAttribute("y2", String(H - 0.5));
  baseline.setAttribute("class", "chart-baseline");
  svg.appendChild(baseline);

  const { minY, maxY } = opts;
  const range = Math.max(1e-9, maxY - minY);

  if (opts.refLine !== undefined) {
    const y = H - ((opts.refLine - minY) / range) * H;
    if (y >= 0 && y <= H) {
      const ref = document.createElementNS(ns, "line");
      ref.setAttribute("x1", "0");
      ref.setAttribute("y1", y.toFixed(2));
      ref.setAttribute("x2", String(W));
      ref.setAttribute("y2", y.toFixed(2));
      ref.setAttribute("class", opts.refClass ?? "chart-refline");
      svg.appendChild(ref);
    }
  }

  if (vals.length < 2) return;

  const n = vals.length;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * W;
    const y = H - ((vals[i]! - minY) / range) * H;
    pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("class", "chart-line");
  svg.appendChild(line);
}

/** Numerically stable softmax (temperature sampling in the worker mirrors this). */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return sum > 0 ? exps.map((e) => e / sum) : exps.map(() => 0);
}
