// by-id.ts — tiny typed DOM utilities used throughout the UI layer.
//
// The app is framework-free by design (house style): main.ts builds the static
// shell once, controllers query their subtree by id and re-render dynamic
// parts imperatively. These three helpers are all the "framework" there is.

/** Typed getElementById — throws loudly on a missing id (a wiring bug). */
export function byId<T extends Element = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as unknown as T;
}

/** Resolved element type for a tag ("button" → HTMLButtonElement, "svg" → SVGSVGElement). */
type ElFor<K> = K extends keyof HTMLElementTagNameMap
  ? HTMLElementTagNameMap[K]
  : K extends keyof SVGElementTagNameMap
    ? SVGElementTagNameMap[K]
    : never;

const SVG_TAGS = new Set(["svg", "line", "polyline", "circle", "rect", "path", "g", "text"]);

/**
 * Create an element with attributes/children in one call, covering HTML and
 * SVG tag names (charts are inline <svg>). All attributes go through
 * setAttribute (works for both families); `text` sets textContent instead.
 * `false`/`undefined` values skip the attribute entirely.
 */
export function el<K extends keyof HTMLElementTagNameMap | keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number | boolean> = {},
  ...children: Node[]
): ElFor<K> {
  const node: HTMLElement | SVGElement = SVG_TAGS.has(tag as string)
    ? document.createElementNS("http://www.w3.org/2000/svg", tag as string)
    : document.createElement(tag as string);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === undefined) continue;
    if (k === "text") node.textContent = String(v);
    else node.setAttribute(k, String(v));
  }
  for (const c of children) node.appendChild(c);
  return node as ElFor<K>;
}

/** Shorthand for a text node. */
export function txt(s: string): Text {
  return document.createTextNode(s);
}
