// theme.ts — light/dark via data-theme on <html>, defaulting to the system
// preference on first load; the choice persists.
export type Theme = "dark" | "light";

const STORAGE_KEY = "tankwar.theme";

export function currentTheme(): Theme {
  return (document.documentElement.getAttribute("data-theme") as Theme) || "dark";
}

export function setTheme(t: Theme): void {
  document.documentElement.setAttribute("data-theme", t);
  try {
    localStorage.setItem(STORAGE_KEY, t);
  } catch {
    // Best-effort persistence only.
  }
}

/** Apply the persisted theme, or the system preference when none is stored. */
export function initTheme(): Theme {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    // localStorage unavailable — fall through to system preference.
  }
  const sys: Theme =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  const t: Theme = stored === "light" || stored === "dark" ? stored : sys;
  document.documentElement.setAttribute("data-theme", t);
  return t;
}

export function toggleTheme(): void {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
}
