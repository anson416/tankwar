// boot.spec.ts — deploy gate against the BUILT dist (playwright.config boots
// `vite preview` on :4173). Four things:
//   1. the page boots: header title + the three mode tabs (train default);
//   2. the engine comes alive — `window.__engineReady` flips on the first
//      metrics push from the train worker (30 s: tfjs backend init plus the
//      auto-start round trip can be slow on CI's software GL);
//   3. every tab switches and renders its landmark (data-testid hooks);
//   4. zero console errors / page errors in every test — a worker boot failure,
//      a missing asset or an unhandled rejection all surface here.
import { test, expect, type Page } from "@playwright/test";

/** Collect console/page errors; each test asserts this is empty at the end. */
function watchConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });
  return errors;
}

test("boots: header, three tabs, engine ready within 30s", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("./");

  await expect(page.getByTestId("app-title")).toBeVisible();
  for (const tab of ["tab-train", "tab-play", "tab-watch"]) {
    await expect(page.getByTestId(tab)).toBeVisible();
  }

  await page.waitForFunction<boolean>(
    () =>
      (window as unknown as { __engineReady?: boolean }).__engineReady === true,
    undefined,
    { timeout: 30_000 },
  );
  expect(errors).toEqual([]);
});

test("every tab switches and renders its landmark", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("./");

  // Train is the default tab: the start control and the telemetry tiles are
  // the mode's irreducible landmarks.
  await expect(page.getByTestId("train-start")).toBeVisible();
  await expect(page.getByTestId("stat-episodes")).toBeVisible();

  // Play: the human seat's arena canvas.
  await page.getByTestId("tab-play").click();
  await expect(page.getByTestId("play-canvas")).toBeVisible();

  // Watch: the spectator arena canvas.
  await page.getByTestId("tab-watch").click();
  await expect(page.getByTestId("watch-canvas")).toBeVisible();

  expect(errors).toEqual([]);
});
