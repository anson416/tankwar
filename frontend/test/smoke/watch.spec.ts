// watch.spec.ts — the spectator lane: ticks flow, the scoreboard mirrors the
// lane's cast, Pause freezes the stream in place, and the speed slider re-arms
// the lane at the new rate. Independent of boot/play (each spec opens its own
// page; the playwright config runs a single worker anyway).
//
// `__tickCount` is the shared smoke hook the watch/play controllers stamp on
// every snapshot — this spec only ever shows the watch tab, so every tick it
// reads is the watch lane's.
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

const tickOf = (page: Page): Promise<number> =>
  page.evaluate(() => (window as unknown as { __tickCount?: number }).__tickCount ?? -1);

async function waitForTicks(page: Page, timeout: number): Promise<void> {
  await page.waitForFunction(
    () => ((window as unknown as { __tickCount?: number }).__tickCount ?? 0) > 0,
    undefined,
    { timeout },
  );
}

/** Drive a range input programmatically: Playwright has no drag primitive for
 *  sliders, and the controller listens for `input` — the same event a real
 *  drag bubbles. */
const setRange = (page: Page, testid: string, value: string): Promise<void> =>
  page.getByTestId(testid).evaluate((node, v) => {
    const input = node as HTMLInputElement;
    input.value = String(v);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);

test("watch lane ticks, scores its cast, freezes on pause, re-arms on speed", async ({
  page,
}) => {
  const errors = watchConsole(page);
  await page.goto("./");
  await page.getByTestId("tab-watch").click();

  // The lane starts on controller init — it has been running behind the train
  // tab since boot, so ticks must already be flowing.
  await waitForTicks(page, 30_000);

  // Scoreboard: the cast mirrors the toolbar's tank count. Narrow the default
  // 4-tank melee to the 2-tank duel and expect exactly one row per tank.
  await page.getByTestId("watch-tanks").selectOption("2");
  await expect(page.getByTestId("watch-tank-0")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("watch-tank-1")).toBeVisible();
  await expect(page.getByTestId("watch-score").locator(".score-row")).toHaveCount(2);

  // Running: the 30 Hz lane (coalesced snapshots) advances the counter.
  const running = await tickOf(page);
  await page.waitForTimeout(500);
  expect(await tickOf(page)).toBeGreaterThan(running);

  // Pause: allow one in-flight tick to land, then the counter must hold still
  // across 700 ms — a frozen lane streams no snapshots at all.
  await page.getByTestId("watch-pause").click();
  await page.waitForTimeout(300);
  const frozen = await tickOf(page);
  await page.waitForTimeout(700);
  expect(await tickOf(page)).toBe(frozen);

  // Resume, then retune speed: the ×N readout tracks the input immediately
  // (the lane restart itself is debounced 250 ms in the controller).
  await page.getByTestId("watch-pause").click();
  await setRange(page, "watch-speed", "2");
  await expect(page.getByTestId("watch-speed-out")).toHaveText("×2");

  // The ×2 lane restarts from a fresh episode (tick counter resets), so the
  // assertion is "ticks flow again", not "the level rose past the old one".
  await waitForTicks(page, 15_000);
  const rearmed = await tickOf(page);
  await page.waitForTimeout(500);
  expect(await tickOf(page)).toBeGreaterThan(rearmed);

  expect(errors).toEqual([]);
});
