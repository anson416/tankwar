// play.spec.ts — the human seat: the match auto-starts on the tab's first
// activation, the canvas and HUD are live, a held W key drives through the
// SAME 18-action table the AI decodes (TankInput → actionFromKeys), and the
// assist/difficulty controls re-arm the lane. Independent of boot/watch —
// each spec opens its own page.
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

test("play: canvas + HUD, keyboard drives, assist toggles, difficulty retunes", async ({
  page,
}) => {
  const errors = watchConsole(page);
  await page.goto("./");
  // Boot sync (same hook boot.spec polls): the match round-trips through the
  // worker, which is still compiling its backend for the first seconds — a
  // keypress sampled before __engineReady measures the boot, not the input layer.
  await page.waitForFunction<boolean>(
    () => (window as unknown as { __engineReady?: boolean }).__engineReady === true,
    undefined,
    { timeout: 30_000 },
  );
  await page.getByTestId("tab-play").click();

  // Activation auto-starts the match; the Rematch button is the manual path
  // and is only needed if the auto-start surfaced an error banner.
  const rematch = page.getByTestId("play-rematch");
  await expect(rematch).toBeVisible();
  const banner = page.getByTestId("play-banner");
  if (await banner.isVisible()) {
    await rematch.click();
    await expect(banner).toBeHidden({ timeout: 30_000 });
  }

  // The arena canvas is on-screen and sized (world 900×700, letterboxed).
  const box = await page.getByTestId("play-canvas").boundingBox();
  if (!box) throw new Error("play canvas has no bounding box");
  expect(box.width).toBeGreaterThan(0);
  expect(box.height).toBeGreaterThan(0);
  await expect(page.getByTestId("play-hud")).toBeVisible();

  // Hold W for ~1 s: the key must decode through the shared table (W →
  // throttle 1 = forward) AND the lane must keep ticking while it is held.
  await page.keyboard.down("w");
  const held = await page.evaluate(() =>
    (window as unknown as { __inputState?: { current(): { throttle: number } } })
      .__inputState?.current(),
  );
  expect(held?.throttle).toBe(1);
  const before = await tickOf(page);
  await page.waitForTimeout(1_000);
  const after = await tickOf(page);
  await page.keyboard.up("w");
  expect(after).toBeGreaterThan(before);

  // Aim assist ships ON; the checkbox mirrors into the input layer and back.
  const assist = page.getByTestId("play-assist");
  await expect(assist).toBeChecked();
  await assist.click();
  await expect(assist).not.toBeChecked();

  // Difficulty slider → the τ readout: s=1 ⇒ τ=0.05 (near-greedy), s=0 ⇒
  // τ=1.05 (near-uniform), the same curve the protocol documents.
  await page.getByTestId("play-difficulty").evaluate((node) => {
    const input = node as HTMLInputElement;
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.getByTestId("play-diff-out")).toHaveText("τ=1.05");

  expect(errors).toEqual([]);
});
