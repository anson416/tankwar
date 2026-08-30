import { defineConfig, devices } from "@playwright/test";

// Decoupled from the vitest config so node tests never pull in a browser
// env. Boots the BUILT dist/ via `vite preview` — which honors the relative
// base "./" from vite.config.ts — reproducing GitHub Pages URL resolution.
const isCI = !!process.env.CI;

export default defineConfig({
  testDir: "./test/smoke",
  workers: 1,
  fullyParallel: false,
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:4173",
    headless: true,
  },
  webServer: {
    command: "npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !isCI,
    cwd: new URL(".", import.meta.url).pathname,
    timeout: 120_000,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
