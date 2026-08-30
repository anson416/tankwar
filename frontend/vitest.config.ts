import { defineConfig } from "vitest/config";

// Node env: the rules engine, encodings, and protocol are DOM-free pure TS —
// they must also run inside the Web Worker, so tests must NOT depend on a
// browser environment. tfjs (if imported by a test) falls back to its CPU
// backend under Node.
export default defineConfig({
  test: {
    environment: "node",
    include: ["test/unit/**/*.spec.ts"],
    globals: true,
    // Game-engine property tests play thousands of random games; keep the
    // default timeout generous but bounded.
    testTimeout: 30_000,
  },
});
