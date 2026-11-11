import { defineConfig } from "vite";
import { resolve } from "node:path";

// Relative base so the built site works from any folder on any static host.
// Two pages: index.html (flow) and fea.html (structure).
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 900,
    rollupOptions: { input: { index: resolve(__dirname, "index.html"), fea: resolve(__dirname, "fea.html") } },
  },
  worker: { format: "es" },
});
