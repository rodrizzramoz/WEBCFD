import { defineConfig } from "vite";
import { resolve } from "node:path";

// Builds the solver-only benchmark page: npx vite build -c dev/vite.bench.config.js
export default defineConfig({
  base: "./",
  build: { outDir: "dist-bench", emptyOutDir: true, target: "es2022", rollupOptions: { input: resolve(__dirname, "bench.html") } },
});
