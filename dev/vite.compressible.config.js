import { defineConfig } from "vite";
import { resolve } from "node:path";

// Builds the solver-only check page for the compressible solver:
//   npx vite build -c dev/vite.compressible.config.js
export default defineConfig({
  base: "./",
  build: { outDir: "/home/claude/c-dist", emptyOutDir: true, target: "es2022", rollupOptions: { input: resolve(__dirname, "compressible-check.html") } },
});
