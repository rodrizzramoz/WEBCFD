import { defineConfig } from "vite";
import { resolve } from "node:path";

// Builds the standalone FEA GPU check page: npx vite build -c dev/vite.feagpu.config.js
export default defineConfig({
  base: "./",
  build: { outDir: "/home/claude/g-dist", emptyOutDir: true, target: "es2022", rollupOptions: { input: resolve(__dirname, "fea-gpu-check.html") } },
  worker: { format: "es" },
});
