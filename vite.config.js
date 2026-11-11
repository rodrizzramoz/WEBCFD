import { defineConfig } from "vite";

// Relative base so the built site works from any folder on any static host.
export default defineConfig({
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
});
