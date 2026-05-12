import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_URL || "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      proxy: {
        // Proxy /api/* → backend (used when VITE_API_URL is not set in client code)
        "/api": {
          target: apiTarget,
          rewrite: (p) => p.replace(/^\/api/, ""),
          changeOrigin: true,
        },
      },
    },
  };
});
