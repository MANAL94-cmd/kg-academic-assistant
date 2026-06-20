import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the Vite dev server (http://localhost:5173) proxies all
// /api/* requests to the Flask backend on :5000, so the frontend and backend
// behave as one origin without CORS headaches. `npm run build` emits the
// production bundle into dist/, which the Flask app serves directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
