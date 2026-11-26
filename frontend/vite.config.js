import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` serves the UI on 127.0.0.1:5173 and forwards /api to the backend (`dt serve --port 8765`).
// Port 8000 is not used here because another local project (POC-MESH) listens on it. Override with DT_API.
const API = process.env.DT_API || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": API },
  },
});
