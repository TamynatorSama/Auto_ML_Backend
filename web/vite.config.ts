import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api goes to the API, so the browser sees one origin (and cookies just work)
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: { "/api": "http://127.0.0.1:8000" } },
});
