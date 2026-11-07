import { createReadStream, existsSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Repo root (web/..). The dev-only mock routes read the engine fixtures and the cached test audio;
// nothing outside web/ is written.
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const MOCK_FILES: Record<string, { file: string; type: string }> = {
  '/__mock/analysis.json': { file: `${ROOT}shared/fixtures/huijia.analysis.json`, type: 'application/json; charset=utf-8' },
  '/__mock/display.json': { file: `${ROOT}shared/fixtures/huijia.display.json`, type: 'application/json; charset=utf-8' },
  '/__mock/audio': { file: `${ROOT}cache/yt-Y7boUFbM76k/play.m4a`, type: 'audio/mp4' },
};

/** Streams a file with HTTP Range support (the <audio> element seeks with Range requests). */
function sendFile(req: IncomingMessage, res: ServerResponse, file: string, type: string): void {
  const size = statSync(file).size;
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', type);
  res.setHeader('Cache-Control', 'no-cache');
  const range = req.headers.range;
  const m = range ? /^bytes=(\d*)-(\d*)$/.exec(range) : null;
  if (!m) {
    res.setHeader('Content-Length', String(size));
    createReadStream(file).pipe(res);
    return;
  }
  let start = m[1] ? Number(m[1]) : 0;
  let end = m[2] ? Number(m[2]) : size - 1;
  if (!m[1] && m[2]) {
    start = Math.max(0, size - Number(m[2]));
    end = size - 1;
  }
  end = Math.min(end, size - 1);
  if (start > end || start >= size) {
    res.statusCode = 416;
    res.setHeader('Content-Range', `bytes */${size}`);
    res.end();
    return;
  }
  res.statusCode = 206;
  res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
  res.setHeader('Content-Length', String(end - start + 1));
  createReadStream(file, { start, end }).pipe(res);
}

/** `?mock=1` support in `npm run dev`: serves the 回家 fixtures and the cached audio under /__mock/. */
function mockFixtures(): Plugin {
  return {
    name: 'nathan-chordify-mock-fixtures',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = (req.url ?? '').split('?')[0];
        const entry = MOCK_FILES[path];
        if (!entry) return next();
        if (!existsSync(entry.file)) {
          res.statusCode = 404;
          res.end(`mock file not found: ${entry.file}`);
          return;
        }
        sendFile(req, res, entry.file, entry.type);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), mockFixtures()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', changeOrigin: false },
      '/health': { target: 'http://127.0.0.1:8765', changeOrigin: false },
    },
  },
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});
