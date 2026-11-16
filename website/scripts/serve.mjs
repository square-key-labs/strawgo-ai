import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const outDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'out');
const base = '/strawgo-ai';
const port = Number(process.env.PORT ?? 4173);

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

function send(res, status, file) {
  res.writeHead(status, { 'content-type': types[path.extname(file)] ?? 'application/octet-stream' });
  createReadStream(file).pipe(res);
}

createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (url.pathname === '/' || url.pathname === base) {
    res.writeHead(302, { location: `${base}/` });
    res.end();
    return;
  }
  if (!url.pathname.startsWith(`${base}/`)) {
    send(res, 404, path.join(outDir, '404.html'));
    return;
  }
  const rel = decodeURIComponent(url.pathname.slice(base.length + 1));
  let file = path.join(outDir, rel);
  if (!file.startsWith(outDir)) {
    res.writeHead(400);
    res.end();
    return;
  }
  if (existsSync(file) && statSync(file).isDirectory()) file = path.join(file, 'index.html');
  if (!existsSync(file)) {
    send(res, 404, path.join(outDir, '404.html'));
    return;
  }
  send(res, 200, file);
}).listen(port, () => {
  console.log(`serving ${outDir} at http://localhost:${port}${base}/`);
});
