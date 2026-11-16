import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(websiteDir, 'out');
const base = '/strawgo-ai';
const external = /^(https?:|mailto:|tel:|data:|javascript:)/i;

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return entry.name.endsWith('.html') ? [full] : [];
  });
}

function decode(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}

function markup(file) {
  return readFileSync(file, 'utf8')
    .replace(/<link\b[^>]*\brel="(?:preconnect|dns-prefetch)"[^>]*>/gi, '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, (tag) => tag.replace(/>[\s\S]*<\//, '></'));
}

if (!existsSync(outDir)) {
  console.error('out/ does not exist; run next build first');
  process.exit(1);
}

const pages = walk(outDir);
const html = new Map(pages.map((file) => [file, markup(file)]));
const ids = new Map();

function idsOf(file) {
  if (!ids.has(file)) {
    const found = new Set();
    for (const match of html.get(file).matchAll(/\sid="([^"]+)"/g)) found.add(decode(match[1]));
    ids.set(file, found);
  }
  return ids.get(file);
}

function resolveTarget(urlPath) {
  const rel = decodeURIComponent(urlPath.slice(base.length)).replace(/^\/+/, '');
  const candidate = path.join(outDir, rel);
  if (!candidate.startsWith(outDir)) return null;
  if (existsSync(candidate) && statSync(candidate).isFile()) return candidate;
  const index = path.join(candidate, 'index.html');
  if (existsSync(index)) return index;
  if (existsSync(`${candidate}.html`)) return `${candidate}.html`;
  return null;
}

const problems = [];
let checked = 0;

for (const [file, content] of html) {
  const where = path.relative(outDir, file);
  for (const match of content.matchAll(/\s(href|src)="([^"]*)"/g)) {
    const attr = match[1];
    const value = decode(match[2]);
    if (!value || external.test(value)) continue;
    checked += 1;
    const hashAt = value.indexOf('#');
    const target = hashAt === -1 ? value : value.slice(0, hashAt);
    const fragment = hashAt === -1 ? '' : decodeURIComponent(value.slice(hashAt + 1));
    if (!target) {
      if (fragment && !idsOf(file).has(fragment)) problems.push(`${where}: ${attr}="${value}" has no matching id on the page`);
      continue;
    }
    if (target !== base && !target.startsWith(`${base}/`)) {
      problems.push(`${where}: ${attr}="${value}" does not resolve under ${base}/`);
      continue;
    }
    const resolved = resolveTarget(target.split('?')[0]);
    if (!resolved) {
      problems.push(`${where}: ${attr}="${value}" points to a missing file`);
      continue;
    }
    if (fragment && resolved.endsWith('.html') && !idsOf(resolved).has(fragment)) {
      problems.push(`${where}: ${attr}="${value}" points to a missing anchor`);
    }
  }
}

if (problems.length) {
  console.error(`link check failed: ${problems.length} broken link(s)`);
  for (const problem of problems) console.error(`  ${problem}`);
  process.exit(1);
}

console.log(`link check passed: ${checked} internal links and assets in ${pages.length} HTML files`);
