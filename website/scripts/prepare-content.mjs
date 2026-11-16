import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(websiteDir, '..');
const docsDir = path.join(repoRoot, 'docs');
const outDir = path.join(websiteDir, 'content');
const repoUrl = 'https://github.com/square-key-labs/strawgo-ai';
const branch = 'main';
const excluded = new Set(['agents']);

const fence = /^[ \t]*(`{3,}|~{3,})/;
const inlineCode = /(`+)(?:(?!\1)[\s\S])+?\1/g;
const linkPattern = /(\]\()([^)\s]+)((?:\s+"[^"]*")?\))/g;
const scheme = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;

function walk(dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    const rel = path.relative(docsDir, full).split(path.sep).join('/');
    if (excluded.has(rel.split('/')[0])) continue;
    if (entry.isDirectory()) files.push(...walk(full));
    else if (entry.name.endsWith('.md')) files.push(rel);
  }
  return files.sort();
}

function targetPath(rel) {
  if (rel === 'README.md') return 'overview.md';
  if (rel.endsWith('/README.md')) return rel.slice(0, -'README.md'.length) + 'index.md';
  return rel;
}

function pageUrl(rel) {
  const target = targetPath(rel).replace(/\.md$/, '');
  const clean = target === 'index' ? '' : target.replace(/\/index$/, '');
  return clean ? `/${clean}` : '/';
}

function splitFenced(markdown) {
  const parts = [];
  let buffer = [];
  let open = null;
  for (const line of markdown.split(/(?<=\n)/)) {
    const match = fence.exec(line);
    if (open === null) {
      if (match) {
        parts.push({ code: false, text: buffer.join('') });
        buffer = [line];
        open = match[1];
      } else {
        buffer.push(line);
      }
    } else {
      buffer.push(line);
      if (match && match[1][0] === open[0] && match[1].length >= open.length && line.trim() === match[1]) {
        parts.push({ code: true, text: buffer.join('') });
        buffer = [];
        open = null;
      }
    }
  }
  parts.push({ code: open !== null, text: buffer.join('') });
  return parts;
}

function githubUrl(absPath, anchor) {
  const rel = path.relative(repoRoot, absPath).split(path.sep).join('/');
  const kind = statSync(absPath).isDirectory() ? 'tree' : 'blob';
  return `${repoUrl}/${kind}/${branch}/${rel}${anchor ? `#${anchor}` : ''}`;
}

function rewriteTarget(target, pageRel, published) {
  if (target.startsWith('#') || target.startsWith('/') || scheme.test(target)) return target;
  const hash = target.indexOf('#');
  const file = hash === -1 ? target : target.slice(0, hash);
  const anchor = hash === -1 ? '' : target.slice(hash + 1);
  if (!file) return target;
  const relInDocs = path.posix.normalize(path.posix.join(path.posix.dirname(pageRel), file));
  const abs = path.join(docsDir, relInDocs);
  if (!existsSync(abs)) return target;
  if (published.has(relInDocs)) return pageUrl(relInDocs) + (anchor ? `#${anchor}` : '');
  return githubUrl(abs, anchor);
}

function rewriteLinks(text, pageRel, published) {
  const replace = (chunk) =>
    chunk.replace(linkPattern, (_, open, target, close) => open + rewriteTarget(target, pageRel, published) + close);
  let out = '';
  let last = 0;
  for (const match of text.matchAll(inlineCode)) {
    out += replace(text.slice(last, match.index));
    out += match[0];
    last = match.index + match[0].length;
  }
  return out + replace(text.slice(last));
}

function plain(text) {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/(\*\*|__)(.+?)\1/g, '$2')
    .replace(/(\*|_)(.+?)\1/g, '$2')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractMeta(markdown, rel) {
  const lines = markdown.split('\n');
  const h1 = lines.findIndex((line) => /^# /.test(line));
  if (h1 === -1) throw new Error(`${rel}: no top-level heading`);
  const title = plain(lines[h1].slice(2));
  const body = [...lines.slice(0, h1), ...lines.slice(h1 + 1)].join('\n').replace(/^\s*\n/, '');
  const paragraph = [];
  for (const line of lines.slice(h1 + 1)) {
    if (!line.trim()) {
      if (paragraph.length) break;
      continue;
    }
    if (/^(#|>|\||```|~~~|- |\* |\d+\. |<)/.test(line.trim())) {
      if (paragraph.length) break;
      continue;
    }
    paragraph.push(line.trim());
  }
  return { title, description: plain(paragraph.join(' ')), body };
}

function parseNav(readme, titles) {
  const toc = readme.split(/^## Table of contents\s*$/m)[1]?.split(/^## /m)[0];
  if (!toc) throw new Error('docs/README.md: no "Table of contents" section');
  const sections = [];
  let current = null;
  for (const line of toc.split('\n')) {
    const heading = /^### (.+)$/.exec(line);
    if (heading) {
      current = { title: heading[1].trim(), items: [] };
      sections.push(current);
      continue;
    }
    const item = /^- \[([^\]]+)\]\(([^)]+)\)/.exec(line);
    if (!item || !current) continue;
    const [, label, href] = item;
    const relInDocs = path.posix.normalize(href);
    const abs = path.join(docsDir, relInDocs);
    if (titles.has(relInDocs)) {
      current.items.push({ type: 'page', title: label, url: pageUrl(relInDocs) });
    } else if (existsSync(abs) && statSync(abs).isDirectory() && !relInDocs.startsWith('../')) {
      const pages = [...titles.keys()].filter((rel) => rel.startsWith(`${relInDocs.replace(/\/$/, '')}/`)).sort();
      current.items.push({
        type: 'folder',
        title: label,
        items: pages.map((rel) => ({ type: 'page', title: titles.get(rel), url: pageUrl(rel) })),
      });
    } else if (existsSync(abs)) {
      current.items.push({ type: 'external', title: label, url: githubUrl(abs, '') });
    } else {
      throw new Error(`docs/README.md: table of contents links to missing ${href}`);
    }
  }
  const start = sections.find((section) => section.title === 'Getting started');
  start?.items.unshift({ type: 'page', title: 'Overview', url: pageUrl('README.md') });
  return sections;
}

function collectUrls(items, into = new Set()) {
  for (const item of items) {
    if (item.type === 'page') into.add(item.url);
    if (item.type === 'folder') collectUrls(item.items, into);
  }
  return into;
}

const files = walk(docsDir);
const published = new Set(files);
const titles = new Map();

rmSync(outDir, { recursive: true, force: true });

for (const rel of files) {
  const source = readFileSync(path.join(docsDir, rel), 'utf8');
  const { title, description, body } = extractMeta(source, rel);
  titles.set(rel, title);
  const rewritten = splitFenced(body)
    .map((part) => (part.code ? part.text : rewriteLinks(part.text, rel, published)))
    .join('');
  const frontmatter = [
    '---',
    `title: ${JSON.stringify(title)}`,
    ...(description ? [`description: ${JSON.stringify(description)}`] : []),
    '---',
    '',
  ].join('\n');
  const dest = path.join(outDir, 'docs', targetPath(rel));
  mkdirSync(path.dirname(dest), { recursive: true });
  writeFileSync(dest, frontmatter + rewritten);
}

const nav = parseNav(readFileSync(path.join(docsDir, 'README.md'), 'utf8'), titles);
const inNav = collectUrls(nav.flatMap((section) => section.items));
const missing = files.map(pageUrl).filter((url) => !inNav.has(url));
if (missing.length) throw new Error(`pages missing from the navigation: ${missing.join(', ')}`);

writeFileSync(path.join(outDir, 'nav.json'), `${JSON.stringify(nav, null, 2)}\n`);
console.log(`prepared ${files.length} pages from docs/ into ${path.relative(websiteDir, outDir)}/`);
