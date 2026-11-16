export const appName = 'Strawgo';
export const basePath = '/strawgo-ai';
export const siteUrl = 'https://square-key-labs.github.io/strawgo-ai/';
export const repoUrl = 'https://github.com/square-key-labs/strawgo-ai';
export const branch = 'main';

export function sourcePathOf(contentPath: string) {
  if (contentPath === 'overview.md') return 'docs/README.md';
  if (contentPath === 'index.md' || contentPath.endsWith('/index.md')) {
    return `docs/${contentPath.slice(0, -'index.md'.length)}README.md`;
  }
  return `docs/${contentPath}`;
}
