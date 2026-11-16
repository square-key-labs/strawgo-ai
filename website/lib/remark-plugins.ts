import type { BlockContent, Code, Parent, Root, RootContent } from 'mdast';

const alertKinds = ['note', 'tip', 'important', 'warning', 'caution'] as const;
const alertMarker = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*(?:\r?\n)?/;

const codeTitles: Record<string, string> = {
  go: 'Go',
  sh: 'Terminal',
  bash: 'Terminal',
  shell: 'Terminal',
  json: 'JSON',
  yaml: 'YAML',
  yml: 'YAML',
  xml: 'XML',
  html: 'HTML',
  js: 'JavaScript',
  ts: 'TypeScript',
  dockerfile: 'Dockerfile',
  text: 'Text',
};

function visit(node: Root | RootContent, fn: (node: RootContent, index: number, parent: Parent) => void) {
  if (!('children' in node)) return;
  const parent = node as Parent;
  parent.children.forEach((child, index) => {
    fn(child, index, parent);
    visit(child, fn);
  });
}

export function remarkGithubAlerts() {
  return (tree: Root) => {
    visit(tree, (node, index, parent) => {
      if (node.type !== 'blockquote') return;
      const first = node.children[0];
      if (first?.type !== 'paragraph') return;
      const lead = first.children[0];
      if (lead?.type !== 'text') return;
      const match = alertMarker.exec(lead.value);
      if (!match) return;
      const kind = match[1].toLowerCase() as (typeof alertKinds)[number];
      lead.value = lead.value.slice(match[0].length);
      if (!lead.value) first.children.shift();
      if (first.children[0]?.type === 'break') first.children.shift();
      const children = (first.children.length ? node.children : node.children.slice(1)) as BlockContent[];
      parent.children[index] = {
        type: 'mdxJsxFlowElement',
        name: 'Alert',
        attributes: [{ type: 'mdxJsxAttribute', name: 'kind', value: kind }],
        children,
      } as unknown as RootContent;
    });
  };
}

const detailsOpen = /^<details>\s*(?:<summary>([\s\S]*?)<\/summary>)?\s*$/;

function jsx(name: string, children: unknown[]) {
  return { type: 'mdxJsxFlowElement', name, attributes: [], children } as unknown as RootContent;
}

export function remarkHtmlDetails() {
  return (tree: Root, file: { path?: string }) => {
    const rewrite = (parent: Parent) => {
      for (let i = 0; i < parent.children.length; i++) {
        const node = parent.children[i];
        if (node.type === 'html') {
          const open = detailsOpen.exec(node.value.trim());
          const close = parent.children.findIndex(
            (child, j) => j > i && child.type === 'html' && child.value.trim() === '</details>',
          );
          if (!open || close === -1) {
            throw new Error(`${file.path ?? 'page'}: unsupported raw HTML ${JSON.stringify(node.value.slice(0, 60))}`);
          }
          const summary = jsx('summary', [{ type: 'text', value: (open[1] ?? 'Details').trim() }]);
          const body = parent.children.slice(i + 1, close);
          parent.children.splice(i, close - i + 1, jsx('details', [summary, ...body]));
        }
        if ('children' in parent.children[i]) rewrite(parent.children[i] as Parent);
      }
    };
    rewrite(tree);
  };
}

export function remarkCodeTitles() {
  return (tree: Root) => {
    visit(tree, (node) => {
      if (node.type !== 'code') return;
      const code = node as Code;
      if (!code.lang || /\btitle=/.test(code.meta ?? '')) return;
      const title = codeTitles[code.lang.toLowerCase()];
      if (!title) return;
      code.meta = `${code.meta ?? ''} title="${title}"`.trim();
    });
  };
}
