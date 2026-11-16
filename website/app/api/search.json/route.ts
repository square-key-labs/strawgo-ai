import { pageTree, source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';
import type * as PageTree from 'fumadocs-core/page-tree';

export const revalidate = false;

const trails = new Map<string, string[]>();

function collect(nodes: PageTree.Node[], trail: string[]) {
  for (const node of nodes) {
    if (node.type === 'folder') collect(node.children, [...trail, String(node.name)]);
    else if (node.type === 'page' && !trails.has(node.url)) trails.set(node.url, trail);
  }
}

collect(pageTree.children, []);

export const { staticGET: GET } = createFromSource(source, {
  language: 'english',
  async buildIndex(page) {
    return {
      title: page.data.title,
      description: page.data.description,
      url: page.url,
      id: page.url,
      structuredData: page.data.structuredData,
      breadcrumbs: trails.get(page.url) ?? [],
    };
  },
});
