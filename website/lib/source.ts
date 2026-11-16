import { loader } from 'fumadocs-core/source';
import type * as PageTree from 'fumadocs-core/page-tree';
import { remarkMdxMermaid } from 'fumadocs-core/mdx-plugins';
import { applyMdxPreset } from 'fumadocs-mdx/config';
import { defineDocs } from 'fumadocs-mdx/macro';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import { remarkCodeTitles, remarkGithubAlerts, remarkHtmlDetails } from './remark-plugins';
import nav from '../content/nav.json';

const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema,
    mdxOptions: applyMdxPreset({
      remarkPlugins: [remarkHtmlDetails, remarkGithubAlerts, remarkMdxMermaid, remarkCodeTitles],
      rehypeCodeOptions: {
        themes: { light: 'github-light', dark: 'github-dark' },
      },
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export const source = loader({
  baseUrl: '/',
  source: docs.toFumadocsSource(),
});

type NavItem =
  | { type: 'page'; title: string; url: string }
  | { type: 'external'; title: string; url: string }
  | { type: 'folder'; title: string; items: NavItem[] };

type NavSection = { title: string; items: NavItem[] };

function toNode(item: NavItem, id: string): PageTree.Node {
  if (item.type === 'folder') {
    return {
      $id: id,
      type: 'folder',
      name: item.title,
      defaultOpen: true,
      children: item.items.map((child, i) => toNode(child, `${id}.${i}`)),
    };
  }
  return {
    $id: id,
    type: 'page',
    name: item.title,
    url: item.url,
    external: item.type === 'external',
  };
}

export const sections = (nav as NavSection[]).map((section) => ({
  title: section.title,
  url: (section.items.find((item) => item.type === 'page') as { url: string }).url,
}));

export const pageTree: PageTree.Root = {
  $id: 'root',
  name: 'Strawgo',
  children: (nav as NavSection[]).map((section, i) => ({
    $id: `section.${i}`,
    type: 'folder',
    name: section.title,
    root: true,
    children: section.items.map((item, j) => toNode(item, `section.${i}.${j}`)),
  })),
};
