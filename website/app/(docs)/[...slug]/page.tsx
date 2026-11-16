import { source } from '@/lib/source';
import {
  DocsBody,
  DocsPage,
  DocsTitle,
  EditOnGitHub,
} from 'fumadocs-ui/layouts/notebook/page';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { getMDXComponents } from '@/components/mdx';
import { branch, repoUrl, sourcePathOf } from '@/lib/shared';

type Props = { params: Promise<{ slug: string[] }> };

export default async function Page(props: Props) {
  const { slug } = await props.params;
  const page = source.getPage(slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const editUrl = `${repoUrl}/edit/${branch}/${sourcePathOf(page.path)}`;

  return (
    <DocsPage toc={page.data.toc} tableOfContent={{ style: 'clerk' }}>
      <DocsTitle className="sg-title">{page.data.title}</DocsTitle>
      <DocsBody className="sg-docs-body">
        <MDX components={getMDXComponents()} />
      </DocsBody>
      <EditOnGitHub href={editUrl} className="sg-edit" />
    </DocsPage>
  );
}

export function generateStaticParams() {
  return source.generateParams().filter((params) => params.slug.length > 0);
}

export async function generateMetadata(props: Props): Promise<Metadata> {
  const { slug } = await props.params;
  const page = source.getPage(slug);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
  };
}
