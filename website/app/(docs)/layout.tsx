import { DocsLayout } from 'fumadocs-ui/layouts/notebook';
import { baseOptions } from '@/lib/layout.shared';
import { pageTree } from '@/lib/source';

export default function Layout({ children }: { children: React.ReactNode }) {
  const { nav, ...base } = baseOptions();
  return (
    <DocsLayout {...base} tree={pageTree} tabMode="navbar" nav={{ ...nav, mode: 'top' }}>
      {children}
    </DocsLayout>
  );
}
