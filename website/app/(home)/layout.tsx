import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';
import { sections } from '@/lib/source';
import './home.css';

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <HomeLayout
      {...baseOptions()}
      links={sections.map((section) => ({ text: section.title, url: section.url, active: 'none' as const }))}
    >
      {children}
    </HomeLayout>
  );
}
