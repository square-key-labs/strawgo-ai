import Link from 'next/link';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';
import './(home)/home.css';

export default function NotFound() {
  return (
    <HomeLayout {...baseOptions()}>
      <main className="sg-home sg-missing">
        <h1>This page does not exist</h1>
        <p>The link may be out of date, or the page may have moved when the docs were reorganised.</p>
        <div className="sg-hero-actions">
          <Link href="/" className="sg-btn sg-btn-primary">
            Back to the start
          </Link>
          <Link href="/overview" className="sg-btn sg-btn-secondary">
            Documentation map
          </Link>
        </div>
      </main>
    </HomeLayout>
  );
}
