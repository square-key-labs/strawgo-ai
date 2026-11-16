import type { Metadata, Viewport } from 'next';
import { Provider } from '@/components/provider';
import { siteUrl } from '@/lib/shared';
import '@fontsource-variable/hanken-grotesk/wght.css';
import '@fontsource-variable/jetbrains-mono/wght.css';
import './global.css';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: 'Strawgo: real-time voice agents in Go',
    template: '%s | Strawgo',
  },
  description:
    'Strawgo is a Go framework for real-time voice agents: programs that listen to a caller, understand what they say, and answer out loud.',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f7f8f9' },
    { media: '(prefers-color-scheme: dark)', color: '#0e1014' },
  ],
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex min-h-dvh flex-col">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
