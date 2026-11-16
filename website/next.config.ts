import { createMDX } from 'fumadocs-mdx/next';
import type { NextConfig } from 'next';
import { basePath } from './lib/shared';

const withMDX = createMDX();

const config: NextConfig = {
  output: 'export',
  basePath,
  trailingSlash: true,
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default withMDX(config);
