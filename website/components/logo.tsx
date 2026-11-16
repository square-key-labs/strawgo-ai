import type { SVGProps } from 'react';

export function LogoMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
      <rect width="24" height="24" rx="6" fill="var(--sg-accent)" />
      <g fill="var(--sg-on-accent)">
        <rect x="5" y="10" width="2.4" height="4" rx="1.2" />
        <rect x="9.2" y="6" width="2.4" height="12" rx="1.2" />
        <rect x="13.4" y="8" width="2.4" height="8" rx="1.2" />
        <rect x="17.6" y="10.5" width="2.4" height="3" rx="1.2" />
      </g>
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="sg-wordmark">
      <LogoMark className="size-6 shrink-0" />
      <span>Strawgo</span>
    </span>
  );
}
