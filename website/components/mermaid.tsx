'use client';

import { useEffect, useId, useState, useSyncExternalStore } from 'react';

const palettes = {
  light: {
    background: '#fbfbfc',
    primaryColor: '#fdf3e1',
    primaryTextColor: '#1b1d22',
    primaryBorderColor: '#c98a2b',
    secondaryColor: '#eef0f3',
    tertiaryColor: '#f4f5f7',
    lineColor: '#6b7180',
    textColor: '#1b1d22',
    mainBkg: '#fdf3e1',
    nodeBorder: '#c98a2b',
    clusterBkg: '#f4f5f7',
    clusterBorder: '#d6d9df',
    edgeLabelBackground: '#fbfbfc',
    actorBkg: '#fdf3e1',
    actorBorder: '#c98a2b',
    actorTextColor: '#1b1d22',
    actorLineColor: '#9aa0ab',
    signalColor: '#3a3f4a',
    signalTextColor: '#1b1d22',
    labelBoxBkgColor: '#fdf3e1',
    labelBoxBorderColor: '#c98a2b',
    labelTextColor: '#1b1d22',
    noteBkgColor: '#eef0f3',
    noteBorderColor: '#c3c7cf',
    noteTextColor: '#1b1d22',
    activationBkgColor: '#f6e2bd',
    activationBorderColor: '#c98a2b',
  },
  dark: {
    background: '#12141a',
    primaryColor: '#2a2214',
    primaryTextColor: '#eceef2',
    primaryBorderColor: '#d9a246',
    secondaryColor: '#1d2129',
    tertiaryColor: '#181b22',
    lineColor: '#8f96a3',
    textColor: '#eceef2',
    mainBkg: '#2a2214',
    nodeBorder: '#d9a246',
    clusterBkg: '#181b22',
    clusterBorder: '#2e333d',
    edgeLabelBackground: '#12141a',
    actorBkg: '#2a2214',
    actorBorder: '#d9a246',
    actorTextColor: '#eceef2',
    actorLineColor: '#5d6470',
    signalColor: '#c9ced8',
    signalTextColor: '#eceef2',
    labelBoxBkgColor: '#2a2214',
    labelBoxBorderColor: '#d9a246',
    labelTextColor: '#eceef2',
    noteBkgColor: '#1d2129',
    noteBorderColor: '#3a404b',
    noteTextColor: '#eceef2',
    activationBkgColor: '#3a2e18',
    activationBorderColor: '#d9a246',
  },
};

let queue: Promise<unknown> = Promise.resolve();

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => observer.disconnect();
}

function readTheme() {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function useDocumentTheme() {
  return useSyncExternalStore(subscribe, readTheme, () => null);
}

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, '');
  const resolvedTheme = useDocumentTheme();
  const [svg, setSvg] = useState<string>('');
  const [failed, setFailed] = useState(false);
  const [natural, setNatural] = useState(0);
  const theme = resolvedTheme === 'dark' ? 'dark' : 'light';

  useEffect(() => {
    if (!resolvedTheme) return;
    let cancelled = false;
    const run = async () => {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        themeVariables: { ...palettes[theme], fontFamily: 'Hanken Grotesk Variable, ui-sans-serif, system-ui, sans-serif', fontSize: '15px' },
      });
      const { svg: rendered } = await mermaid.render(`mermaid-${id}-${theme}`, chart);
      if (!cancelled) {
        setSvg(rendered);
        setNatural(Number(/max-width:\s*([\d.]+)px/.exec(rendered)?.[1] ?? 0));
        setFailed(false);
      }
    };
    queue = queue.then(run).catch(() => {
      if (!cancelled) setFailed(true);
    });
    return () => {
      cancelled = true;
    };
  }, [chart, id, theme, resolvedTheme]);

  if (failed) {
    return (
      <pre className="sg-mermaid-fallback">
        <code>{chart}</code>
      </pre>
    );
  }

  return (
    <figure
      className="sg-mermaid not-prose"
      data-theme={theme}
      data-ready={svg ? 'true' : 'false'}
      style={natural ? ({ '--sg-mermaid-min': `${Math.min(natural, 720)}px` } as React.CSSProperties) : undefined}
    >
      <div className="sg-mermaid-canvas" dangerouslySetInnerHTML={{ __html: svg }} />
    </figure>
  );
}
