import type { ReactNode } from 'react';
import { CircleAlert, Info, Lightbulb, MessageSquareWarning, TriangleAlert } from 'lucide-react';

const kinds = {
  note: { label: 'Note', Icon: Info },
  tip: { label: 'Tip', Icon: Lightbulb },
  important: { label: 'Important', Icon: MessageSquareWarning },
  warning: { label: 'Warning', Icon: TriangleAlert },
  caution: { label: 'Caution', Icon: CircleAlert },
} as const;

export function Alert({ kind = 'note', children }: { kind?: keyof typeof kinds; children: ReactNode }) {
  const { label, Icon } = kinds[kind] ?? kinds.note;
  return (
    <aside className="sg-alert" data-kind={kind} aria-label={label}>
      <p className="sg-alert-title not-prose">
        <Icon aria-hidden="true" strokeWidth={2} />
        {label}
      </p>
      <div className="sg-alert-body">{children}</div>
    </aside>
  );
}
