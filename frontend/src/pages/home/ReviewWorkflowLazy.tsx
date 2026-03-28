import React, { Suspense, lazy } from 'react';
import type { WorkflowGraphProps } from '@/components/creator/WorkflowGraph';

const WorkflowGraph = lazy(() =>
  import('@/components/creator/WorkflowGraph').then((m) => ({ default: m.WorkflowGraph }))
);

type Props = WorkflowGraphProps & { className?: string; loadingLabel?: string };

function Skeleton({ height, width, label }: { height: number; width: number; label: string }) {
  return (
    <div
      className="review-workflow-skeleton"
      style={{
        width,
        height,
        maxWidth: '100%',
        margin: '0 auto',
        minHeight: 280,
      }}
      role="status"
      aria-label={label}
    />
  );
}

export function ReviewWorkflowLazy({ className, height, width, loadingLabel = '…', ...rest }: Props) {
  return (
    <Suspense fallback={<Skeleton height={height} width={width} label={loadingLabel} />}>
      <div className={className} style={{ display: 'flex', justifyContent: 'center' }}>
        <WorkflowGraph height={height} width={width} {...rest} />
      </div>
    </Suspense>
  );
}
