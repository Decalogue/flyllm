import React from 'react';
import { Drawer, Tag } from 'antd';
import { CloseOutlined } from '@ant-design/icons';
import { motion } from 'framer-motion';
import type { GraphNode, MemoryGraphData } from './memoryGraphData';
import { CREATOR_THEME } from './creatorTheme';

const T = CREATOR_THEME;

export interface MemoryDetailDrawerProps {
  open: boolean;
  onClose: () => void;
  node: GraphNode | null;
  graphData: MemoryGraphData;
}

const typeLabel: Record<string, string> = {
  entity: '实体',
  fact: '事实',
  atom: '原子笔记',
};

const typeIcon: Record<string, string> = {
  entity: '◆',
  fact: '◇',
  atom: '◎',
};

export const MemoryDetailDrawer: React.FC<MemoryDetailDrawerProps> = ({
  open,
  onClose,
  node,
  graphData,
}) => {
  if (!node) return null;

  const related = (node.related && node.related.length > 0)
    ? node.related
    : graphData.links
        .filter((l) => l.source === node.id || l.target === node.id)
        .map((l) => {
          const otherId = l.source === node.id ? l.target : l.source;
          const other = graphData.nodes.find((n) => n.id === otherId);
          return { node: other, relation: l.relation };
        })
        .filter((r) => r.node);

  const accent = T.accent;

  return (
    <Drawer
      placement="right"
      width={400}
      open={open}
      onClose={onClose}
      styles={{
        body: {
          padding: 0,
          background: 'linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%)',
          color: T.text,
        },
        header: {
          display: 'none',
        },
      }}
    >
      <div style={{ position: 'relative', minHeight: '100%' }}>
        {/* 头部渐变光晕 */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            opacity: 0.2,
            pointerEvents: 'none',
            background: `radial-gradient(circle at top right, ${accent}, transparent 55%)`,
          }}
        />

        {/* 主内容 */}
        <div style={{ position: 'relative', padding: 24 }}>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            style={{
              position: 'absolute',
              top: 16,
              right: 16,
              width: 36,
              height: 36,
              borderRadius: 10,
              border: 'none',
              background: 'rgba(255,255,255,0.08)',
              color: T.text,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1,
            }}
          >
            <CloseOutlined />
          </button>
          {/* 标题区 */}
          <div style={{ marginBottom: 20 }}>
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: 16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 24,
                background: `${accent}25`,
                color: accent,
                marginBottom: 12,
              }}
            >
              {typeIcon[node.type] || '◦'}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
              <Tag
                style={{
                  margin: 0,
                  background: `${accent}20`,
                  color: accent,
                  border: 'none',
                  fontSize: 11,
                  fontWeight: T.fontWeightSemibold,
                }}
              >
                {typeLabel[node.type] || node.type}
              </Tag>
              {node.source && (
                <span style={{ fontSize: 11, color: T.textMuted }}>{node.source}</span>
              )}
            </div>
            <h2
              style={{
                fontSize: 18,
                fontWeight: T.fontWeightBold,
                color: T.textBright,
                margin: 0,
                lineHeight: 1.35,
              }}
            >
              {node.label}
            </h2>
          </div>

          {/* 记忆内容 */}
          {(node.brief || node.body) && (
            <div style={{ marginBottom: 20 }}>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: T.fontWeightSemibold,
                  color: T.textMuted,
                  marginBottom: 8,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                记忆内容
              </div>
              <div
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: 12,
                  padding: 14,
                  fontSize: 13,
                  lineHeight: 1.6,
                  color: 'rgba(255,255,255,0.85)',
                }}
              >
                {node.body || node.brief}
              </div>
            </div>
          )}

          {/* 元信息 */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 12,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: 12,
                padding: 12,
              }}
            >
              <div style={{ fontSize: 10, color: T.textDim, marginBottom: 4 }}>关联数</div>
              <div style={{ fontSize: 16, fontWeight: T.fontWeightBold, color: T.textBright }}>
                {related.length}
              </div>
            </div>
            <div
              style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: 12,
                padding: 12,
              }}
            >
              <div style={{ fontSize: 10, color: T.textDim, marginBottom: 4 }}>类型</div>
              <div style={{ fontSize: 13, fontWeight: T.fontWeightSemibold, color: accent }}>
                {typeLabel[node.type] || node.type}
              </div>
            </div>
          </div>

          {/* 关联记忆 */}
          {related.length > 0 && (
            <div>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: T.fontWeightSemibold,
                  color: T.textMuted,
                  marginBottom: 10,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                关联记忆
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {related.map((r, idx) => {
                  const n = r.node as { id?: string; label?: string; type?: string } | undefined;
                  const rid = n?.id ?? '';
                  const rlabel = n?.label ?? '';
                  const rtype = n?.type ?? 'entity';
                  return (
                    <motion.div
                      key={rid || rlabel || `rel-${idx}`}
                      whileHover={{ background: 'rgba(255,255,255,0.08)' }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        padding: 12,
                        background: 'rgba(255,255,255,0.04)',
                        borderRadius: 12,
                        cursor: 'default',
                        border: '1px solid transparent',
                      }}
                    >
                      <span style={{ fontSize: 16 }}>{typeIcon[rtype] || '◦'}</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: T.fontWeightMedium, color: T.textBright }}>
                          {rlabel}
                        </div>
                        {r.relation && (
                          <div style={{ fontSize: 11, color: T.textMuted }}>{r.relation}</div>
                        )}
                      </div>
                      <div
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          background: accent,
                          flexShrink: 0,
                        }}
                      />
                    </motion.div>
                  );
                })}
              </div>
            </div>
          )}

          {/* 底部操作（占位，可后续接编辑） */}
          <div
            style={{
              marginTop: 24,
              paddingTop: 16,
              borderTop: `1px solid ${T.border}`,
              display: 'flex',
              gap: 12,
            }}
          >
            <button
              type="button"
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: 12,
                border: 'none',
                background: accent,
                color: '#fff',
                fontSize: 13,
                fontWeight: T.fontWeightSemibold,
                cursor: 'pointer',
              }}
            >
              编辑记忆
            </button>
            <button
              type="button"
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: 12,
                border: `1px solid ${T.borderStrong}`,
                background: 'rgba(255,255,255,0.06)',
                color: T.text,
                fontSize: 13,
                fontWeight: T.fontWeightMedium,
                cursor: 'pointer',
              }}
            >
              创建关联
            </button>
          </div>
        </div>
      </div>
    </Drawer>
  );
};
