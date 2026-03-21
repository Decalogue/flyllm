import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BulbOutlined,
  EditOutlined,
  DatabaseOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';
import { CREATOR_THEME } from './creatorTheme';

const T = CREATOR_THEME;

export interface AgentSlot {
  key: string;
  name: string;
  icon: React.ReactNode;
  color: string;
}

export interface OrchestrationFlowProps {
  agents: AgentSlot[];
  completed: string[];
  active: string | null;
  /** 是否显示流动光效 */
  flow?: boolean;
}

export const OrchestrationFlow: React.FC<OrchestrationFlowProps> = ({
  agents,
  completed,
  active,
  flow = true,
}) => {
  const [hovered, setHovered] = useState<string | null>(null);
  const runningCount = active ? 1 : 0;
  const doneCount = completed.length;

  return (
    <div
      style={{
        background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%)',
        borderRadius: 16,
        border: `1px solid ${T.borderStrong}`,
        overflow: 'hidden',
      }}
    >
      {/* 头部 — Agent 指挥中心 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 16px',
          borderBottom: `1px solid ${T.border}`,
          background: 'rgba(255,255,255,0.03)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ position: 'relative' }}>
            <ThunderboltOutlined style={{ fontSize: 18, color: T.accent }} />
            {active && (
              <span
                style={{
                  position: 'absolute',
                  top: -2,
                  right: -2,
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#22c55e',
                  boxShadow: '0 0 8px rgba(34,197,94,0.6)',
                }}
              />
            )}
          </div>
          <span style={{ fontSize: 14, fontWeight: T.fontWeightBold, color: T.textBright }}>
            Agent 指挥中心
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#22c55e',
            }}
          />
          <span style={{ fontSize: 11, color: '#22c55e', fontFamily: 'monospace' }}>
            系统正常
          </span>
        </div>
      </div>

      {/* Agent 列表 */}
      <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {agents.map((a, i) => {
          const done = completed.includes(a.key);
          const isActive = active === a.key;
          const isHovered = hovered === a.key;
          const status = isActive ? 'running' : done ? 'completed' : 'idle';
          const task =
            isActive ? '执行中' : done ? '已完成' : '等待输入';

          return (
            <motion.div
              key={a.key}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05, duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
              onMouseEnter={() => setHovered(a.key)}
              onMouseLeave={() => setHovered(null)}
              style={{
                position: 'relative',
                padding: 14,
                borderRadius: 12,
                cursor: 'default',
                background: isActive
                  ? `linear-gradient(90deg, ${a.color}18 0%, transparent 100%)`
                  : 'rgba(255,255,255,0.03)',
                border: `1px solid ${isActive ? `${a.color}40` : 'transparent'}`,
                overflow: 'hidden',
              }}
            >
              {flow && isActive && (
                <motion.div
                  animate={{ opacity: [0.2, 0.6, 0.2] }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: `linear-gradient(90deg, transparent, ${a.color}25, transparent)`,
                  }}
                />
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: 12, position: 'relative' }}>
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 12,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: `${a.color}25`,
                    color: a.color,
                    flexShrink: 0,
                    transform: isActive ? 'scale(1.05)' : undefined,
                  }}
                >
                  {a.icon}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: T.fontWeightSemibold, color: T.textBright }}>
                      {a.name}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        padding: '2px 8px',
                        borderRadius: 10,
                        background: `${a.color}20`,
                        color: a.color,
                      }}
                    >
                      {status === 'running' && '执行中'}
                      {status === 'completed' && '已完成'}
                      {status === 'idle' && '待机'}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: T.textMuted,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}
                  >
                    {isActive && (
                      <motion.span
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        style={{ display: 'inline-flex' }}
                      >
                        ◐
                      </motion.span>
                    )}
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {task}
                    </span>
                  </div>

                  {isActive && (
                    <div
                      style={{
                        marginTop: 8,
                        height: 4,
                        borderRadius: 2,
                        background: 'rgba(255,255,255,0.08)',
                        overflow: 'hidden',
                      }}
                    >
                      <motion.div
                        animate={{ width: ['0%', '100%'] }}
                        transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 0.3, ease: 'easeInOut' }}
                        style={{
                          height: '100%',
                          background: a.color,
                          borderRadius: 2,
                        }}
                      />
                    </div>
                  )}
                </div>

                {done && !isActive && (
                  <div
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      background: '#22c55e',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <CheckCircleOutlined style={{ fontSize: 10, color: '#fff' }} />
                  </div>
                )}
              </div>

              {isHovered && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                  style={{
                    marginTop: 12,
                    paddingTop: 12,
                    borderTop: `1px solid ${T.border}`,
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr 1fr',
                    gap: 8,
                    textAlign: 'center',
                  }}
                >
                  <div>
                    <div style={{ fontSize: 10, color: T.textDim, marginBottom: 2 }}>状态</div>
                    <div style={{ fontSize: 11, fontWeight: T.fontWeightMedium, color: a.color }}>
                      {task}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: T.textDim, marginBottom: 2 }}>进度</div>
                    <div style={{ fontSize: 11, color: T.text }}>
                      {isActive ? '…' : done ? '100%' : '0%'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: T.textDim, marginBottom: 2 }}>记忆</div>
                    <div style={{ fontSize: 11, color: T.text }}>—</div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* 底部统计 */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr 1fr',
          gap: 8,
          padding: 12,
          borderTop: `1px solid ${T.border}`,
          background: 'rgba(255,255,255,0.03)',
          textAlign: 'center',
        }}
      >
        <div>
          <div style={{ fontSize: 15, fontWeight: T.fontWeightBold, color: T.accent }}>{runningCount}</div>
          <div style={{ fontSize: 10, color: T.textDim }}>运行中</div>
        </div>
        <div>
          <div style={{ fontSize: 15, fontWeight: T.fontWeightBold, color: '#22c55e' }}>{doneCount}</div>
          <div style={{ fontSize: 10, color: T.textDim }}>已完成</div>
        </div>
        <div>
          <div style={{ fontSize: 15, fontWeight: T.fontWeightBold, color: '#3b82f6' }}>
            {agents.length}
          </div>
          <div style={{ fontSize: 10, color: T.textDim }}>智能体</div>
        </div>
        <div>
          <div style={{ fontSize: 15, fontWeight: T.fontWeightBold, color: '#a78bfa' }}>—</div>
          <div style={{ fontSize: 10, color: T.textDim }}>平均效率</div>
        </div>
      </div>
    </div>
  );
};
