/** 八股星图 — 类型与分组色（供 llmGraphData / corpus 共用） */

export type BaguGroup = 'arch' | 'train' | 'infer' | 'app';

export interface BaguNode {
  id: string;
  label: string;
  group: BaguGroup;
  tier: 1 | 2 | 3;
  tips: string[];
  /** 仓库内 Markdown 路径，如 llm/TransformerArchitecture.md */
  sourcePath: string;
}

export interface BaguLink {
  source: string;
  target: string;
}

export const BAGU_GROUP_LABEL: Record<BaguGroup, string> = {
  arch: '架构',
  train: '训练与对齐',
  infer: '推理与加速',
  app: '应用与系统',
};

export const BAGU_GROUP_COLOR: Record<BaguGroup, string> = {
  arch: '#7dd3fc',
  train: '#f0abfc',
  infer: '#fde047',
  app: '#86efac',
};
