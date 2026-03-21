/** 记忆系统展示数据：实体、事实、原子笔记 → 节点；关系 → 边 */

export type NodeType = 'entity' | 'fact' | 'atom';

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  brief?: string;
  source?: string;
  /** 原子笔记正文片段 */
  body?: string;
  /** 关联节点（来自 /api/memory/note 时使用） */
  related?: Array<{ node: { id: string; label: string }; relation?: string }>;
}

export interface GraphLink {
  source: string;
  target: string;
  relation?: string;
}

export interface MemoryGraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

/** 首页「探索记忆系统」展示用：实体（跨章人物/设定）、事实（情节）、原子笔记（伏笔/长线） */
export const MEMORY_GRAPH_DATA: MemoryGraphData = {
  nodes: [
    { id: 'e1', label: '主角', type: 'entity', brief: '跨章人物' },
    { id: 'e2', label: '导师', type: 'entity', brief: '跨章人物' },
    { id: 'e3', label: '玄灵大陆', type: 'entity', brief: '长线设定' },
    { id: 'e4', label: '灵科体系', type: 'entity', brief: '长线设定' },
    { id: 'e5', label: '天枢城', type: 'entity', brief: '地点' },
    { id: 'e6', label: '宗门大比', type: 'entity', brief: '事件' },
    { id: 'f1', label: '第3章关键抉择', type: 'fact', source: '第3章' },
    { id: 'f2', label: '第7章身份揭露', type: 'fact', source: '第7章' },
    { id: 'f3', label: '初入天枢', type: 'fact', source: '第1章' },
    { id: 'f4', label: '拜师入门', type: 'fact', source: '第2章' },
    { id: 'a1', label: '神秘吊坠', type: 'atom', brief: '伏笔', body: '与身世、巨门相关。' },
    { id: 'a2', label: '千年封印', type: 'atom', brief: '伏笔', body: '灵气复苏纪元，封印到期倒计时。' },
    { id: 'a3', label: '七脉会武', type: 'atom', brief: '长线', body: '三年一度，决定资源分配。' },
    { id: 'a4', label: '上古传承', type: 'atom', brief: '伏笔', body: '主角血脉与秘境呼应。' },
  ],
  links: [
    { source: 'e1', target: 'e2', relation: '师徒' },
    { source: 'e1', target: 'e3', relation: '所在' },
    { source: 'e1', target: 'e4', relation: '修炼' },
    { source: 'e1', target: 'e5', relation: '抵达' },
    { source: 'e1', target: 'e6', relation: '参与' },
    { source: 'e1', target: 'f1', relation: '涉及' },
    { source: 'e1', target: 'f2', relation: '涉及' },
    { source: 'e1', target: 'f3', relation: '涉及' },
    { source: 'e1', target: 'f4', relation: '涉及' },
    { source: 'e1', target: 'a1', relation: '持有' },
    { source: 'e1', target: 'a4', relation: '关联' },
    { source: 'e2', target: 'f1', relation: '涉及' },
    { source: 'e2', target: 'f4', relation: '涉及' },
    { source: 'e3', target: 'a2', relation: '关联' },
    { source: 'e3', target: 'a3', relation: '设定' },
    { source: 'e5', target: 'f3', relation: '发生地' },
    { source: 'e6', target: 'a3', relation: '对应' },
    { source: 'a1', target: 'a2', relation: '线索' },
    { source: 'a1', target: 'a4', relation: '线索' },
    { source: 'f1', target: 'f2', relation: '因果' },
  ],
};
