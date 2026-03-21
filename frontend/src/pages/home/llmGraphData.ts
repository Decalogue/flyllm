/**
 * 大模型八股知识星图 — 数据与工具函数
 * 节点与边来自 llm 目录全部 Markdown 的汇总（见 llmGraphData.corpus.ts）
 */
import type { BaguGroup, BaguLink, BaguNode } from './llmGraphTypes';
import { corpusLinks, corpusNodes } from './llmGraphData.corpus';

export type { BaguGroup, BaguLink, BaguNode } from './llmGraphTypes';
export { BAGU_GROUP_COLOR, BAGU_GROUP_LABEL } from './llmGraphTypes';

/** 星图节点（与 corpus 同步） */
export const baguNodes: BaguNode[] = corpusNodes;

/** 星图边 */
export const baguLinks: BaguLink[] = corpusLinks;

/** 按 Tab 过滤后仍保持连通的子图 */
export function filterBaguGraph(
  tab: 'all' | BaguGroup,
  nodes: BaguNode[],
  links: BaguLink[],
): { nodes: BaguNode[]; links: BaguLink[] } {
  if (tab === 'all') return { nodes, links };
  const idSet = new Set(nodes.filter((n) => n.group === tab).map((n) => n.id));
  const keptLinks = links.filter((l) => idSet.has(l.source) && idSet.has(l.target));
  const used = new Set<string>();
  keptLinks.forEach((l) => {
    used.add(l.source);
    used.add(l.target);
  });
  const keptNodes = nodes.filter((n) => used.has(n.id));
  return { nodes: keptNodes, links: keptLinks };
}

export function nodeDegrees(links: BaguLink[]): Map<string, number> {
  const m = new Map<string, number>();
  const add = (id: string) => m.set(id, (m.get(id) ?? 0) + 1);
  links.forEach((l) => {
    add(l.source);
    add(l.target);
  });
  return m;
}
