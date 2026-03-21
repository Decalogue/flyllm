#!/usr/bin/env node
/**
 * 从仓库根目录 llm 下所有 md 生成 llmGraphData.corpus.ts
 * 在 frontend 目录执行: node scripts/build-llm-graph-data.mjs
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.join(__dirname, '..');
const REPO_ROOT = path.join(FRONTEND, '..');
const LLM_DIR = path.join(REPO_ROOT, 'llm');
const OUT_FILE = path.join(FRONTEND, 'src/pages/home/llmGraphData.corpus.ts');

const SPECIAL = {
  显存计算: { id: 'gpu-memory-estimation', label: '显存计算' },
};

const HUB_IDS = new Set([
  'transformer-architecture',
  'self-attention-math',
  'rag',
  'rlhf-full-process',
  'tool-calling',
  'react-framework',
  'flash-attention',
  'lora-principle',
  'agent-core-modules',
  'multi-head-attention',
  'vllm-continuous-batching',
  'paged-attention',
  'memory-hierarchy-architecture',
  'multi-agent-system-design',
]);

const GROUP_SPOKE_HUB = {
  arch: 'transformer-architecture',
  train: 'rlhf-full-process',
  infer: 'flash-attention',
  app: 'rag',
};

function walkMdFiles(dir, acc = []) {
  if (!fs.existsSync(dir)) {
    console.error('Missing directory:', LLM_DIR);
    process.exit(1);
  }
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walkMdFiles(p, acc);
    else if (ent.name.endsWith('.md')) acc.push(p);
  }
  return acc;
}

function toKebabId(str) {
  const s = str
    .replace(/_/g, '-')
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/([A-Z])([A-Z][a-z])/g, '$1-$2')
    .toLowerCase();
  return s
    .replace(/[^a-z0-9\u4e00-\u9fff-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function relToId(relFromLlm) {
  const norm = relFromLlm.replace(/\\/g, '/');
  const base = path.basename(norm, '.md');
  if (SPECIAL[base]) return SPECIAL[base].id;
  const dir = path.dirname(norm);
  const combined = dir === '.' ? base : `${dir}-${base}`.replace(/\//g, '-');
  return toKebabId(combined) || `doc-${Buffer.from(norm).toString('hex').slice(0, 10)}`;
}

function defaultLabelFromBase(base) {
  return base
    .replace(/([A-Z])/g, ' $1')
    .replace(/[-_]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseTitle(raw) {
  const lines = raw.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i].trim();
    if (L.startsWith('#')) return L.replace(/^#+\s*/, '').trim();
  }
  return '';
}

function parseTips(raw) {
  const lines = raw.split(/\r?\n/);
  let i = 0;
  for (; i < lines.length; i++) {
    if (lines[i].trim().startsWith('#')) break;
  }
  const rest = lines.slice(i + 1).join('\n');
  let core = '';
  const m = rest.match(/##\s*1\.\s*核心定性\s*([\s\S]*?)(?=##\s*2\.|$)/);
  if (m) core = m[1];
  else {
    const m2 = rest.match(/##\s*1[^\n]*\n+([\s\S]*?)(?=##\s*2\.|$)/);
    if (m2) core = m2[1];
  }
  let tipText = core || rest.slice(0, 900);
  tipText = tipText
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$]+\$/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[#>*|`]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const sentences = tipText.split(/(?<=[。！？!?])\s+/).filter((s) => s.length > 10);
  const tips = [];
  if (sentences[0]) tips.push(sentences[0].slice(0, 200));
  if (sentences[1]) tips.push(sentences[1].slice(0, 200));
  const title = parseTitle(raw);
  if (tips.length === 0 && title) tips.push(title.length > 200 ? `${title.slice(0, 197)}…` : title);
  if (tips.length === 0) tips.push('详见仓库 llm 目录对应 Markdown。');
  return tips.slice(0, 3);
}

function classifyGroup(relFromLlm, base, title) {
  const s = `${relFromLlm} ${base} ${title}`.toLowerCase();
  if (
    /rag|retrieval|embedding|bm25|chunk|rerank|vector|hybrid|memory|skill|tool|agent|react|langchain|memgpt|function|call|orchestr|multi-agent|query|expansion|milvus|qdrant|light.?rag|ocr|composab|selection|writing|storm|blind|scoring|conflict|hierarchy|evaluation|metrics|layered|virtual|context|skill|library/.test(
      s,
    )
  )
    return 'app';
  if (
    /flash|kv|cache|quant|inference|batch|vllm|stream|decode|latency|throughput|paged|sglang|parallelism|zero|continuous|batching|linear.?attention|sparse|local.?attention|longformer|model.?inference|batch.?infer|bf16|fp16|stream|llmstream|cache.?design|parallel|显存/.test(
      s,
    )
  )
    return 'infer';
  if (
    /rlhf|ppo|dpo|sft|lora|qlora|reward|policy|finetun|pretrain|grpo|adapter|alignment|instruction|catastrophic|stopping|kl|actor|critic|offline|dapo|synthesis|data|augmentation|exploration|exploitation|tokenizer|bpe|sentencepiece|wordpiece|subword|negative|value|function|hacking|reinforce|actorcritic|infonce|full.?vs|parameter.?efficient|multi.?task|early|learning.?rate|loss.?design|instructions|construction/.test(
      s,
    )
  )
    return 'train';
  if (
    /attention|transformer|rope|norm|mha|gqa|token|positional|decoder|encoder|sparse|causal|moe|alibi|cross|self|architecture|special|vocabulary|vanishing|pattern|decoder-only|longformer|causal.?mask|complexity|multi.?head|self.?attention|linear|local|batch|norm|rms|layer|wordpiece|unigram|what-is-tokenizer|vocabulary|subword|sentencepiece|bpe/.test(
      s,
    )
  )
    return 'arch';
  return 'app';
}

function extractMdRefs(content, fromFile) {
  const set = new Set();
  const re = /(?:\]\(|href=\s*["']?)([^)\s"']+\.md)/gi;
  let m;
  while ((m = re.exec(content))) {
    const rel = m[1].split('#')[0].replace(/^\.\//, '');
    const abs = path.resolve(path.dirname(fromFile), rel);
    if (fs.existsSync(abs)) set.add(abs);
  }
  return [...set];
}

function escapeTs(str) {
  return String(str)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r?\n/g, ' ');
}

function build() {
  const files = walkMdFiles(LLM_DIR).sort();
  const nodes = [];
  const fileToId = new Map();

  for (const file of files) {
    const rel = path.relative(LLM_DIR, file);
    const base = path.basename(file, '.md');
    const id = relToId(rel);
    const raw = fs.readFileSync(file, 'utf8');
    const title = parseTitle(raw);
    const tips = parseTips(raw);
    let label = SPECIAL[base]?.label ?? defaultLabelFromBase(base);
    if (title) {
      if (title.length <= 56) label = title.replace(/\s+/g, ' ').trim();
      else {
        const short = title.split(/[？?]/)[0].trim();
        label = short.slice(0, 52) || label;
      }
    }
    const group = classifyGroup(rel, base, title);
    let tier = 2;
    if (rel.startsWith(`research${path.sep}`) || /vs|comparison|differences/i.test(base)) tier = 1;
    if (HUB_IDS.has(id)) tier = 3;

    nodes.push({ id, label, group, tier, tips });
    fileToId.set(file, id);
  }

  const idSet = new Set(nodes.map((n) => n.id));
  if (idSet.size !== nodes.length) {
    const cnt = new Map();
    for (const n of nodes) cnt.set(n.id, (cnt.get(n.id) ?? 0) + 1);
    for (const [k, v] of cnt) if (v > 1) console.warn('duplicate id:', k, v);
  }

  const links = [];
  const seen = new Set();
  function addLink(a, b) {
    if (!a || !b || a === b) return;
    if (!idSet.has(a) || !idSet.has(b)) return;
    const [x, y] = a < b ? [a, b] : [b, a];
    const key = `${x}\0${y}`;
    if (seen.has(key)) return;
    seen.add(key);
    links.push({ source: x, target: y });
  }

  for (const file of files) {
    const fromId = fileToId.get(file);
    const raw = fs.readFileSync(file, 'utf8');
    for (const abs of extractMdRefs(raw, file)) {
      const toId = fileToId.get(abs);
      if (toId) addLink(fromId, toId);
    }
  }

  const deg = new Map();
  for (const l of links) {
    deg.set(l.source, (deg.get(l.source) ?? 0) + 1);
    deg.set(l.target, (deg.get(l.target) ?? 0) + 1);
  }

  for (const n of nodes) {
    const hub = GROUP_SPOKE_HUB[n.group];
    if (hub && hub !== n.id && idSet.has(hub)) addLink(n.id, hub);
  }

  for (const n of nodes) {
    if (n.tier === 2 && (deg.get(n.id) ?? 0) >= 12) n.tier = 3;
  }

  const nodeBlocks = nodes
    .map(
      (n) => `  {
    id: '${escapeTs(n.id)}',
    label: '${escapeTs(n.label)}',
    group: '${n.group}',
    tier: ${n.tier} as 1 | 2 | 3,
    tips: [${n.tips.map((t) => `'${escapeTs(t)}'`).join(', ')}],
  }`,
    )
    .join(',\n');

  const linkLines = links
    .map((l) => `  { source: '${l.source}', target: '${l.target}' },`)
    .join('\n');

  const out = `/**
 * 从仓库 llm 目录全部 Markdown 自动生成（见 scripts/build-llm-graph-data.mjs）
 * 更新后执行: cd frontend && node scripts/build-llm-graph-data.mjs
 */
import type { BaguLink, BaguNode } from './llmGraphTypes';

export const corpusNodes: BaguNode[] = [
${nodeBlocks}
];

export const corpusLinks: BaguLink[] = [
${linkLines}
];
`;

  fs.writeFileSync(OUT_FILE, out, 'utf8');
  console.log(
    `OK: ${nodes.length} nodes, ${links.length} edges -> ${path.relative(FRONTEND, OUT_FILE)}`,
  );
}

build();
