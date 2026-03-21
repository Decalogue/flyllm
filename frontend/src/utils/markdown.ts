/**
 * 共享 Markdown 渲染器 — 用于 AI 对话、创作助手等
 */
import MarkdownIt from 'markdown-it';

export const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  breaks: true,
});
