/**
 * 创作助手主题 — 参考 Kimi 2.5 Agent 并行蜂群模式
 * 深色指挥中心、品牌红橙 #ff4b2f、玻璃质感、游戏化状态与记忆网络
 */
export const CREATOR_THEME = {
  /** 字体 */
  fontFamily: '"Inter", "Public Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  /** 字重 */
  fontWeightMedium: 500,
  fontWeightSemibold: 600,
  fontWeightBold: 700,

  /** 品牌色（Kimi 蜂群） */
  brand: '#ff4b2f',
  brandLight: '#ffa7a7',
  brandLighter: '#ffb29e',

  /** 页面背景：深色渐变 */
  bgPage:
    'linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 35%, #1a1a2e 70%, #0a0a0a 100%)',
  /** 画布 */
  bgCanvas: 'rgba(15, 15, 26, 0.5)',
  /** 侧栏：深色玻璃 */
  bgSidebar: 'rgba(15, 15, 26, 0.85)',
  sidebarBlur: 'blur(20px)',
  /** 顶栏 */
  bgHeader: 'rgba(10, 10, 16, 0.9)',
  headerBlur: 'blur(20px)',

  /** 边框 */
  border: 'rgba(255, 255, 255, 0.08)',
  borderStrong: 'rgba(255, 255, 255, 0.12)',
  borderHeader: 'rgba(255, 255, 255, 0.06)',

  /** 文字 */
  text: '#e4e4e7',
  textBright: '#fafafa',
  textMuted: 'rgba(255, 255, 255, 0.5)',
  textDim: 'rgba(255, 255, 255, 0.35)',

  /** 分段器 */
  segBg: 'rgba(255, 255, 255, 0.06)',
  segSelectedBg: 'rgba(255, 75, 47, 0.2)',
  segSelectedText: '#ff4b2f',
  segUnselectedText: 'rgba(255, 255, 255, 0.5)',

  /** 主色 / 强调（与 brand 统一） */
  accent: '#ff4b2f',
  accentDim: 'rgba(255, 75, 47, 0.15)',
  accentBorder: 'rgba(255, 75, 47, 0.35)',

  /** Ghost 按钮 */
  ghostBorder: 'rgba(255, 255, 255, 0.12)',
  ghostText: '#e4e4e7',
  ghostHoverBorder: 'rgba(255, 75, 47, 0.4)',
  ghostHoverBg: 'rgba(255, 75, 47, 0.08)',

  /** 主按钮 */
  primaryBg: '#ff4b2f',
  primaryHover: '#e5432a',

  /** 输入区 */
  bgInput: 'rgba(15, 15, 26, 0.9)',

  /** 用户消息 */
  bgMsgUser: 'rgba(255, 75, 47, 0.1)',
  borderMsgUser: 'rgba(255, 75, 47, 0.25)',
  textMsgUser: '#fef2f2',

  /** 助手消息 */
  bgMsgBot: 'rgba(26, 26, 46, 0.88)',
  borderMsgBot: 'rgba(255, 255, 255, 0.06)',

  /** 头像 */
  avatarUser: 'linear-gradient(135deg, #ff4b2f, #ffa7a7)',
  avatarBot: 'linear-gradient(135deg, #ff4b2f, #ffb29e)',

  /** 空状态 */
  emptyIconBg: 'linear-gradient(135deg, #ff4b2f 0%, #ffa7a7 100%)',
  emptyIconGlow: 'rgba(255, 75, 47, 0.3)',

  /** 图谱 / 抽屉 */
  bgGraph: 'rgba(10, 10, 16, 0.6)',
  bgGraphSolid: '#0f0f1a',
  bgDrawer: 'rgba(26, 26, 46, 0.98)',
  bgDrawerHeader: 'rgba(15, 15, 26, 0.95)',

  /** 工作流（Agent 状态） */
  flowDoneBg: 'rgba(34, 197, 94, 0.12)',
  flowDoneBorder: 'rgba(34, 197, 94, 0.35)',
  flowDoneIcon: '#22c55e',
  flowPendingIcon: 'rgba(255, 255, 255, 0.4)',
  flowPendingBg: 'rgba(255, 255, 255, 0.04)',
  flowRunningBg: 'rgba(255, 75, 47, 0.1)',
  flowRunningBorder: 'rgba(255, 75, 47, 0.4)',

  /** 检索卡片 */
  bgRecall: 'rgba(255, 75, 47, 0.08)',

  /** 圆角 */
  radiusSm: 8,
  radiusMd: 12,
  radiusLg: 16,
  radiusXl: 20,

  /** 阴影 */
  shadowPanel: '0 0 0 1px rgba(255,255,255,0.06), 0 4px 24px rgba(0,0,0,0.5)',
  shadowCard: '0 0 0 1px rgba(255,255,255,0.06), 0 2px 12px rgba(0,0,0,0.3)',
  glowBrand: '0 0 40px rgba(255, 75, 47, 0.2)',
} as const;

export type CreatorTheme = typeof CREATOR_THEME;

/**
 * 主页亮色主题 — 参考 Kimi 参考 app Hero（白底、深色字、品牌色）
 */
export const INTRO_THEME = {
  fontFamily: CREATOR_THEME.fontFamily,
  fontWeightMedium: 500,
  fontWeightSemibold: 600,
  fontWeightBold: 700,

  bgPage: '#ffffff',
  bgCard: 'rgba(255,255,255,0.9)',
  bgCardHover: 'rgba(243,243,243,0.95)',
  bgBadge: 'rgba(255,255,255,0.9)',
  bgCta: 'rgba(255,255,255,0.95)',

  text: '#0f0f0f',
  textBright: '#0f0f0f',
  textMuted: '#4a4a49',
  textDim: '#6b7280',

  border: 'rgba(0,0,0,0.08)',
  borderStrong: '#e5e5e5',
  stroke: '#e5e5e5',

  accent: '#ff4b2f',
  accentDim: 'rgba(255, 75, 47, 0.1)',
  accentBorder: 'rgba(255, 75, 47, 0.3)',

  primaryBg: '#ff4b2f',
  primaryHover: '#e5432a',

  radiusSm: 8,
  radiusMd: 12,
  radiusLg: 16,
  radiusXl: 20,

  shadowCard: '0 0 0 1px rgba(0,0,0,0.04), 0 2px 12px rgba(0,0,0,0.06)',
  shadowCta: '0 0 0 1px rgba(255,75,47,0.2), 0 4px 24px rgba(255,75,47,0.15)',
} as const;

export type IntroTheme = typeof INTRO_THEME;
