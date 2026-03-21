# 创想AI 前端

基于 Umi 4 + React + Ant Design 的 AI 创作助手前端应用。

## 页面结构

| 路径 | 组件 | 说明 |
|------|------|------|
| `/` | `home/intro` | 主页：Hero、工作流、记忆系统（默认 3D 图谱）、CTA |
| `/creator` | `home/creator` | 创作助手主界面：模式（大纲/章节/润色/对话）、指挥中心、章节列表与单章全文 |
| `/ai` | `home/ai` | AI 对话 |

## 主要组件

- **IntroFooter**：主页页脚，含品牌、联系方式、链接、订阅区
- **WorkflowGraph**：工作流编排可视化。`variant="creation"` 展示创作流程（构思→记忆召回(跨章人物、伏笔、长线设定)→续写→质检⇄重写→实体提取→记忆入库），`variant="research"` 展示通用研究流程；箭头向右，召回分三路到三模块再汇聚到续写
- **MemoryGraphThree**：3D 记忆图谱；图例按类型上色，节点按类型区分几何（实体八面体、事实扁圆柱、原子笔记球体）；intro 模式下节点下展示 brief/source
- **MemoryGraphD3**：2D 力导向图谱；初始布局按圆分布，避免节点挤在一起再弹开
- **OrchestrationFlow**：编排流程展示，与指挥中心步骤一致（构思、记忆召回、续写、质检、润色）

## 创作页（/creator）

- **模式**：顶部与输入区下方均为 Segmented（大纲 / 章节 / 润色 / 对话）
- **章节列表**：Modal「点击查看完整章节列表」中，已写章节可点击；请求 `GET /api/creator/chapter?project_id=&number=` 成功后用 Drawer 展示全文；404 时提示确认后端已支持该接口并重启
- **对话**：请求体 `model: 'kimi-k2-5'`，系统提示设定身份为创作助手/Kimi，避免模型自称 Claude

## 多语言（i18n）

- **语言**：中文（zh-CN）、English（en-US），通过右上角/顶栏 **LocaleSwitcher** 切换。
- **持久化**：当前语言存 `localStorage`（`creator_locale`），刷新后保持。
- **文案**：`src/locales/zh-CN.json`、`src/locales/en-US.json`；`src/utils/i18n.ts` 初始化 react-i18next；各页使用 `useTranslation()` 与 `t('intro.xxx')` / `t('creator.xxx')` / `t('ai.xxx')`。

## 主题

- `INTRO_THEME`：主页亮色主题（白底、品牌红橙）
- `CREATOR_THEME`：创作助手深色主题（指挥中心风格）

## 联系方式（页脚）

- 邮箱：decalogue80@gmail.com
- 电话：13661445290
- 地址：上海市

## 开发

```bash
pnpm install
pnpm dev
```

## 构建

```bash
pnpm build
```
