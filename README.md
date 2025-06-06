# flyllm: LLM 从基础到深入

## 标签说明 Tag Introduction

| Tag | Items |
|:-----|:-----|
| 强化学习 | RLHF(基于人类反馈的强化学习), PPO(近端策略优化), DPO(直接偏好优化), GRPO(群体相对策略优化), 奖励模型训练, 人类反馈数据收集, 偏好对齐, 策略梯度, 价值函数, 优势估计 |
| 模型结构 | Transformer(自注意力机制), 混合专家模型(MoE), 稀疏注意力, 长上下文处理, 模型架构设计, 参数规模优化, 模型并行, 张量并行, 流水线并行 |
| 位置编码 | RoPE(旋转位置编码), ALiBi(注意力线性偏置), 相对位置编码, 旋转位置编码, 位置编码优化, 长序列处理, 位置插值, 位置外推 |
| 注意力机制 | 多头注意力, 稀疏注意力, 局部注意力, 全局注意力, 注意力计算优化, 注意力模式设计, 注意力掩码, 注意力头剪枝 |
| 数据处理 | 数据清洗, 数据增强, 数据质量评估, 数据去重, 数据预处理, 数据标注, 数据筛选, 数据平衡, 数据隐私, 数据安全 |
| 训练微调 | 预训练, 指令微调, 参数高效微调(PEFT), LoRA(低秩适应), QLoRA(量化低秩适应), 持续预训练, 领域适应, 多任务学习, 迁移学习 |
| 量化压缩 | 量化技术(INT8/INT4), 知识蒸馏, 模型剪枝, 模型压缩, 混合精度训练, 模型量化优化, 量化感知训练, 量化校准 |
| 多模态 | 视觉语言模型, 跨模态对齐, 多模态预训练, 图像理解, 视频理解, 多模态融合, 跨模态迁移, 多模态推理, 多模态生成 |
| 推理优化 | KV缓存, 批处理, 并行计算, 推理加速, 模型部署, 推理框架优化, 硬件加速, 推理延迟优化, 推理吞吐量优化 |
| 评估基准 | 基准测试, 人类评估, 自动评估, 安全性评估, 性能评估, 质量评估, 效率评估, 鲁棒性评估, 公平性评估 |
| 应用部署 | 模型服务, 推理框架, 部署优化, 成本控制, 服务架构, 负载均衡, 监控告警, 服务扩展, 服务高可用 |
| 智能体 | Agent(智能体), 智能体架构, 智能体行为, 智能体交互, 智能体学习, 智能体规划, 智能体决策, 智能体协作 |

| Tag | Title | Arxiv | Github | QA |
|:-----|:-----|:-----|:-----|:-----|
| 强化学习 | GRPO | [arxiv](https://arxiv.org/pdf/2402.03300) | [github](https://github.com/deepseek-ai/DeepSeek-Math) | [qa](https://github.com/Decalogue/flyllm/blob/main/qa/强化学习.csv) |
| 量化压缩 | DFloat11 | [arxiv](https://arxiv.org/abs/2504.11651) | [github](https://github.com/LeanModels/DFloat11) | [qa](https://github.com/Decalogue/flyllm/blob/main/qa/量化压缩.csv) |
| 多模态 | BLIP3-o | [arxiv](https://arxiv.org/abs/2505.09568) | [github](https://github.com/JiuhaiChen/BLIP3o) | [qa](https://github.com/Decalogue/flyllm/blob/main/qa/多模态.csv) |
|  |  |  |  |  |