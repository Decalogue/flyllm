# 全量微调和参数高效微调的区别是什么？分别适用于什么场景？

## 1. 核心定性
本质上，**全量微调**是更新预训练模型所有参数（$\theta \in \mathbb{R}^{d_{\text{total}}}$）的密集优化，**参数高效微调**则通过冻结主干仅更新少量增量参数（$\Delta\theta \ll \theta$）实现任务适配，在保持 95% 性能的同时将训练成本降低 90%。

## 2. 具体流程
1. **全量微调**：加载预训练权重 $\theta_{\text{pretrained}}$，解冻全部参数，在下游数据上执行梯度下降更新所有 $d_{\text{total}}$ 参数，每个权重矩阵都参与反向传播
2. **参数高效微调**：冻结 $\theta_{\text{pretrained}}$，仅训练少量新增模块（Adapter、LoRA 矩阵、Prompt Embedding），梯度仅流过增量参数，主干参数不变
3. **场景选择**：全量微调用于领域迁移（医疗、法律）需深度适配；参数高效微调用于快速迭代、多租户服务、资源受限设备部署

## 3. 数学基础
**全量微调目标函数**：
$$\mathcal{L}_{\text{full}}(\theta) = \mathbb{E}_{(x,y)\sim D_{\text{task}}}[-\log p_\theta(y|x)]$$

参数更新：
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{full}}(\theta_t), \quad \theta \in \mathbb{R}^{d_{\text{total}}}$$

**LoRA（参数高效微调代表）**：
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}BAx$$

其中：
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：冻结的预训练权重
- $B \in \mathbb{R}^{d_{\text{out}} \times r}, A \in \mathbb{R}^{r \times d_{\text{in}}}$：可训练的低秩矩阵
- $r \ll \min(d_{\text{out}}, d_{\text{in}})$：秩（通常 4-64）
- $\alpha$：缩放系数（通常 $r$ 或 $2r$）

**参数量对比**：
$$d_{\text{LoRA}} = r(d_{\text{out}} + d_{\text{in}}) \ll d_{\text{total}} = \sum_{l=1}^L d_{\text{out}}^{(l)} d_{\text{in}}^{(l)}$$

LLaMA 7B：$d_{\text{total}} = 6.7 \times 10^9$，LoRA $r=16$ 时 $d_{\text{LoRA}} = 2.5 \times 10^7$（0.37%）

## 4. 工程考量
**Trade-off**:
- **全量微调**：以 10-100 倍计算成本换取 2-5% 的最终性能提升，需 8xA100-80GB 训练 7B 模型，适配周期长（天级），但领域迁移能力最强
- **参数高效微调**：牺牲少量性能（通常 1-3%）换取 10-100 倍速度提升和 100-1000 倍存储效率，单卡 24GB 可训 70B 模型，适配周期分钟级，适合多任务并行

**致命弱点**:
- **全量微调**：灾难性遗忘严重，每训练一个新任务需完整副本，存储成本 $O(k \cdot d_{\text{total}})$（$k$ 任务数），跨任务干扰导致性能下降 5-10%
- **参数高效微调**：
  - **秩瓶颈**：低秩近似无法捕捉任务特定的高秩模式，在需要大幅偏离预训练分布的任务（如代码→诗歌）上效果差 10-20%
  - **模块冲突**：多个 LoRA 模块同时激活时，增量参数叠加导致分布漂移，在多任务批处理中性能下降 3-5%
  - **推理延迟**：LoRA 合并需在推理时加载 $W_0 + BA$，若动态切换任务，内存拷贝开销使延迟增加 50-100ms，不适合低延迟场景

## 5. 工业映射
在工业界，微调策略是**LLM 服务化的核心决策点**：
- **全量微调（OpenAI GPT-4）**：在代码专用模型（Codex）训练时全量微调，消耗 10,000+ A100 小时，性能登顶 HumanEval（85%），但仅服务单一场景，成本高但收益明确
- **LoRA（Hugging Face PEFT）**：成为多租户 LLM 服务的标准，Mistral 7B 部署时每个租户加载独立 LoRA adapter（50MB），单 GPU 服务 1000+ 定制化模型，显存占用仅增加 5%，吞吐量保持 95%
- **QLoRA（bitsandbytes）**：在 LLaMA 65B 微调中，NF4 量化 + LoRA 使显存从 780GB 降至 48GB（单卡），在 Vicuna 评估中保持 97% 性能，成为开源社区微调标配
- **多任务服务（S-LoRA）**：NVIDIA Triton Inference Server 支持 1000+ LoRA 模块动态切换，KV-cache 共享，在客服场景中每个客户加载专属微调模块，首 token 延迟仅增加 10%
- **边缘设备（Llama.cpp）**：手机端部署使用 LoRA 微调模块，基础模型 4GB + 每个 LoRA 20MB，iPhone 15 Pro 可同时加载 50+ 领域适配模型，实现离线个性化 AI

**趋势**：现代 LLM 服务采用"**预训练模型 + 海量 LoRA 适配器**"的架构，全量微调仅在基础模型训练阶段使用，微调层完全参数高效化，形成"训练密集、推理稀疏"的工业化范式。
