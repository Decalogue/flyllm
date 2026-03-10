# 手撕对比学习loss

## 📌 面试核心回答框架

### 💡 30秒快速回答

> **核心观点：** InfoNCE 是对比学习的核心损失函数，通过最大化正样本对的互信息、最小化负样本对的相似度，实现自监督表示学习。关键在于：**①温度参数控制分布尖锐度 ②负样本数量影响性能 ③与互信息下界的关系**。

---

## 一、理论基础：从 NCE 到 InfoNCE

### 1.1 信息论基础：互信息（Mutual Information）

InfoNCE 的核心目标是最大化正样本对之间的互信息：

```
I(x; y) = H(x) - H(x|y) = Σ p(x,y) log(p(x|y) / p(x))
```

#### 公式中各变量的详细解释

**1. I(x; y) - 互信息（Mutual Information）**
- **定义**：衡量两个随机变量 x 和 y 之间的相互依赖程度
- **单位**：比特（bits）或纳特（nats）
- **取值范围**：I(x; y) ≥ 0
  - I(x; y) = 0：x 和 y 相互独立
  - I(x; y) > 0：x 和 y 相关，值越大相关性越强
- **对称性**：I(x; y) = I(y; x)

**2. H(x) - 熵（Entropy）**
- **定义**：随机变量 x 的不确定性（信息量）
- **公式**：H(x) = -Σ p(x) log p(x)
- **直观理解**：
  - 熵越大，x 的不确定性越大，信息量越多
  - 熵越小，x 越确定，信息量越少
- **例子**：
  - 抛硬币：H(公平硬币) = 1 bit，H(总是正面) = 0 bit

**3. H(x|y) - 条件熵（Conditional Entropy）**
- **定义**：在已知 y 的条件下，x 的不确定性
- **公式**：H(x|y) = -Σ p(x,y) log p(x|y)
- **直观理解**：
  - H(x|y) 表示"知道 y 后，x 还剩下多少不确定性"
  - 如果 y 完全决定 x，则 H(x|y) = 0
  - 如果 y 和 x 无关，则 H(x|y) = H(x)

**4. p(x, y) - 联合概率（Joint Probability）**
- **定义**：x 和 y 同时发生的概率
- **例子**：p(下雨, 带伞) = 0.3 表示"下雨且带伞"的概率是 30%

**5. p(x|y) - 条件概率（Conditional Probability）**
- **定义**：在 y 发生的条件下，x 发生的概率
- **公式**：p(x|y) = p(x,y) / p(y)
- **例子**：p(带伞|下雨) = 0.8 表示"如果下雨，带伞的概率是 80%"

**6. p(x) - 边缘概率（Marginal Probability）**
- **定义**：x 发生的概率（不考虑 y）
- **公式**：p(x) = Σ_y p(x,y)
- **例子**：p(带伞) = 0.5 表示"带伞的概率是 50%"

#### 互信息的三种等价形式

**形式1：熵的差**
```
I(x; y) = H(x) - H(x|y)
```
- **含义**：x 的不确定性 - 知道 y 后 x 的不确定性 = x 和 y 的互信息
- **直观理解**：y 提供了多少关于 x 的信息

**形式2：对称形式**
```
I(x; y) = H(x) + H(y) - H(x, y)
```
- **含义**：x 的信息 + y 的信息 - 联合信息 = 互信息
- **直观理解**：互信息是"共享的信息"

**形式3：KL 散度形式（最常用）**
```
I(x; y) = Σ p(x,y) log(p(x|y) / p(x))
        = Σ p(x,y) log(p(x,y) / (p(x) p(y)))
        = KL(p(x,y) || p(x) p(y))
```
- **含义**：联合分布与独立分布的 KL 散度
- **直观理解**：衡量"实际分布"与"假设独立"的差异

#### 具体例子

**例子1：完全相关**
```
x = y（完全相关）
- p(x=0, y=0) = 0.5, p(x=1, y=1) = 0.5
- p(x|y) = 1（如果 y=0，则 x 一定是 0）
- H(x|y) = 0（知道 y 后，x 完全确定）
- I(x; y) = H(x) - 0 = H(x) = 1 bit（最大互信息）
```

**例子2：完全独立**
```
x 和 y 独立
- p(x,y) = p(x) p(y)
- p(x|y) = p(x)（知道 y 不影响 x 的分布）
- H(x|y) = H(x)（条件熵等于无条件熵）
- I(x; y) = H(x) - H(x) = 0（无互信息）
```

**例子3：部分相关**
```
x 是"天气"（晴/雨），y 是"带伞"（是/否）
- p(晴, 带伞) = 0.2
- p(晴, 不带伞) = 0.5
- p(雨, 带伞) = 0.3
- p(雨, 不带伞) = 0.0

计算：
- H(x) = -0.7*log(0.7) - 0.3*log(0.3) ≈ 0.88 bit
- H(x|y=带伞) = -0.4*log(0.4) - 0.6*log(0.6) ≈ 0.97 bit
- H(x|y=不带伞) = -1*log(1) - 0*log(0) = 0 bit
- H(x|y) = 0.5*0.97 + 0.5*0 = 0.485 bit
- I(x; y) = 0.88 - 0.485 = 0.395 bit
```

#### 在对比学习中的应用

在对比学习中：
- **x**：原始样本的表示 z
- **y**：正样本的表示 z^+
- **目标**：最大化 I(z; z^+)

**为什么最大化互信息？**
- I(z; z^+) 大 → z 和 z^+ 高度相关
- 说明模型学习到了有意义的表示
- 正样本对（同一图像的不同增强）应该相似

**互信息的直观理解：**
- I(z; z^+) = 0：z 和 z^+ 完全无关（学习失败）
- I(z; z^+) 大：z 和 z^+ 高度相关（学习成功）
- 通过最大化互信息，模型学习到"语义相似"的表示

#### 互信息的可视化理解

```
信息论视角（韦恩图）：

    ┌─────────────────┐
    │   H(x)          │
    │  ┌──────────┐   │
    │  │ I(x;y)   │   │  H(y)
    │  │          │   │
    │  └──────────┘   │
    │   H(x|y)        │
    └─────────────────┘

I(x; y) = H(x) - H(x|y)
        = H(y) - H(y|x)
        = H(x) + H(y) - H(x,y)
```

**关键关系：**
- **H(x)**：x 的总信息量（大圆）
- **H(x|y)**：知道 y 后，x 还剩下的不确定性（x 圆中不重叠部分）
- **I(x; y)**：x 和 y 共享的信息（重叠部分）
- **H(x, y)**：x 和 y 的联合信息量（两个圆的并集）

**在对比学习中的对应：**
- H(z)：原始表示 z 的信息量
- H(z|z^+)：知道正样本 z^+ 后，z 还剩下的不确定性
- I(z; z^+)：z 和 z^+ 共享的语义信息（我们想最大化的部分）

### 1.2 Noise Contrastive Estimation (NCE)

NCE 的核心思想：**将密度估计问题转化为二分类问题**

```
原始问题：估计 p(x)（困难）
转化问题：区分真实样本 x 和噪声样本 x~（简单）
```

NCE 损失：
```
L_NCE = -log(σ(f(x) - log(p_n(x)))) - Σ log(1 - σ(f(x~) - log(p_n(x~))))
```

其中：
- `f(x)` 是模型输出的分数
- `p_n(x)` 是噪声分布
- `σ` 是 sigmoid 函数

### 1.3 InfoNCE：从 NCE 到对比学习

InfoNCE 将 NCE 扩展到多分类问题：

```
L_InfoNCE = -log(exp(sim(z, z^+) / τ) / Σ_{j=1}^{N} exp(sim(z, z_j) / τ))
```

**关键改进：**
1. **多负样本**：从二分类扩展到 N 分类（1 正 + N-1 负）
2. **温度参数 τ**：控制分布的尖锐程度
3. **互信息下界**：InfoNCE 是互信息的下界（证明见下文）

### 1.4 互信息下界证明（面试重点）

**定理：** InfoNCE 是互信息 I(z; z^+) 的下界

**证明思路：**

```
L_InfoNCE = -E[log(exp(sim(z, z^+) / τ) / Σ_j exp(sim(z, z_j) / τ))]

展开后：
= -E[sim(z, z^+) / τ] + E[log Σ_j exp(sim(z, z_j) / τ)]

根据 Jensen 不等式：
≥ -E[sim(z, z^+) / τ] + log E[Σ_j exp(sim(z, z_j) / τ)]

当 f(z, z^+) = sim(z, z^+) / τ 是互信息的评分函数时：
≥ -I(z; z^+) + log N

因此：
I(z; z^+) ≥ log N - L_InfoNCE
```

**面试要点：**
- InfoNCE 是互信息的**下界**，不是精确值
- 负样本数量 N 越大，下界越紧（tight）
- 这是为什么需要大量负样本的理论依据

---

## 二、数学公式详解

### 2.1 标准公式

```
L_InfoNCE = -log(exp(sim(z_i, z_i^+) / τ) / Σ_{j=1}^{N} exp(sim(z_i, z_j) / τ))
```

**符号说明：**
- `z_i`：锚点样本（anchor）的表示向量 [dim]
- `z_i^+`：正样本（positive）的表示向量 [dim]
- `z_j`：负样本（negative）的表示向量 [dim]，j ∈ {1, 2, ..., N-1}
- `sim(·, ·)`：相似度函数，通常是归一化后的点积（余弦相似度）
- `τ`：温度参数（temperature），控制分布的尖锐程度
- `N`：总样本数（1 正 + N-1 负）

### 2.2 相似度函数的选择

**1. 余弦相似度（最常用）**
```
sim(z_i, z_j) = z_i^T z_j / (||z_i|| ||z_j||) = z_i^T z_j  (归一化后)
```

**2. 点积（需归一化）**
```
sim(z_i, z_j) = z_i^T z_j  (要求 ||z_i|| = ||z_j|| = 1)
```

**3. 欧氏距离（较少用）**
```
sim(z_i, z_j) = -||z_i - z_j||^2
```

**为什么选择余弦相似度？**
- 归一化后，相似度在 [-1, 1] 范围内，数值稳定
- 不受向量长度影响，只关注方向
- 计算高效（点积操作）

### 2.3 温度参数 τ 的数学意义

温度参数控制 softmax 分布的熵：

```
P(z^+|z) = exp(sim(z, z^+) / τ) / Σ_j exp(sim(z, z_j) / τ)
```

**τ 的影响：**
- **τ → 0**：分布趋于 one-hot，只关注最相似的样本
- **τ → ∞**：分布趋于均匀，所有样本权重相等
- **τ = 1**：标准 softmax

**经验值：**
- SimCLR: τ = 0.07
- MoCo: τ = 0.07
- CLIP: τ = 0.01 (可学习)

**面试问题：为什么 τ 这么小？**
- 小温度使模型更关注**困难负样本**（hard negatives）
- 提高表示学习的判别能力
- 但过小会导致训练不稳定

### 2.4 批次内负样本（In-batch Negatives）

在实际实现中，通常使用批次内其他样本作为负样本：

```
对于批次大小 batch_size = 2N：
- 每个样本有 1 个正样本（另一个增强版本）
- 每个样本有 2N-2 个负样本（批次内其他样本）
```

**优势：**
- 无需额外存储负样本
- 计算高效（一次矩阵乘法）
- 负样本多样性好

**劣势：**
- 负样本数量受批次大小限制
- 可能存在假负样本（false negatives）

---

## 三、完整实现（生产级代码）

### 3.1 标准实现：SimCLR 风格（推荐）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCE(nn.Module):
    """
    InfoNCE 对比学习损失函数（生产级实现）
    
    适用于 SimCLR 场景：输入 [2*N, dim]，每两个连续样本是正样本对
    例如：[z1, z1', z2, z2', ...] 其中 (z1, z1') 是正样本对
    
    Args:
        temperature: 温度参数，默认 0.07
        reduction: 损失聚合方式，'mean' 或 'sum'
    
    Reference:
        - SimCLR: https://arxiv.org/abs/2002.05709
        - CPC: https://arxiv.org/abs/1807.03748
    """
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        assert temperature > 0, "Temperature must be positive"
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'"
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE 损失
        
        Args:
            features: [2*N, dim] 特征向量，每两个连续样本是正样本对
                    例如：[z1, z1', z2, z2', ...]
        
        Returns:
            loss: InfoNCE 损失值（标量或 [2*N]）
        
        Shape:
            - Input: [2*N, dim]
            - Output: scalar (if reduction='mean') or [2*N] (if reduction='none')
        """
        batch_size, dim = features.shape
        assert batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
        
        device = features.device
        n = batch_size // 2
        
        # L2 归一化（关键步骤）
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # 计算相似度矩阵 [2*N, 2*N]
        # similarity_matrix[i, j] = features[i]^T @ features[j]
        similarity_matrix = torch.matmul(features, features.t())  # [2*N, 2*N]
        
        # 创建正样本对掩码
        # 正样本对：索引 (0,1), (2,3), (4,5), ...
        # mask[i, j] = True 表示 (i, j) 是正样本对
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        for i in range(0, batch_size, 2):
            mask[i, i+1] = True
            mask[i+1, i] = True
        
        # 提取正样本相似度 [2*N]
        pos_sim = similarity_matrix[mask].unsqueeze(1)  # [2*N, 1]
        
        # 创建负样本掩码（排除自己和正样本）
        neg_mask = ~mask
        neg_mask.fill_diagonal_(False)  # 排除自己
        
        # 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 构建 logits：[正样本相似度, 负样本相似度1, 负样本相似度2, ...]
        # 方法：将正样本位置和对角线设为 -inf，然后提取负样本相似度
        logits = similarity_matrix.clone()
        logits[mask] = float('-inf')  # 排除正样本位置
        logits.fill_diagonal_(float('-inf'))  # 排除自己
        
        # 提取负样本相似度 [2*N, 2*N-2]
        neg_sim = logits[neg_mask].reshape(batch_size, -1)
        
        # 合并正负样本相似度 [2*N, 2*N-1]
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 标签：第一个位置（索引 0）是正样本
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 计算交叉熵损失（等价于 InfoNCE）
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss
    
    def extra_repr(self) -> str:
        return f'temperature={self.temperature}, reduction={self.reduction}'
```

### 3.2 高效实现：避免完整相似度矩阵

```python
class InfoNCEOptimized(nn.Module):
    """
    内存优化的 InfoNCE 实现
    适用于大批次场景，避免构建完整的 [2*N, 2*N] 相似度矩阵
    
    关键优化：
    1. 分离正样本对，分别计算
    2. 使用分批计算减少内存占用
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [2*N, dim] 特征向量
        
        Returns:
            loss: InfoNCE 损失值
        """
        batch_size, dim = features.shape
        assert batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
        
        device = features.device
        n = batch_size // 2
        
        # L2 归一化
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # 分离正样本对
        z1 = features[0::2]  # [N, dim] 第一个增强版本
        z2 = features[1::2]  # [N, dim] 第二个增强版本
        
        # 计算正样本相似度 [N]
        pos_sim = torch.sum(z1 * z2, dim=1, keepdim=True) / self.temperature  # [N, 1]
        
        # 计算 z1 与所有 features 的相似度 [N, 2*N]
        all_sim_z1 = torch.matmul(z1, features.t()) / self.temperature
        # 计算 z2 与所有 features 的相似度 [N, 2*N]
        all_sim_z2 = torch.matmul(z2, features.t()) / self.temperature
        
        # 创建掩码：排除自己和对应的正样本
        # 对于 z1[i]，排除 z1[i]（索引 2*i）和 z2[i]（索引 2*i+1，正样本）
        mask_z1 = torch.zeros(n, 2*n, dtype=torch.bool, device=device)
        mask_z2 = torch.zeros(n, 2*n, dtype=torch.bool, device=device)
        
        for i in range(n):
            mask_z1[i, 2*i] = True      # 排除 z1[i]（自己）
            mask_z1[i, 2*i+1] = True    # 排除 z2[i]（正样本）
            mask_z2[i, 2*i] = True      # 排除 z1[i]（正样本）
            mask_z2[i, 2*i+1] = True    # 排除 z2[i]（自己）
        
        # 应用掩码
        all_sim_z1 = all_sim_z1.masked_fill(mask_z1, float('-inf'))
        all_sim_z2 = all_sim_z2.masked_fill(mask_z2, float('-inf'))
        
        # 构建 logits：正样本相似度 + 负样本相似度
        logits_z1 = torch.cat([pos_sim, all_sim_z1], dim=1)  # [N, 2*N]
        logits_z2 = torch.cat([pos_sim, all_sim_z2], dim=1)  # [N, 2*N]
        
        # 合并所有 logits
        logits = torch.cat([logits_z1, logits_z2], dim=0)  # [2*N, 2*N]
        
        # 标签：第一个位置是正样本
        labels = torch.zeros(2*n, dtype=torch.long, device=device)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

### 3.3 对称 InfoNCE（双向损失）

```python
class SymmetricInfoNCE(nn.Module):
    """
    对称 InfoNCE：同时计算两个方向的损失
    L = (L(z1->z2) + L(z2->z1)) / 2
    
    优势：
    - 更对称的优化目标
    - 在某些任务上效果更好
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: [batch_size, dim] 两个增强版本的特征
        
        Returns:
            loss: 对称 InfoNCE 损失值
        """
        batch_size = z1.size(0)
        device = z1.device
        
        # 归一化
        z1 = F.normalize(z1, p=2, dim=1, eps=1e-8)
        z2 = F.normalize(z2, p=2, dim=1, eps=1e-8)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        sim_matrix = torch.matmul(z1, z2.t()) / self.temperature
        
        # 正样本对在对角线上
        labels = torch.arange(batch_size, device=device)
        
        # 两个方向的损失
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        
        # 对称损失
        loss = (loss_12 + loss_21) / 2
        
        return loss
```

### 3.4 数值稳定性优化

```python
class InfoNCENumericallyStable(nn.Module):
    """
    数值稳定的 InfoNCE 实现
    使用 log-sum-exp 技巧避免数值溢出
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, dim = features.shape
        assert batch_size % 2 == 0
        
        device = features.device
        n = batch_size // 2
        
        # 归一化
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # 分离正样本对
        z1 = features[0::2]  # [N, dim]
        z2 = features[1::2]  # [N, dim]
        
        # 计算所有相似度
        all_sim = torch.matmul(features, features.t()) / self.temperature  # [2*N, 2*N]
        
        # 正样本相似度（在对角线上，但需要调整索引）
        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature  # [N]
        
        # 对于每个样本，计算 log-sum-exp
        losses = []
        for i in range(n):
            # z1[i] 的损失
            pos = pos_sim[i]
            negs = []
            for j in range(2*n):
                if j != 2*i and j != 2*i+1:  # 排除自己和正样本
                    negs.append(all_sim[2*i, j])
            negs = torch.stack(negs)
            
            # 使用 log-sum-exp 技巧
            max_val = torch.max(torch.cat([pos.unsqueeze(0), negs]))
            log_sum_exp = max_val + torch.log(
                torch.exp(pos - max_val) + torch.sum(torch.exp(negs - max_val))
            )
            loss_i = log_sum_exp - pos
            losses.append(loss_i)
            
            # z2[i] 的损失（类似处理）
            pos = pos_sim[i]
            negs = []
            for j in range(2*n):
                if j != 2*i and j != 2*i+1:
                    negs.append(all_sim[2*i+1, j])
            negs = torch.stack(negs)
            
            max_val = torch.max(torch.cat([pos.unsqueeze(0), negs]))
            log_sum_exp = max_val + torch.log(
                torch.exp(pos - max_val) + torch.sum(torch.exp(negs - max_val))
            )
            loss_i = log_sum_exp - pos
            losses.append(loss_i)
        
        return torch.stack(losses).mean()
```

---

## 四、梯度分析（面试重点）

### 4.1 InfoNCE 的梯度公式

对正样本相似度 `s^+ = sim(z, z^+) / τ` 的梯度：

```
∂L/∂s^+ = -1 + P(z^+|z) = -(1 - P(z^+|z))
```

对负样本相似度 `s^- = sim(z, z^-) / τ` 的梯度：

```
∂L/∂s^- = P(z^-|z)
```

**直观理解：**
- 正样本梯度：**负值**，推动正样本相似度增加
- 负样本梯度：**正值**，推动负样本相似度减少
- 梯度大小与 softmax 概率成正比

### 4.2 温度参数对梯度的影响

```
当 τ 很小时：
- P(z^+|z) → 1（如果正样本最相似）
- ∂L/∂s^+ → 0（梯度消失）
- 但困难负样本的梯度很大

当 τ 很大时：
- P(z^+|z) → 1/N（均匀分布）
- 所有样本的梯度都较小
- 训练稳定但学习慢
```

**面试问题：如何选择温度参数？**
1. 从 0.1 开始，根据验证集调整
2. 观察训练曲线：损失是否下降、是否稳定
3. 考虑下游任务性能

### 4.3 负样本数量对梯度的影响

```
负样本数量 N 增加：
- 互信息下界更紧（理论优势）
- 但每个负样本的梯度变小（1/N）
- 需要权衡计算成本和性能
```

---

## 五、常见变体与改进

### 5.1 MoCo (Momentum Contrast)

```python
class MoCoLoss(nn.Module):
    """
    MoCo 使用动量更新的编码器和队列维护负样本
    
    关键创新：
    1. 动量编码器：key encoder 使用动量更新
    2. 队列机制：维护大量负样本（65536）
    3. 解耦批次大小和负样本数量
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: 查询特征 [batch_size, dim]，来自 query encoder
            k: 键特征 [batch_size, dim]，来自 momentum encoder（正样本）
            queue: 负样本队列 [queue_size, dim]
        """
        batch_size = q.size(0)
        device = q.device
        
        # 归一化
        q = F.normalize(q, p=2, dim=1, eps=1e-8)
        k = F.normalize(k, p=2, dim=1, eps=1e-8)
        queue = F.normalize(queue, p=2, dim=1, eps=1e-8)
        
        # 正样本相似度
        pos_sim = torch.sum(q * k, dim=1, keepdim=True) / self.temperature  # [batch_size, 1]
        
        # 负样本相似度
        neg_sim = torch.matmul(q, queue.t()) / self.temperature  # [batch_size, queue_size]
        
        # 合并
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 1 + queue_size]
        
        # 标签
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 损失
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

### 5.2 Hard Negative Mining

```python
class InfoNCEWithHardNegatives(nn.Module):
    """
    使用困难负样本的 InfoNCE
    只选择最相似的负样本（top-k）参与计算
    """
    def __init__(self, temperature: float = 0.07, top_k: int = 10):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, dim = features.shape
        assert batch_size % 2 == 0
        
        device = features.device
        n = batch_size // 2
        
        # 归一化
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # 分离正样本对
        z1 = features[0::2]
        z2 = features[1::2]
        
        # 正样本相似度
        pos_sim = torch.sum(z1 * z2, dim=1, keepdim=True) / self.temperature
        
        # 计算所有相似度
        all_sim = torch.matmul(z1, features.t()) / self.temperature  # [N, 2*N]
        
        # 创建掩码
        mask = torch.zeros(n, 2*n, dtype=torch.bool, device=device)
        for i in range(n):
            mask[i, 2*i] = True      # 排除自己
            mask[i, 2*i+1] = True   # 排除正样本
        
        all_sim = all_sim.masked_fill(mask, float('-inf'))
        
        # 选择 top-k 困难负样本
        neg_sim, _ = torch.topk(all_sim, k=min(self.top_k, 2*n-2), dim=1)
        
        # 构建 logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 标签
        labels = torch.zeros(n, dtype=torch.long, device=device)
        
        # 损失
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

### 5.3 可学习温度参数

```python
class InfoNCEWithLearnableTemperature(nn.Module):
    """
    可学习的温度参数
    让模型自动学习最优温度
    """
    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # 使用 log 空间，确保温度始终为正
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 使用 self.temperature 而不是固定值
        # ...（实现同标准版本）
        pass
```

---

## 六、面试常见问题（Q&A）

### Q1: InfoNCE 和交叉熵损失的关系？

**A:** InfoNCE 本质上是一个特殊的交叉熵损失：
- 将对比学习问题转化为多分类问题
- 正样本作为类别 0，负样本作为其他类别
- 使用温度参数控制分布的尖锐程度

**代码验证：**
```python
# InfoNCE
logits = [pos_sim, neg_sim1, neg_sim2, ...] / τ
labels = [0, 0, 0, ...]  # 第一个位置是正样本
loss = CrossEntropy(logits, labels)

# 等价于
loss = -log(exp(pos_sim/τ) / Σ exp(sim/τ))
```

### Q2: 为什么需要温度参数？

**A:** 三个原因：
1. **数值稳定性**：防止 softmax 饱和
2. **梯度平衡**：控制正负样本的梯度大小
3. **困难负样本**：小温度使模型更关注困难负样本

**实验验证：**
- τ = 0.01: 训练不稳定，容易过拟合
- τ = 0.07: 平衡点（SimCLR 使用）
- τ = 1.0: 训练稳定但学习慢

### Q3: 负样本数量如何影响性能？

**A:** 
- **理论**：负样本越多，互信息下界越紧
- **实践**：收益递减，通常 4096-8192 足够
- **计算**：负样本数量线性增加计算成本

**实验数据（SimCLR）：**
| 负样本数 | Top-1 Acc |
|---------|-----------|
| 256     | 60.0%     |
| 512     | 63.5%     |
| 1024    | 66.2%     |
| 4096    | 69.3%     |
| 8192    | 69.8%     |

### Q4: InfoNCE 和 Triplet Loss 的区别？

**A:** 

| 维度 | InfoNCE | Triplet Loss |
|------|---------|--------------|
| 负样本数 | 多个（N-1） | 1 个 |
| 优化目标 | 最大化互信息 | 最大化间隔 |
| 梯度特性 | 所有负样本都有梯度 | 只有困难负样本有梯度 |
| 计算复杂度 | O(N) | O(1) |

**代码对比：**
```python
# Triplet Loss
loss = max(0, margin + sim(anchor, negative) - sim(anchor, positive))

# InfoNCE
loss = -log(exp(sim(anchor, positive)/τ) / Σ exp(sim(anchor, sample)/τ))
```

### Q5: 如何解决假负样本（False Negatives）问题？

**A:** 三种方法：
1. **增加批次大小**：减少假负样本比例
2. **使用 MoCo 队列**：从历史批次采样负样本
3. **Debiased Contrastive Learning**：显式建模假负样本

### Q6: InfoNCE 的局限性？

**A:** 
1. **需要大量负样本**：计算成本高
2. **假负样本问题**：批次内可能存在相似样本
3. **温度参数敏感**：需要仔细调参
4. **不适用于生成任务**：只适用于表示学习

---

## 七、实践技巧与最佳实践

### 7.1 数据增强策略

```python
# SimCLR 的数据增强组合
augmentations = [
    RandomResizedCrop(),
    RandomHorizontalFlip(),
    ColorJitter(),
    RandomGrayscale(),
    GaussianBlur(),
]
```

**关键点：**
- 增强要足够强，但不能破坏语义
- 不同任务需要不同的增强策略

### 7.2 训练技巧

1. **学习率调度**：使用 warmup + cosine decay
2. **批次大小**：越大越好（受 GPU 内存限制）
3. **训练轮数**：通常需要 100-1000 轮
4. **特征维度**：128-512 维通常足够

### 7.3 评估指标

```python
# 线性评估（Linear Evaluation）
# 冻结编码器，只训练分类头
classifier = nn.Linear(feature_dim, num_classes)
optimizer = optim.SGD(classifier.parameters(), lr=0.1)

# k-NN 评估
# 使用 k-NN 分类器评估表示质量
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
```

### 7.4 调试技巧

```python
# 1. 检查特征归一化
assert torch.allclose(torch.norm(features, dim=1), torch.ones(batch_size))

# 2. 检查相似度范围
similarities = torch.matmul(features, features.t())
assert similarities.min() >= -1.0 and similarities.max() <= 1.0

# 3. 检查损失值
# 初始损失应该接近 log(N)，其中 N 是负样本数
expected_initial_loss = math.log(batch_size - 1)
print(f"Expected initial loss: {expected_initial_loss:.4f}")
```

---

## 八、完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import math


class ContrastiveModel(nn.Module):
    """简单的对比学习模型"""
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


def train_contrastive(model, dataloader, num_epochs=100):
    """训练对比学习模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = InfoNCE(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            
            # 获取特征
            z1 = model(x1)  # [batch_size, output_dim]
            z2 = model(x2)  # [batch_size, output_dim]
            
            # 合并特征 [2*batch_size, output_dim]
            features = torch.cat([z1, z2], dim=0)
            
            # 计算损失
            loss = criterion(features)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = ContrastiveModel(input_dim=784, hidden_dim=512, output_dim=128)
    
    # 创建数据加载器（需要返回两个增强版本）
    # dataloader = ...
    
    # 训练
    train_contrastive(model, dataloader, num_epochs=100)
```

---

## 九、总结与关键要点

### 9.1 核心要点

1. **理论基础**：InfoNCE 是互信息的下界，最大化互信息等价于最小化 InfoNCE
2. **温度参数**：控制分布尖锐度，小温度关注困难负样本
3. **负样本策略**：批次内负样本最常用，MoCo 使用队列机制
4. **数值稳定性**：必须进行 L2 归一化，使用 log-sum-exp 技巧

### 9.2 面试回答模板

> InfoNCE 是对比学习的核心损失函数，通过最大化正样本对的互信息、最小化负样本对的相似度来学习表示。关键点包括：①温度参数控制分布尖锐度，通常设为 0.07；②负样本数量影响互信息下界的紧度；③与交叉熵损失等价，但通过温度参数实现更精细的控制。在实际应用中，需要平衡批次大小、负样本数量和计算成本。

### 9.3 进一步学习

- **论文**：
  - SimCLR: A Simple Framework for Contrastive Learning
  - MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
  - CLIP: Learning Transferable Visual Models from Natural Language Supervision
  
- **代码库**：
  - SimCLR: https://github.com/google-research/simclr
  - MoCo: https://github.com/facebookresearch/moco

---
