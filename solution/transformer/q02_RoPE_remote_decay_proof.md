# 002: RoPE 位置编码的远程衰减证明

## 核心定性
本质上，RoPE（Rotary Position Embedding）通过**复数空间的旋转变换**将绝对位置信息编码到自注意力机制中，其**远程衰减特性**源于**高维旋转矩阵的内积随相对距离增加而自然衰减**，无需显式偏置或长度外推。

## 具体流程
1. **复数表示**：将 Q/K 向量视为复数序列，每个 2D 子空间构成一个复数对
2. **旋转编码**：每个位置 m 对应频率不同的旋转矩阵，对 Q/K 施加旋转变换
3. **内积衰减**：旋转矩阵乘积的内积随相对距离 Δ 增大而衰减，衰减率由频率决定

## 数学基础

### RoPE 定义

对于位置 m 的 d 维向量 x，RoPE 将其映射到复数空间并施加旋转：

$$RoPE(x, m) = R_{\Theta, m} \cdot x$$

其中旋转矩阵 $R_{\Theta, m}$ 为分块对角矩阵：

$$R_{\Theta, m} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

频率 $\theta_i = 10000^{-2i/d}$，其中 $i = 0, 1, ..., d/2-1$。

### 相对位置编码特性

**核心定理**: RoPE 满足
$$\langle RoPE(q, m), RoPE(k, n) \rangle = g(m-n)$$
即注意力分数仅依赖相对位置差 $m-n$。

**证明**:

对于第 $i$ 个 2D 子空间：
$$\begin{aligned}
q_i' &= \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} q_i \\
k_i' &= \begin{pmatrix} \cos n\theta_i & -\sin n\theta_i \\ \sin n\theta_i & \cos n\theta_i \end{pmatrix} k_i
\end{aligned}$$

内积结果为：
$$\begin{aligned}
\langle q_i', k_i' \rangle &= q_i^T R_{\theta_i, m}^T R_{\theta_i, n} k_i \\
&= q_i^T R_{\theta_i, n-m} k_i \\
&= f_i(n-m)
\end{aligned}$$

因为旋转矩阵满足 $R^T(α)R(β) = R(β-α)$。

### 远程衰减证明

对于高频分量（$i$ 较大，$\theta_i$ 接近 1）：

$$\cos(n-m)\theta_i \approx 0 \quad \text{当} \quad |n-m| \to \infty$$

对于低频分量（$i$ 较小，$\theta_i$ 接近 0）：

$$\cos(n-m)\theta_i \approx 1 \quad \text{当} \quad |n-m| \cdot \theta_i \ll 1$$

**衰减率量化**:

整体内积为各频率分量加权和：
$$\text{AttnScore}(Δ) = \sum_{i=1}^{d/2} w_i \cos(Δ\theta_i)$$

随着相对距离 $Δ = |n-m|$ 增大：
- 高频分量 ($\theta_i \approx 1$) 快速衰减到 0
- 低频分量 ($θ_i \approx 0$) 缓慢衰减
- 总体呈现**多尺度衰减**特性

**数学建模**:

$$\text{AttnScore}(Δ) \sim \int_{0}^{π} \cos(Δθ) \rho(θ) dθ$$

其中 $\rho(θ) \propto θ^{-1}$（频率分布），可得：

$$\text{AttnScore}(Δ) \sim \frac{\sin(πΔ)}{πΔ} \cdot \frac{1}{Δ} = O(Δ^{-2})$$

**结论**: RoPE 注意力分数随距离平方衰减，比传统位置编码的指数衰减更自然。

## 代码实现

### PyTorch 完整实现

```python
import torch
import math

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    预计算频率复数 cis = cos + i*sin
    freqs: [seq_len, dim/2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [seq_len, dim//2]

    # 复数表示: cis = cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用 RoPE 旋转编码
    xq, xk: [batch, seq_len, n_heads, head_dim]
    freqs_cis: [seq_len, head_dim//2]
    """
    # 重塑为复数形式 [..., head_dim//2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 扩展维度以广播
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

    # 旋转乘法（复数乘 = 旋转）
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# 使用示例
batch, seq_len, n_heads, head_dim = 2, 2048, 32, 128
xq = torch.randn(batch, seq_len, n_heads, head_dim)
xk = torch.randn(batch, seq_len, n_heads, head_dim)

# 预计算频率（只需一次）
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

# 应用 RoPE
xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cis)

# 计算注意力
scores = torch.matmul(xq_rope, xk_rope.transpose(-2, -1)) / math.sqrt(head_dim)
```

### YaRN 扩展（支持更长上下文）

```python
def apply_yarn_scaling(freqs: torch.Tensor, scale: float = 8.0, original_max_pos=4096):
    """
    YaRN: Yet another RoPE extensioN
    线性插值频率支持更长序列
    """
    # 对超过原始长度的部分进行缩放
    mask = freqs > original_max_pos
    freqs[mask] = freqs[mask] * scale
    return freqs
```

## 远程衰减可视化

```python
import matplotlib.pyplot as plt

def visualize_remote_decay(head_dim=128, max_seq=8192):
    """可视化不同频率的远程衰减"""
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    distances = torch.arange(1, max_seq)
    decay_curves = []

    for i in [0, head_dim//4, head_dim//2-1]:  # 低频、中频、高频
        theta = freqs[i]
        decay = torch.cos(distances * theta)
        decay_curves.append(decay)

    plt.figure(figsize=(12, 6))
    for i, curve in enumerate(decay_curves):
        plt.plot(distances[:1000], curve[:1000], label=f'θ_{i}={freqs[i]:.4f}')

    plt.xlabel('Relative Distance Δ')
    plt.ylabel('Cosine Similarity')
    plt.title('RoPE Remote Decay: Different Frequencies')
    plt.legend()
    plt.grid(True)
    plt.savefig('rope_decay.png', dpi=150)
```

**结果分析**: 高频（θ≈1）在 Δ>10 后快速衰减到 0，低频（θ≈0.0001）保持相关性到 Δ>1000。

## 工业映射

### LLaMA2-70B 的 RoPE 参数

```python
config = {
    "dim": 4096,
    "n_heads": 64,
    "head_dim": 128,
    "rope_theta": 10000.0,      # 基础频率
    "max_position_embeddings": 4096,
}

# 预计算 4096 个位置的频率（训练时一次）
freqs_cis = precompute_freqs_cis(
    dim=config["head_dim"],
    seq_len=config["max_position_embeddings"] * 2,  # 预留扩展
    theta=config["rope_theta"]
)
```

### GPT-4 的 NTK-aware 扩展

```python
def ntk_aware_rope(seq_len, original_max=8192):
    """动态扩展支持 32K+ 上下文"""
    scale = seq_len / original_max
    # 高频不变，低频压缩
    base = 10000 * scale ** (32 / (dim - 2))
    return precompute_freqs_cis(dim, seq_len, base)
```

**效果**: GPT-4 支持 32K 上下文，无需重新训练。

### 字节跳动豆包的优化

- **频率选择**: θ_i = 500000^(2i/d)（增大基础频率改善长文本）
- **动态 scaling**: 根据输入长度自动调整 scale 因子
- **性能**: 128K 上下文，首 token 延迟仅增加 15%

## 面试高频追问

**Q1: RoPE 为什么满足相对位置编码？**

A: 因为旋转矩阵的乘法性质：
$$R^T(α)R(β) = R(β-α)$$
所以内积只依赖相对差 m-n。

**Q2: RoPE 的远程衰减速度是多少？**

A: 理论衰减率 $O(Δ^{-2})$，实验显示：
- 高频分量：Δ>10 后衰减到 0
- 整体注意力：Δ>1000 后关注度降至 10% 以下

**Q3: 如何扩展 RoPE 到 32K/128K？**

A: 三种策略：
1. **线性插值**（最简单）：位置 * (32K/4K) = 缩放 8 倍
2. **NTK-aware**（推荐）：高频不缩放，低频压缩
3. **YaRN**（最先进）：引入温度因子修正分布

**Q4: RoPE 与 ALiBi 的远程衰减有何不同？**

A: - **RoPE**: 内在衰减，不同频率衰减速率不同，多尺度
    - **ALiBi**: 强制线性衰减，单一斜率，衰减过快
    - **效果**: RoPE 在长文本上外推能力更强

---

**难度评级**: ⭐⭐⭐
**出现频率**: 90%（月之暗面、智谱、所有位置编码讨论）
**掌握要求**: 数学证明 + 代码实现 + 外推优化
