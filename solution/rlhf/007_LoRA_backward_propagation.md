# 007: LoRA 反向传播与梯度计算

## 核心定性
LoRA（Low-Rank Adaptation）的反向传播本质上是一个**冻结主网络梯度** + **低秩矩阵高效更新**的优化过程，通过**梯度旁路**（gradient bypass）机制，将梯度计算限制在可训练的 $BA$ 低秩分解上，主网络 $W$ 不参与梯度回传，从而实现**显存节省 3×** 和**训练加速 2×** 的效果。

## 前向传播回顾

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)  # 主网络（冻结）
        self.W.requires_grad = False

        self.rank = rank
        self.alpha = alpha

        # 低秩分解: ΔW = BA
        self.B = nn.Linear(rank, out_features, bias=False)  # B: [out, rank]
        self.A = nn.Linear(in_features, rank, bias=False)   # A: [rank, in]

        # 初始化: A 随机小值, B 初始化为 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        """
        y = Wx + (BA)x / α
        """
        # 主网络输出（无梯度）
        main_output = self.W(x)

        # LoRA 输出（可训练）
        lora_output = self.B(self.A(x)) / self.alpha

        return main_output + lora_output
```

## 反向传播推导

### 梯度流分析

**假设**: 输入 $x$，输出 $y$，上游梯度 $\frac{\partial L}{\partial y} = g_y$

**步骤 1**: 主网络 $W$（冻结，无梯度）
$$\frac{\partial L}{\partial W} = 0 \quad \text{(requires_grad=False)}$$

**步骤 2**: 低秩矩阵 $B$ 和 $A$（可训练）

**计算图**:
```
x ----> A ----> h ----> B ----> Δh ----> /α ----> + ----> y
                                              ↑
                                              | W(x)  (冻结)
```

### 梯度公式

**B 的梯度**:
$$\frac{\partial L}{\partial B} = \frac{1}{α} \cdot g_y \cdot h^T$$
其中 $h = A(x)$ 是中间激活值。

**A 的梯度**:
$$\frac{\partial L}{\partial A} = \frac{1}{α} \cdot B^T \cdot g_y \cdot x^T$$

**实现代码**:
```python
def lora_backward(x, h, gy, B, alpha):
    """
    Args:
        x: 输入 [batch, in_features]
        h: A(x) 输出 [batch, rank]
        gy: 上游梯度 [batch, out_features]
        B: B 矩阵 [out_features, rank]
        alpha: 缩放因子
    """
    # B 的梯度: (gy.T @ h) / alpha
    dB = torch.matmul(gy.transpose(0, 1), h) / alpha  # [out, rank]

    # A 的梯度: (B.T @ gy).T @ x / alpha
    dA = torch.matmul(torch.matmul(B.T, gy).transpose(0, 1), x) / alpha  # [rank, in]

    # 返回梯度（用于 optimizer.step）
    return dA, dB
```

## PyTorch 自动微分版本

```python
# 使用 PyTorch 的自动微分（推荐）
lora_linear = LoRALinear(in_features=4096, out_features=4096, rank=16)

# 前向
x = torch.randn(32, 4096)  # batch=32
y = lora_linear(x)

# 计算 Loss
loss = y.sum()

# 反向传播（只计算 B 和 A 的梯度）
loss.backward()

# 检查梯度
assert lora_linear.A.weight.grad is not None
assert lora_linear.B.weight.grad is not None
assert lora_linear.W.weight.grad is None  # 主网络无梯度

# 梯度形状
print(lora_linear.A.weight.grad.shape)  # [rank=16, in=4096]
print(lora_linear.B.weight.grad.shape)  # [out=4096, rank=16]
```

### 梯度检查（Numerical Gradient Check）

```python
def gradient_check(lora_layer, x, epsilon=1e-5):
    """验证梯度计算正确性"""
    y = lora_layer(x)
    loss = y.sum()
    loss.backward()

    # 数值梯度
    param = lora_layer.A.weight
    analytic_grad = param.grad.clone()

    param_flat = param.data.flatten()
    numeric_grad = torch.zeros_like(param_flat)

    for i in range(len(param_flat)):
        # θ + ε
        param_flat[i] += epsilon
        y_plus = lora_layer(x)
        loss_plus = y_plus.sum()

        # θ - ε
        param_flat[i] -= 2 * epsilon
        y_minus = lora_layer(x)
        loss_minus = y_minus.sum()

        # 中心差分
        numeric_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # 恢复
        param_flat[i] += epsilon

    # 比较
    diff = torch.norm(analytic_grad.flatten() - numeric_grad) /
           torch.norm(analytic_grad.flatten() + numeric_grad)
    print(f"Gradient check diff: {diff:.6f} (should < 1e-5)")
    return diff < 1e-5
```

## 优化器配置

```python
# 只训练 LoRA 参数+
optimizer = torch.optim.AdamW(
    [
        {"params": lora_linear.A.parameters(), "lr": 1e-4},
        {"params": lora_linear.B.parameters(), "lr": 1e-4},
    ],
    weight_decay=0.01
)

# 主网络参数不传入 optimizer
# lora_linear.W.parameters() 被排除

def print_trainable_params(model):
    """统计可训练参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} | Total: {total:,} | Ratio: {trainable/total:.2%}")

# LLaMA-7B 示例
# Trainable: 4,194,304 | Total: 6,742,609,920 | Ratio: 0.06%
```

## 内存节省分析

**标准微调（全参数）**:
```
参数量: Φ = 12B (FP16)
梯度: Φ = 12B (FP16)
优化器状态: 2Φ (momentum + variance, FP32) = 24B
总计: 48GB（不可训）
```

**LoRA 微调（rank=16）**:
```
主网络: 2Φ = 24GB（冻结，无梯度）
LoRA 参数量: 2 * Φ*d * rank = 40MB（可训）
LoRA 梯度: 40MB（可训）
LoRA 优化器: 80MB（可训）
总计: ~24.1GB（可训）

节省: 99.9% 可训练参数量
速度: 前向+反向快 2-3x（小矩阵乘法）
```

## QLoRA 扩展（4-bit 量化）

```python
class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        # 主网络: NF4 量化（冻结）
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.data = quantize_to_nf4(self.W.weight.data)  # 4-bit
        self.W.requires_grad = False

        # LoRA: BF16 保持精度
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        # 主网络: 反量化计算（dequantize）
        output = dequantize_fp16(self.W(x))

        # LoRA: 高精度计算
        lora_output = self.lora_B(self.lora_A(x)) / self.alpha

        return output + lora_output

# 效果: 65B 模型可在单张 48GB GPU 上微调
