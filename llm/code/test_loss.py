import torch
import torch.nn.functional as F


def ForCausalLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    面试常见 Causal LM Loss（next-token prediction）。

    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """
    # 1) 预测第 t+1 个 token：丢掉最后一个时刻的 logits
    shift_logits = logits[:, :-1, :].contiguous()
    # 2) 标签左移一位：丢掉第一个 token
    shift_labels = labels[:, 1:].contiguous()

    # 3) 展平后做交叉熵
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
