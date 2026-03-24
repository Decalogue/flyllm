import torch
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask if not None:
        scores = scores.mask_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # 投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 1.线性投影，分 head
        Q = self.W_Q(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        # Q K V shape: (batch_size, h, seq_len, d_k)

        # 2. Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn.output: (batch_size, h, seq_len, d_k)

        # 3.合并 head 并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)

        return output