import torch
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """标准 MHA"""
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

        # 1. 线性投影，分 head
        Q = self.W_Q(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        # Q K V shape: (batch_size, h, seq_len, d_k)

        # 2. Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn.output: (batch_size, h, seq_len, d_k)

        # 3. 合并 head 并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)

        return output


class MultiQueryAttention(nn.Module):
    """MQA - 所有 head 共享 KV"""
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # 投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, self.d_k)
        self.W_V = nn.Linear(d_model, self.d_k)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 1. Q 投影，分 head
        Q = self.W_Q(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        K = self.W_K(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        # 广播到 h 个 head
        K = K.expand(batch_size, self.h, seq_len, self.d_k)
        V = V.expand(batch_size, self.h, seq_len, self.d_k)

        # 2. Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并 head 并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        return output


class GroupedQueryAttention(nn.Module):
    """GQA - g 个 KV 组"""
    def __init__(self, d_model, h, g):
        super().__init__()
        assert d_model % h == 0
        assert h % g == 0
        self.d_model = d_model
        self.h = h
        self.g = g
        self.d_k = d_model // h
        self.group_size = h // g

        # 投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, g * self.d_k)
        self.W_V = nn.Linear(d_model, g * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Q 投影，分 head
        Q = self.W_Q(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # KV 投影：g 个组
        K = self.W_K(x).view(batch_size, seq_len, self.g, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.g, self.d_k).transpose(1, 2)

        # repeat
        K = K.unsqueeze(2).expand(batch_size, self.g, self.group_size, seq_len, self.d_k)
        V = V.unsqueeze(2).expand(batch_size, self.g, self.group_size, seq_len, self.d_k)

        K = K.contiguous().view(batch_size, self.h, seq_len, self.d_k)
        V = V.contiguous().view(batch_size, self.h, seq_len, self.d_k)

        # 2. Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并 head 并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)
        return output
