import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)                                # (B, T, H * D)
        self.kv_proj = nn.Linear(embed_dim, 2 * self.head_dim * num_kv_heads, bias=bias)         # (B, T, 2 * h_kv * D)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, nh, T, hd)
        kv = self.kv_proj(x).reshape(B, T, self.num_kv_heads, 2, self.head_dim).permute(3, 0, 2, 1, 4)
        k, v = kv[0], kv[1]  # (B, n_kv, T, hd)

        # 将 KV 扩展为每组 query 使用
        k = k.repeat_interleave(self.group_size, dim=1)  # (B, nh, T, hd)
        v = v.repeat_interleave(self.group_size, dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        if self.flash:
            try:
                from flash_attn import flash_attn_func
                y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                                   dropout_p=self.dropout if self.training else 0.0,
                                                   is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == float('-inf'), float('-inf'))
            att = self.atten_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    B, T, C = 2, 5, 16
    num_heads = 4
    num_kv_heads = 2
    x = torch.randn(B, T, C).to(device)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ], dtype=torch.float32).to(device)  # shape: (B, T)
    mha = GroupedQueryAttention(embed_dim=C, num_heads=num_heads, num_kv_heads= 2, dropout=0.1)
    mha = mha.to(device)
    output = mha(x, attention_mask)
    print("输出形状:", output.shape)