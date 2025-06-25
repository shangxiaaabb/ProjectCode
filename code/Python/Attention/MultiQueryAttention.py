import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        # MQA: 单独定义 Q 的线性层，K 和 V 共享单头
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 多头 Q
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # 单头 K
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # 单头 V
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        B, T, C = x.shape

        # 计算 Q, K, V
        q = self.q_proj(x)  # (B, T, embed_dim)
        k = self.k_proj(x)  # (B, T, head_dim)
        v = self.v_proj(x)  # (B, T, head_dim)

        # 重塑 Q 为多头形式，K 和 V 保持单头
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, T, head_dim)
        k = k.unsqueeze(1)  # (B, 1, T, head_dim)
        v = v.unsqueeze(1)  # (B, 1, T, head_dim)

        if attention_mask is not None:
            # 扩展为 (B, 1, 1, T) 以广播
            attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        if self.flash:
            try:
                from flash_attn import flash_attn_func
                y = flash_attn_func(q, k, v, 
                                  dropout_p=self.dropout if self.training else 0,
                                  causal=True)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                                 dropout_p=self.dropout if self.training else 0,
                                                 is_causal=False)
        else:
            # 手动计算注意力
            att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
            if attention_mask is not None:
                att = att + attention_mask  # 直接加，因为 mask 已转为 -inf/0
            att = F.softmax(att, dim=-1)
            att = self.atten_dropout(att)
            y = att @ v  # (B, num_heads, T, head_dim)

        # 重塑输出并投影
        y = y.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)
        y = self.resid_dropout(self.out_proj(y))
        return y

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    B, T, C = 2, 5, 16
    num_heads = 4
    x = torch.randn(B, T, C).to(device)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ], dtype=torch.float32).to(device)  # shape: (B, T)
    mqa = MultiQueryAttention(embed_dim=C, num_heads=num_heads, dropout=0.1)
    mqa = mqa.to(device)
    output = mqa(x, attention_mask)
    print("输出形状:", output.shape)