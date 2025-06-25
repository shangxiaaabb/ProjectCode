import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: float, num_heads: int, dropout: float= 0.2, bias: bool=False):
        super().__init__()
        assert embed_dim% num_heads== 0
        self.head_dim = embed_dim// num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.m_attn = nn.Linear(embed_dim, 3* embed_dim,
                                bias= bias)
        self.m_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        
        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.tensor, attention_mask: torch.tensor=None):
        B, T, C = x.shape
        qkv = self.m_attn(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if attention_mask is not None:
            # Expand to shape: (B, 1, 1, T) to broadcast
            attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            # Convert mask to float with -inf where masked
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        if self.flash:
            try:
                from flash_attn import flash_attn_func
                y = flash_attn_func(q, k, v, 
                                    dropout_p= self.dropout if self.training else 0,
                                    causal= True)
            except Exception:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                                   dropout_p= self.dropout if self.training else 0,
                                                   is_causal= False)
        else:
            att = (q@ k.transpose(-2, -1))* (1/ torch.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = self.atten_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.m_proj(y))
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
    mha = MultiHeadAttention(embed_dim=C, num_heads=num_heads, dropout=0.1)
    mha = mha.to(device)
    output = mha(x, attention_mask)
    print("输出形状:", output.shape)