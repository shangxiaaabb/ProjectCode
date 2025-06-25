import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: int, dropout: float = 0.2, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert window_size > 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.dropout = dropout

        self.w_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.w_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        B, T, C = x.shape
        qkv = self.w_attn(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, head_dim)

        # Create window-based attention
        outputs = []
        for t in range(0, T, self.window_size):
            # Extract window
            window_end = min(t + self.window_size, T)
            q_window = q[:, :, t:window_end, :]  # (B, num_heads, window_size, head_dim)
            k_window = k[:, :, t:window_end, :]
            v_window = v[:, :, t:window_end, :]

            # Handle attention mask for the window
            if attention_mask is not None:
                window_mask = attention_mask[:, t:window_end]  # (B, window_size)
                window_mask = window_mask[:, None, None, :]  # (B, 1, 1, window_size)
                window_mask = window_mask.masked_fill(window_mask == 0, float("-inf"))
                window_mask = window_mask.masked_fill(window_mask == 1, 0.0)
            else:
                window_mask = None

            if self.flash:
                try:
                    from flash_attn import flash_attn_func
                    y_window = flash_attn_func(
                        q_window, k_window, v_window,
                        dropout_p=self.dropout if self.training else 0,
                        causal=False
                    )
                except Exception:
                    y_window = F.scaled_dot_product_attention(
                        q_window, k_window, v_window,
                        attn_mask=window_mask,
                        dropout_p=self.dropout if self.training else 0,
                        is_causal=False
                    )
            else:
                att = (q_window @ k_window.transpose(-2, -1)) * (1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
                if window_mask is not None:
                    att = att.masked_fill(window_mask == 0, float('-inf'))
                att = self.atten_dropout(F.softmax(att, dim=-1))
                y_window = att @ v_window

            outputs.append(y_window)

        # Concatenate all window outputs
        y = torch.cat(outputs, dim=2)  # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.w_proj(y))
        return y

if __name__ == '__main__':
    import time
    from MultiHeadAttention import MultiHeadAttention
    from QwenWindowAttention import QwenWindowAttention
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    B, T, C = 2, 25, 1024
    num_heads = 8
    window_size = 4
    x = torch.randn(B, T, C).to(device)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1]*5,
        [1, 1, 1, 0, 0]*5
    ], dtype=torch.float32).to(device)  # shape: (B, T)
    grid_thw = torch.tensor([[2, 1024, 1024]], dtype=torch.int32).to(device)

    wa = WindowAttention(embed_dim=C, num_heads=num_heads, window_size=window_size, dropout=0.1)
    mha = MultiHeadAttention(embed_dim= C, num_heads= num_heads)
    qw_wa = QwenWindowAttention(embed_dim= C, num_heads= num_heads, window_size= window_size)
    qw_wa = qw_wa.to(device)
    wa = wa.to(device)
    mha = mha.to(device)

    start_time_wa = time.time()
    output = wa(x, attention_mask)
    print(f"WindowAttention Time:{time.time()- start_time_wa} {output.shape}")

    start_time_mha = time.time()
    output = mha(x, attention_mask)
    print(f"MultiHeadAttention Time:{time.time()- start_time_wa} {output.shape}")

    start_time_mha = time.time()
    output = qw_wa(x, attention_mask, grid_thw, use_window_attention=True)
    print(f"QwenWindowAttention Time:{time.time()- start_time_wa} {output.shape}")
