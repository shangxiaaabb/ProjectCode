import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenWindowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 8, spatial_merge_size: int = 2, patch_size: int = 2, dropout: float = 0.2, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.window_size = window_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size  # 4

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def apply_rotary_pos_emb_vision(self, q, k, cos, sin):
        # Simplified RoPE application (assumes cos, sin are precomputed)
        # In practice, this would involve rotating Q and K based on position embeddings
        # Here, we assume cos and sin are applied directly (placeholder for clarity)
        q_rotated = q * cos + k * sin  # Simplified, actual RoPE is more complex
        k_rotated = k * cos - q * sin
        return q_rotated, k_rotated

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor, position_embeddings: tuple = None, use_window_attention: bool = True):
        B, T, C = x.shape
        seq_length = B * T

        # Generate window indices and sequence lengths
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        window_index = window_index.to(x.device)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=x.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Reshape and reorder input based on window_index
        x_flat = x.reshape(seq_length, C)  # [32, dim]
        x_reordered = x_flat[window_index]  # Reorder to window-based order
        x_reordered = x_reordered.reshape(seq_length // self.spatial_merge_unit, self.spatial_merge_unit, C)
        x_reordered = x_reordered.reshape(seq_length, C)  # [32, dim]

        # Generate Q, K, V
        qkv = self.qkv(x_reordered)  # [32, 3*dim]
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)  # [3, 32, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [32, num_heads, head_dim]

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos[window_index]  # Reorder to match window order
            sin = sin[window_index]
            q, k = self.apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Generate attention mask based on cu_seqlens
        attention_mask = torch.zeros([1, seq_length, seq_length], device=x.device, dtype=torch.bool)
        if use_window_attention:
            cu_seqlens = cu_window_seqlens
        else:
            # Compute global cu_seqlens (per frame)
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2] // (self.patch_size ** 2), grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(x.device)
        
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True

        # Compute attention
        q = q.transpose(0, 1)  # [num_heads, 32, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if self.flash:
            try:
                from flash_attn import flash_attn_func
                y = flash_attn_func(q, k, v, 
                                    dropout_p= self.dropout if self.training else 0,
                                    causal= True)
            except Exception:
                y = F.scaled_dot_product_attention(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=False
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
            att = att.masked_fill(~attention_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.atten_dropout(att)
            y = att @ v

        y = y.squeeze(0).transpose(0, 1).contiguous().view(seq_length, C)  # [32, dim]
        
        # Revert to original patch order
        reverse_indices = torch.argsort(window_index)
        y = y[reverse_indices]  # [32, dim]
        y = y.reshape(B, T, C)  # [B, T, C]
        
        # Project output
        y = self.resid_dropout(self.proj(y))
        return y

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    B, T, C = 2, 16, 1024
    num_heads = 4
    grid_thw = torch.tensor([[2, 8, 8]], dtype=torch.int32).to(device)
    # NOTE: grid_thw和b、T、C要batch可以对上 T=t*h*w
    x = torch.randn(B, T, C).to(device)  # patch embedding 之后：[2, 16, 16]
    
    # Dummy position embeddings (cos, sin) for RoPE
    position_embeddings = (
        torch.randn(B * T, num_heads, C // num_heads).to(device),  # cos
        torch.randn(B * T, num_heads, C // num_heads).to(device)   # sin
    )

    position_embeddings = None
    mha = QwenWindowAttention(
        embed_dim=C, num_heads=num_heads, window_size=8, spatial_merge_size=2, patch_size=2, dropout=0.1
    ).to(device)
    
    output = mha(x, grid_thw, position_embeddings, use_window_attention=True)
    print("Output shape:", output.shape)
