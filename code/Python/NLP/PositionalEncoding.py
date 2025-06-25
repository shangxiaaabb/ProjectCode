import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class Rotary2DPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 4 == 0, "Embed dim must be divisible by 4 for RoPE-2D"
        self.embed_dim = embed_dim
        self.freqs = 10000

    def get_2d_encoding(self, h, w, device):
        d_model = self.embed_dim
        d_half = d_model // 2
        d_quarter = d_model // 4

        y_embed = torch.arange(h, device=device).unsqueeze(1)
        x_embed = torch.arange(w, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_quarter, dtype=torch.float, device=device) *
                             -(math.log(self.freqs) / d_quarter))

        pos_x = x_embed * div_term
        pos_y = y_embed * div_term

        sin_x = torch.sin(pos_x).repeat(h, 1)
        cos_x = torch.cos(pos_x).repeat(h, 1)
        sin_y = torch.sin(pos_y).repeat(w, 1).transpose(0, 1)
        cos_y = torch.cos(pos_y).repeat(w, 1).transpose(0, 1)

        # Shape [h, w, d_model]
        pos = torch.cat([sin_y.unsqueeze(2), cos_y.unsqueeze(2),
                         sin_x.unsqueeze(2), cos_x.unsqueeze(2)], dim=2)
        pos = pos.view(h * w, d_model)
        return pos

    def forward(self, x, h, w):
        B, T, C = x.shape
        assert T == h * w, f"T={T} does not match h*w={h*w}"

        pos = self.get_2d_encoding(h, w, x.device)  # [T, C]
        pos = pos.unsqueeze(0).expand(B, -1, -1)    # [B, T, C]
        return x + pos

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        '''旋转位置编码'''
        super(RotaryPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x:torch.tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(AbsolutePositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                             -(torch.log(torch.tensor(10000.0)) / embed_dim))  # (embed_dim / 2,)

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        return x + pe

class LearnedPositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int=5000):
        super(LearnedPositionEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.size()
        position_indices = torch.arange(0, T, device=x.device)  # (T,)
        position_encodings = self.position_embeddings(position_indices)  # (T, C)
        position_encodings = position_encodings.unsqueeze(0)  # (1, T, C)
        return x + position_encodings
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    B, T, C = 2, 4, 16
    x = torch.randn(B, T, C).to(device)
    rope_embedding = RotaryPositionalEncoding(embed_dim= C)
    rope_2d_embedding = Rotary2DPositionalEncoding(embed_dim= C)
    abso_embedding = AbsolutePositionEmbedding(embed_dim=C)
    lear_embedding = LearnedPositionEmbedding(embed_dim= C)

    rope_embedding = rope_embedding.to(device)
    rope_2d_embedding = rope_2d_embedding.to(device)
    abso_embedding = abso_embedding.to(device)
    lear_embedding = lear_embedding.to(device)
    rope_out = rope_embedding(x)
    rope2d_out = rope_2d_embedding(x, 2, 2) # T=h*w
    abso_out = abso_embedding(x)
    lear_out = lear_embedding(x)
    print(rope_out.shape, rope2d_out.shape, abso_out.shape, lear_out.shape)