import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size[0]
        else:
            self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, n_patches**0.5, n_patches**0.5]
        x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, n_patches] -> [B, n_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        print(self.pe.shape, x.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        assert embed_dim% num_heads==0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = int(self.embed_dim/ self.num_heads)
        self.dropout = dropout

        self.m_atten = nn.Linear(self.embed_dim, 3* self.embed_dim)
        self.m_proj = nn.Linear(self.embed_dim, embed_dim)
        self.atten_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        pe = torch.zeros(5000, embed_dim)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.tensor, return_attn: bool=False):
        B, T, C = x.size()
        qkv = self.m_atten(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pos_enc = self.pe[:T]  # [T, embed_dim]
        pos_enc = pos_enc.view(T, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)  # [1, nh, T, hd]
        q = q + pos_enc
        k = k + pos_enc
        if self.flash and return_attn== False:
            try:
                from flash_attn import flash_attn_func
                y = flash_attn_func(q, k, v,
                                    dropout_p= self.dropout if self.training else 0,
                                    causal= True)
            except Exception:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                    dropout_p= self.dropout if self.training else 0,
                                                                    is_causal= True)
        else:
            attn = (q@ k.transpose(-2, -1))* (1/ sqrt(k.size(-1)))
            attn = self.atten_dropout(F.softmax(attn, dim=-1))
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.m_proj(y))
        if return_attn:
            return y, attn
        return y

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, embed_dim, dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, return_attn):
        src2 = self.norm1(src)
        if return_attn:
            src2, attn = self.self_attn(src2, return_attn)
        else:
            src2 = self.self_attn(src2)
            attn = None
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if return_attn:
            return src, attn
        else:
            return src

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., num_classes=1000, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim*mlp_ratio), dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_attn=False, attn_layer_idx= None):
        B = x.shape[0]
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.repeat(B, 1, 1).clone()  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        # x = self.pos_encoder(x)  # Add positional encoding

        attentions = []
        for i,  layer in enumerate(self.encoder_layers):
            if return_attn:
                if attn_layer_idx is not None and i == attn_layer_idx:
                    x, attn = layer(x, return_attn=True)
                    attentions.append(attn)
                elif attn_layer_idx is None and i == len(self.encoder_layers) - 1: # 最后一层的atten
                    x, attn = layer(x, return_attn=True)
                    attentions.append(attn)
            else:
                x = layer(x, return_attn)
        x = self.norm(x)
        x = self.fc(x[:, 0])
        if return_attn:
            return x, attentions[0]
        else:
            return x
        
@torch.no_grad()
def visual_attentions(model, x: torch.tensor, return_attn: bool=True, attn_layer_idx=None, overlay: bool=True):
    '''可视化注意力'''
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    model.eval()
    x, attentions = model(in_data, return_attn, attn_layer_idx)
    attn = attn[0]  # [Heads, N, N]
    cls_attn = attn[:, 0, 1:]  # [Heads, Patch_Num]
    
    num_heads, num_patches = cls_attn.shape
    patch_size = int(np.sqrt(num_patches))
    h, w = patch_size, patch_size

    mean_attn = cls_attn.mean(0).reshape(1, 1, h, w)
    
    attn_map_up = F.interpolate(mean_attn, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
    attn_map_up = attn_map_up.squeeze().cpu().numpy()
    attn_map_up = (attn_map_up - attn_map_up.min()) / (attn_map_up.max() - attn_map_up.min())

    img = x.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    if overlay:
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_up), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay_img = 0.6 * img + 0.4 * heatmap
        plt.imshow(overlay_img)
        plt.title("Attention Overlay")
        plt.axis('off')
        plt.show()
    else:
        plt.imshow(attn_map_up, cmap='viridis')
        plt.title("Attention Heatmap")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_data = torch.randn(2, 3, 224, 224).to(device)  # b, c, h, w
    model = VisionTransformer(img_size= (224, 224), patch_size= 16).to(device)

    out_data, attention = model(in_data, return_attn= True)
    print(out_data.shape)
    print(attention.shape)