import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_len=1000):
        super(PositionalEncoding2D, self).__init__()
        self.dim = dim
        
        self.pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, coords): # TODO forces ints
        pe_x = self.pe[coords[..., 0]]
        pe_y = self.pe[coords[..., 1]]
        
        return pe_x + pe_y
    
    def to(self, device):
        super(PositionalEncoding2D, self).to(device)
        self.pe = self.pe.to(device)
        return self


class VisionEncoder(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        # self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        
        self.attention = nn.MultiheadAttention(self.embed_size, self.num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.embed_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x)
        
        x = x.transpose(0, 1)
        attn, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + attn
        x = x.transpose(0, 1)
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViT(nn.Module):
    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, mlp_size: int, dropout: float = 0.1, learnable_pe=True):
        super(ViT, self).__init__()

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.learnable_pe = learnable_pe
        self.mlp_size = mlp_size

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        if learnable_pe:
            self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_size))
        else:
            self.positional_encoding = PositionalEncoding2D(self.embed_size, self.num_patches+1)

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.mlp_size),
            nn.GELU(),
            nn.Linear(self.mlp_size, self.classes),
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()
        
        if not self.learnable_pe:
            coords = torch.cartesian_prod(torch.arange(int(h / self.p)), torch.arange(int(w / self.p))).unsqueeze(0).expand(b, int(h * w / (self.p * self.p)), 2) # TODO looks horrible
            pe = self.positional_encoding(coords).to(x.device)
            x = x + pe
        
        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        
        if self.learnable_pe:
            x = x + self.positional_encoding
        
        x = self.dropout_layer(x)

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = x[:, -1, :]

        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)

        return x

