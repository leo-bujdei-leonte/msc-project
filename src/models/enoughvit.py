import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .vit import PositionalEncoding2D

class MultiheadEnoughAttention(nn.Module):
    # adapted from https://colab.research.google.com/drive/1EEsWWPLJaKUl4y7YdvwS2usjEVBCPN7j?usp=sharing#scrollTo=uX2PTLlxJh7f
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiheadEnoughAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_v = nn.Sequential(nn.Linear(d_model, d_model, bias=False), nn.Dropout(dropout))
        self.theta = nn.Sequential(nn.Linear(d_model, d_model, bias=False), nn.Dropout(dropout))

    def enough_attention(self, X):
        n, d = X.size(1), self.d_model

        # Step 1: Compute XW_v
        XW_v = self.W_v(X)

        # Step 2: Compute the first term
        ones_matrix = torch.ones(n, n, device=X.device)
        first_term = (1 / n) * ones_matrix @ XW_v

        # Step 3: Compute the second term
        scores = torch.matmul(X, self.theta(X).transpose(1, 2))
        second_term = (1 / (n * math.sqrt(d))) * (scores @ XW_v)

        # Combining the terms
        output = first_term + second_term
        return output

    def forward(self, X, Y=None, Z=None, mask=None):
        return self.enough_attention(X)


class EnoughViTEncoder(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super(EnoughViTEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        # self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        
        self.attention = MultiheadEnoughAttention(self.embed_size, self.num_heads, dropout=dropout)

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
        attn = self.attention(x)
        x = x + attn
        x = x.transpose(0, 1)
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class EnoughViT(nn.Module):
    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, mlp_size: int, dropout: float = 0.1, learnable_pe=True):
        super(EnoughViT, self).__init__()

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
            self.encoders.append(EnoughViTEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

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

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = EnoughViT(224, 3, 16, 128, 4, 100, 3, 128, 1024, 0.1, True)
    print(model(x).shape)
    
