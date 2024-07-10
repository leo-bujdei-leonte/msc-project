import torch
from torch import nn
import torch.nn.functional as F

from .vit import VisionEncoder, PositionalEncoding2D


class CoordViT(nn.Module):
    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, dropout: float = 0.1):
        super(CoordViT, self).__init__()

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

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = PositionalEncoding2D(self.embed_size, self.image_size)

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, coords, mask):
        b, n, c, h, w = x.size()

        x = x.reshape(b, n, c*h*w)
        x = self.embeddings(x)

        b, n, e = x.size()

        pe = self.positional_encoding(coords.int()).to(x.device) # TODO int
        x = x + pe

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        mask = torch.cat((mask, torch.tensor([False] * mask.shape[0], device=mask.device).unsqueeze(1)), dim=1)

        x = self.dropout_layer(x)

        for encoder in self.encoders:
            x = encoder(x, mask)

        x = x[:, -1, :]

        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)

        return x

    def to(self, device):
        super(CoordViT, self).to(device)
        self.positional_encoding = self.positional_encoding.to(device)
        return self
