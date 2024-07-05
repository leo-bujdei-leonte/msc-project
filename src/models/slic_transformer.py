import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import Resize

import random
import scipy
import pickle
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage import filters, color, graph

import scipy.ndimage
import scipy.spatial

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

from .vit import VisionEncoder, PositionalEncoding2D
from ..utils.segmentation import segment_image_slic_avg_pool


class SLICTransformer(nn.Module):
    def __init__(self, in_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout, num_regions=256):
        super(SLICTransformer, self).__init__()
        
        self.in_size = in_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_regions = num_regions
        
        self.dropout_layer = nn.Dropout(dropout)
        self.embeddings = nn.Linear(in_size, embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_size))
        
        self.encoders = nn.ModuleList([
            VisionEncoder(embed_size, num_heads, hidden_size, dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_size)
        
        self.classifier = nn.Sequential(nn.Linear(embed_size, classes))
    
    def forward(self, x, mask=None):
        x = self.embeddings(x)
        
        b, n, e = x.size()
        
        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        x = self.dropout_layer(x)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        x = x[:, -1, :]
        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)
        
        return x
    

class CoordViT(nn.Module):
    r"""Vision Transformer Model

        A transformer model to solve vision tasks by treating images as sequences of tokens.

        Args:
            image_size      (int): Size of input image
            channel_size    (int): Size of the channel
            patch_size      (int): Max patch size, determines number of split images/patches and token size
            embed_size      (int): Embedding size of input
            num_heads       (int): Number of heads in Multi-Headed Attention
            classes         (int): Number of classes for classification of data
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

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

        x = x.reshape(b, n, h*w)
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



if __name__ == "__main__":
    image = pickle.load(open("./data/image/digit_example_tensor.pkl", "rb"))
    
    model = SLICTransformer(3, 512, 8, 10, 8, 256, 0.1)
    tokens = segment_image_slic_avg_pool(image).unsqueeze(0)
    model(tokens)
    
    image = np.array(image.squeeze(0))
    seg = slic(image, n_segments=14*14, compactness=0.5, channel_axis=None)
    
    plt.imshow(mark_boundaries(image, seg, color=(255, 0, 0)))
    plt.show()