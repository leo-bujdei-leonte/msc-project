import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

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

from .vit import VisionEncoder


def segment_image_slic_avg_pool(image, n_segments=14*14, compactness=0.5):
    assert image.shape[0] == 1
    
    image = np.array(image.squeeze(0))
    
    seg = slic(image, n_segments=n_segments, compactness=compactness, channel_axis=None)
    reg = regionprops(seg, image)
    
    intensities = torch.tensor(list(map(lambda r: r.mean_intensity, reg))).unsqueeze(-1)
    positions = torch.tensor(list(map(lambda r: r.centroid, reg)))

    tokens = torch.cat((intensities, positions), dim=-1)
    
    return tokens.float()


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
        self.positional_encoding = nn.Parameter(torch.randn(1, num_regions+1, embed_size))
        
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
    

if __name__ == "__main__":
    image = pickle.load(open("./data/image/digit_example_tensor.pkl", "rb"))
    
    model = SLICTransformer(3, 512, 8, 10, 8, 256, 0.1)
    tokens = segment_image_slic_avg_pool(image).unsqueeze(0)
    model(tokens)
    
    image = np.array(image.squeeze(0))
    seg = slic(image, n_segments=14*14, compactness=0.5, channel_axis=None)
    
    plt.imshow(mark_boundaries(image, seg, color=(255, 0, 0)))
    plt.show()