import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from .vit import PositionalEncoding2D

class PixelGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads,
                 num_gat_layers, num_mlp_layers, dropout=0.2,
                 pe="", laplacian_k=5, laplacian_undirected=True,
                 sinusoidal_size=256, image_size=224):
        super(PixelGAT, self).__init__()
        
        self.use_pe = pe
        if pe == "laplacian":
            self.pe = AddLaplacianEigenvectorPE(laplacian_k, is_undirected=laplacian_undirected)
            input_dim += laplacian_k
        elif pe == "sinusoidal":
            self.pe = PositionalEncoding2D(sinusoidal_size, image_size**2)
            input_dim += sinusoidal_size
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        # Hidden layers
        for _ in range(num_gat_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=False))
        
        self.dropout = dropout
        
        self.mlp = nn.Sequential()
        for _ in range(num_mlp_layers-1):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.use_pe == "laplacian":
            pe = self.pe(data).laplacian_eigenvector_pe.to(x.device)
            x = torch.cat((x, pe), dim=-1)
        elif self.use_pe == "sinusoidal":
            pe = self.pe(data.coords.int()).to(x.device)
            x = torch.cat((x, pe), dim=-1)
        
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        return self.mlp(x)
    
    def to(self, device):
        super(PixelGAT, self).to(device)
        if self.use_pe == "sinusoidal":
            self.pe = self.pe.to(device)
        return self
