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

class SPRCNN(nn.Module):
    def __init__(self, ):
        ...

class SPResizedLinear(nn.Module):
    def __init__(self, patch_size, embed_size):
        super(SPResizedLinear, self).__init__()
        self.lin = nn.Linear(patch_size, embed_size)
    
    def forward(self, x):
        n, c, h, w = x.size()
        x = x.reshape(n, c*h*w)
        
        return self.lin(x)

class SPGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads,
                 num_gat_layers, num_mlp_layers, dropout=0.2,
                 sine_pe=False, pe_size=256,
                 image_size=224, patch_size=7,
                 superpixel_aggregator="lrgb"):
        super(SPGAT, self).__init__()
        
        self.sp = superpixel_aggregator
        if superpixel_aggregator == "lrgb":
            self.sp_agg = nn.Linear(input_dim, hidden_dim)
        elif superpixel_aggregator == "cnn":
            self.sp_agg = SPRCNN()
        elif superpixel_aggregator == "linear":
            self.sp_agg = SPResizedLinear(input_dim*patch_size**2, hidden_dim)
            
        self.use_pe = sine_pe
        if sine_pe:
            self.pe = PositionalEncoding2D(pe_size, image_size**2)
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GATConv(hidden_dim+pe_size*sine_pe, hidden_dim, heads=num_heads, dropout=dropout))
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
        # edge_index=[2, 2090], labels=[504, 1], weight=[2090], count=[2090], num_nodes=504, centroid=[504, 2], imgs=[504, 1, 7, 7], masks=[32], y=[32], batch=[504], ptr=[33]
        x, edge_index, batch = data.imgs, data.edge_index, data.batch
        
        if self.sp == "linear":
            x = self.sp_agg(x)
        
        if self.use_pe:
            pe = self.pe(data.centroid.int()).to(x.device)
            x = torch.cat((x, pe), dim=-1)
        
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        return self.mlp(x)
    
    def to(self, device):
        super(SPGAT, self).to(device)
        if self.use_pe:
            self.pe = self.pe.to(device)
        return self

if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from ..datasets.image_classification import CIFAR100
    
    transform = Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize(0, 1),
    ])
    dataset = CIFAR100(root="./data/image/CIFAR100", download=True, transform=transform)
    dataset.to_slic_graphs(resize_stack_patches=(8,8), n_segments=16, compactness=15)
    
    print(dataset[0])