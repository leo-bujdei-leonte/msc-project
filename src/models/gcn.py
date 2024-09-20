import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as pyg_nn
from torch_geometric.transforms import AddLaplacianEigenvectorPE


class GCN(nn.Module):
    def __init__(self, num_layers, in_channels, hid_channels, out_channels, num_mlp_layers,
                 laplacian_pe=False, laplacian_k=5, is_undirected=True) -> None:
        super(GCN, self).__init__()
        
        self.laplacian_pe = laplacian_pe
        if laplacian_pe:
            self.transform = AddLaplacianEigenvectorPE(laplacian_k, is_undirected=is_undirected)
            in_channels += laplacian_k
        
        self.convs = [pyg_nn.GCNConv(in_channels, hid_channels, bias=False)]
        self.convs += [pyg_nn.GCNConv(hid_channels, hid_channels, bias=False) for _ in range(num_layers-1)]
        self.convs = nn.ModuleList(self.convs)
        
        self.mlp = [nn.Linear(hid_channels, hid_channels) for _ in range(num_mlp_layers-1)]
        self.mlp.append(nn.Linear(hid_channels, out_channels))
        self.mlp = nn.ModuleList(self.mlp)
    
    def forward(self, data, batch=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.laplacian_pe:
            pe = self.transform(data).laplacian_eigenvector_pe.to(x.device)
            x = torch.cat((x, pe), dim=-1)
        
        x = self.convs[0](x, edge_index)
        for conv in self.convs[1:]:
            x = conv(F.relu(x), edge_index)
        
        x = pyg_nn.pool.global_mean_pool(x, batch)
    
        x = self.mlp[0](x)
        for lin in self.mlp[1:]:
            x = lin(F.relu(x))
        
        return x
    