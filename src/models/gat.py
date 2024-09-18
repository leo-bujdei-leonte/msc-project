import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb


class GAT1(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT1, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_gat_layers, num_mlp_layers, dropout=0.2):
        super(GAT2, self).__init__()
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

    def forward(self, x, edge_index, batch):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        return self.mlp(x)
