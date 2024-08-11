from torch import nn
from torch.nn import functional as F

class FCN(nn.Module):
    def __init__(self, num_layers, in_channels, hid_channels, out_channels, num_mlp_layers) -> None:
        super(FCN, self).__init__()
        
        self.convs = [nn.Conv2d(
            in_channels, hid_channels, kernel_size=(3, 3), stride=(2, 2),
            padding=(0, 0), bias=False
        )]
        self.convs += [nn.Conv2d(
            hid_channels, hid_channels, kernel_size=(3, 3), stride=(2, 2),
            padding=(0, 0), bias=False
        ) for _ in range(num_layers-1)]
        self.convs = nn.ModuleList(self.convs)
        
        self.mlp = [nn.Linear(hid_channels, hid_channels) for _ in range(num_mlp_layers-1)]
        self.mlp.append(nn.Linear(hid_channels, out_channels))
        self.mlp = nn.ModuleList(self.mlp)
        
    def forward(self, x):
        x = self.convs[0](x)
        for conv in self.convs[1:]:
            x = conv(F.relu(x))
        
        x = x.mean(dim=(2,3))
        
        x = self.mlp[0](x)
        for lin in self.mlp[1:]:
            x = lin(F.relu(x))
        
        return x
