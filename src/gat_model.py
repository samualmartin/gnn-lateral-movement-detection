import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4,
            dropout=0.2
        )

        self.gat2 = GATConv(
            hidden_channels * 4,
            hidden_channels,
            heads=1,
            dropout=0.2
        )

        self.classifier = torch.nn.Linear(
            hidden_channels,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = self.classifier(x)

        return x