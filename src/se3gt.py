import torch
import torch.nn as nn
from se3_transformer_pytorch import SE3Transformer
from egnn import E_GCL
from se3_gnn import Clof_GCL
from torchinfo import summary
from torch_geometric.utils import to_undirected

class SE3GTLayer(nn.Module):
    def __init__(self, indim, hidden, outdim, edgedim, ydim):
        super().__init__()

        # self.gnn = Clof_GCL(indim, outdim, hidden, edges_in_d=edgedim)
        self.gnn =E_GCL(indim, indim, hidden, edges_in_d=edgedim)
        self.transformer = SE3Transformer(
                            dim = indim,
                            heads = 1,
                            depth = 1,
                            dim_head = 64,
                            num_degrees = 4,
                            valid_radius = 10
                        )
        self.combiner = nn.Sequential(
                            nn.Linear(2*outdim, hidden),
                            nn.ReLU(),
                            nn.LayerNorm(hidden),
                            nn.Linear(hidden, ydim),
                        )

        self.outdim = outdim

    def forward(self, batch):
        x, edge_index, edge_attr, coords = batch.x, batch.edge_index, batch.edge_attr, batch.pos
        x_gnn, coords_, edge_attr_ = self.gnn(x, edge_index, coords, edge_attr=edge_attr)

        mask = torch.ones(x.size(0), x.size(1)).bool()
        x_tf = self.transformer(x, coords, mask)

        print (x_gnn.shape, x_tf.shape)
        combined = x_gnn + x_tf

        return self.combiner(combined)