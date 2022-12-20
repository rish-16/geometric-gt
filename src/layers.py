import torch
import torch.nn as nn
from torchinfo import summary

import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from se3gt import SE3GTLayer
from se3_transformer_pytorch import SE3Transformer

target = 0
dim = 64

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = tg.utils.remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9("./bin", transform=transform).shuffle()

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
batch = next(iter(train_loader))
# print (batch)

# layer = SE3GTLayer(dataset.num_features, dataset.num_features, dataset.num_features, dataset.num_edge_features, 1)
# x_ = layer(batch)
print (batch.x.shape, batch.pos.shape)
# print (x_.shape)

layer = SE3Transformer(dim=dataset.num_features, num_degrees = 4, heads=1, depth=1)
y = layer(batch.x, batch.pos, mask=None)