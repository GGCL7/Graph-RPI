import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected


def mask_edge(edge_index, p):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        self.p = p

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges
