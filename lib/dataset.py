import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, graphs, k_hop=10, n_graphs=10**6, quantile=0.5):
        self.graphs = graphs
        self.k_hop = k_hop
        self.n_graphs = n_graphs
        self.prune_graphs(quantile)

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, item):
        idx = np.random.choice(list(self.graphs.keys()))
        return self.sample_subgraph(self.graphs[idx], self.k_hop)

    @staticmethod
    def sample_subgraph(graph, k_hop=10):
        idx = torch.randint(0, len(graph.x) - 1, (1,))
        nodes, edge_idx, _, _ = k_hop_subgraph(idx, k_hop, graph.edge_index, relabel_nodes=True,
                                               num_nodes=graph.num_nodes)
        return Data(edge_index=edge_idx, x=graph.x[nodes], y=graph.y[nodes])

    def prune_graphs(self, quantile):
        for idx in self.graphs:
            graph = self.graphs[idx]
            edge_index, edge_attr = graph.edge_index, graph.edge_attr

            threshold = torch.quantile(graph.edge_attr, quantile)

            mask = (edge_attr > threshold).view(-1)

            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
            graph.edge_index = edge_index
            graph.edge_attr = edge_attr
