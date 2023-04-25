import torch

from torch import nn
from torch_geometric.nn import GCNConv


class Embedding(nn.Module):
    def __init__(self, vocab_ranges, embedding_sizes):
        super(Embedding, self).__init__()
        self.embeds = []
        for vocab_range, embedding_size in zip(vocab_ranges, embedding_sizes):
            self.embeds.append(nn.Embedding(vocab_range[1] - vocab_range[0] + 2, embedding_size))
        self.embeds = nn.ModuleList(self.embeds)

    def forward(self, cat_features):
        res = []
        for embed, cat_feat in zip(self.embeds, cat_features.T):
            res.append(embed(cat_feat))
        return torch.cat(res, dim=1)


class GCNN(nn.Module):
    def __init__(self, h_dim, num_layers):
        super(GCNN, self).__init__()
        module_list = [GCNConv(h_dim, h_dim) for _ in range(num_layers)]
        self.module_list = nn.ModuleList(module_list)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for layer in self.module_list:
            x = layer(x, edge_index)
            x = self.activation(x)
        return x


class FCNN(nn.Module):
    def __init__(self, h_dim, output_dim, num_layers):
        super(FCNN, self).__init__()
        module_list = [nn.Linear(h_dim, h_dim) for _ in range(num_layers - 1)]
        self.last_layer = nn.Linear(h_dim, output_dim)
        self.module_list = nn.ModuleList(module_list)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        return x


class NodePredictor(nn.Module):
    def __init__(self,
            feat_ranges,
            h_dims,
            num_conv_layers=6,
            num_fcn_layers=1,
            output_size=5):
        super(NodePredictor, self).__init__()
        self.h_dim = sum(h_dims)
        self.embed = Embedding(feat_ranges, h_dims)
        self.gcn = GCNN(self.h_dim, num_conv_layers)
        self.fcnn = FCNN(self.h_dim, output_size, num_fcn_layers)

    def forward(self, batch):
        x = self.embed(batch.x)
        x = self.gcn(x, batch.edge_index)
        x = self.fcnn(x)
        if x.shape[-1] == 1:
            x = x[..., 0]
        return x


if __name__ == "__main__":
    from torch_geometric.data import Data, DataLoader

    edge_index_s = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    x_s = torch.randint(10, 12, (5, 1))  # 5 nodes.

    edge_index_t = torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
    ])

    x_t = torch.randint(10, 12, (4, 1))  # 4 nodes.


    gs = Data(x_s, edge_index_s, y=torch.tensor(0.1))
    gt = Data(x_t, edge_index_t, y=torch.tensor(0.3))
    data_list = [gt, gs]

    loader = DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))

    model = NodePredictor([[0, 55], ], [8, ], 2)
    print(model(batch).shape)
