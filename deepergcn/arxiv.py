import torch.nn as nn


class DeeperGCN(nn.Module):
    r"""

    Description
    -----------
    Introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    node_dim: int
        The input dimension of node features.
    edge_dim: int
        The input dimension of edge features.
    hid_dim: int
        Hidden layer dimension.
    out_dim: int
        Output layer dimension.
    num_layers: int
        The number of layers.
    """
    def __init__(self, node_dim, edge_dim, hid_dim, out_dim, num_layers):
        super(DeeperGCN, self).__init__()
        
        self.node_encoder = nn.Linear(node_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_dim, hid_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            conv = GENConv(hid_dim,
                           hid_dim,
                           aggregator='softmax',
                           beta=1.0,
                           learn_beta=True,
                           num_layers=2,
                           norm='layer')
            norm = nn.LayerNorm(hid_dim, elementwise_affine=True)
            activation = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, activation, block='res+', dropout=0.1)
            self.layers.append(layer)
        self.output = nn.Linear(hid_dim, out_dim)
    
    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = self.node_encoder(node_feats)
            g.edata['h'] = self.edge_encoder(edge_feats)

            for layer in self.layers:
                g.ndata['h'] = layer(g)

            h = self.layers[0].activation(self.layers[0].norm(g.ndata['h']))
            h = F.dropout(h, p=0.1)
            return self.output(h)
