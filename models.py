import torch
import torch.nn as nn

from modules import norm_layer, act_layer
from layers import GENConv, DeeperGCNLayer


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
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 conv_type,
                 aggr,
                 num_layers,
                 beta, learn_beta,
                 p, learn_p,
                 msg_norm, learn_msg_scale,
                 norm,
                 activation,
                 mlp_layers,
                 dropout=0.,
                 block='res+'):
        super(DeeperGCN, self).__init__()
        
        self.node_encoder = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList()

        norm = norm_layer(norm, hid_dim)
        act = act_layer(activation, inplace=True)

        for i in range(num_layers):
            if conv_type == 'gen':
                conv = GENConv(in_dim=hid_dim,
                               out_dim=hid_dim,
                               aggregator=aggr,
                               beta=beta, learn_beta=learn_beta,
                               p=p, learn_p=learn_p,
                               msg_norm=msg_norm, learn_msg_scale=learn_msg_scale,
                               norm=norm,
                               mlp_layers=mlp_layers)
            else:
                raise NotImplementedError(f'Conv {conv_type} is not supported.')
            
            self.layers.append(DeeperGCNLayer(conv=conv,
                                              norm=norm,
                                              activation=act,
                                              block=block,
                                              dropout=dropout))
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, node_feats):
        with g.local_scope():
            h = self.node_encoder(node_feats)

            for layer in self.layers:
                h = layer(g, h)
            
            h = self.output(h)

            return torch.log_softmax(h, dim=-1)
