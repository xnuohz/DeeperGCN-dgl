import torch
import torch.nn as nn

from modules import norm_layer, act_layer
from layers import GENConv, DeeperGCNLayer


class DeeperArxiv(nn.Module):
    r"""

    Description
    -----------
    Introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    in_dim: int
        Size of input dimension.
    hid_dim: int
        Size of hidden dimension.
    out_dim: int
        Size of output dimension.
    num_layers: int
        Number of graph convolutional layers.
    activation: str
        Activation function of graph convolutional layer.
    dropout: float
        Dropout rate. Default is 0.
    block: str
        The skip connection operation to use.
        You can chose from set ['plain', 'dense', 'res', 'res+'], default is 'res+'.
    aggr: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    msg_norm: bool
        Whether message normalization is used. Default is True.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 activation='relu',
                 dropout=0.,
                 block='res+',
                 aggr='softmax',
                 beta=1.0,
                 msg_norm=False,
                 learn_msg_scale=False,
                 norm='batch'):
        super(DeeperArxiv, self).__init__()
        
        self.node_encoder = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList()

        norm = norm_layer(norm, hid_dim)
        act = act_layer(activation, inplace=True)

        for i in range(num_layers):
            conv = GENConv(in_dim=hid_dim,
                           out_dim=hid_dim,
                           aggregator=aggr,
                           beta=beta,
                           msg_norm=msg_norm,
                           learn_msg_scale=learn_msg_scale)
            
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
