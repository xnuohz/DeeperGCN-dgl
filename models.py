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
    in_dim: int
        Size of input dimension.
    hid_dim: int
        Size of hidden dimension.
    out_dim: int
        Size of output dimension.
    conv_type: str
        Type of graph convolution.
    aggr: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    num_layers: int
        Number of graph convolutional layers.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is True.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    activation: str
        Activation function of graph convolutional layer.
    mlp_layers: int
        The number of MLP layers. Default is 2.
    dropout: float
        Dropout rate. Default is 0.
    block: str
        The skip connection operation to use.
        You can chose from set ['plain', 'dense', 'res', 'res+'], default is 'res+'.
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 conv_type,
                 aggr,
                 num_layers,
                 beta=1.0, learn_beta=False,
                 p=1.0, learn_p=False,
                 msg_norm=True, learn_msg_scale=False,
                 norm='batch',
                 activation='relu',
                 mlp_layers=1,
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
