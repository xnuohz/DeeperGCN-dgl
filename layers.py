import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.functional import edge_softmax
from modules import MLP, MessageNorm


class GENConv(nn.Module):
    r"""
    
    Description
    -----------
    Generalized Message Aggregator was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    in_dim: int
        Size of input dimension.
    out_dim: int
        Size of output dimension.
    aggregator: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 learn_beta=False,
                 p=1.0,
                 learn_p=False,
                 msg_norm=False,
                 learn_msg_scale=False,
                 norm='batch',
                 mlp_layers=1,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for i in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels, norm=norm)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

    def forward(self, g, node_feats, edge_feats=None):
        with g.local_scope():
            g.ndata['h'] = node_feats

            if edge_feats is not None:
                # Node and edge feature dimension need to match.
                g.edata['h'] = edge_feats
                g.apply_edges(fn.u_add_e('h', 'h', 'm'))
            else:
                g.apply_edges(fn.copy_u('h', 'm'))

            if self.aggr == 'softmax':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            
            elif self.aggr == 'power':
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata['m'], minv, maxv)
                g.update_all(lambda edge: {'x': torch.pow(edge.data['m'], self.p)},
                             fn.mean('x', 'm'))
                torch.clamp_(g.ndata['m'], minv, maxv)
                g.ndata['m'] = torch.pow(g.ndata['m'], self.p)
            
            else:
                raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')
            
            feats = g.ndata['h']
            if self.msg_norm is not None:
                feats = feats + self.msg_norm(g.ndata['h'], g.ndata['m'])
            
            return self.mlp(feats)


class DeeperGCNLayer(nn.Module):
    r"""

    Description
    -----------
    Graph convolution architecture was introduced in:

    - `DeepGCNs: Can GCNs Go as Deep as CNNs? <https://arxiv.org/abs/1904.03751>`_,
    - `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    conv: torch.nn.Module
        The graph convolutional layer.
    norm: torch.nn.Module
        The normalization layer.
    activation: torch.nn.Module
        The activation function.
    block: str, optional
        The skip connection operation to use. 
        You can chose from set ['plain', 'dense', 'res', 'res+'], default is 'res+'.
    dropout: float, optional
        Whether to apply dropout, default is 0.
    """
    def __init__(self, conv=None, norm=None, activation=None, block='res+', dropout=0.):
        super(DeeperGCNLayer, self).__init__()

        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.block = block.lower()
        assert self.block in ['plain', 'dense', 'res', 'res+']
        self.dropout = nn.Dropout(dropout)
        self.norm.reset_parameters()

    def forward(self, g, node_feats, edge_feats=None):
        h = node_feats
        if self.block == 'res+':
            if self.norm is not None:
                h = self.norm(h)
            if self.activation is not None:
                h = self.activation(h)
            h = self.dropout(h)
            if self.conv is not None:
                h = self.conv(g, h, edge_feats)
            return node_feats + h
        else:
            if self.conv is not None:
                h = self.conv(g, h, edge_feats)
            if self.norm is not None:
                h = self.norm(h)
            if self.activation is not None:
                h = self.activation(h)

            if self.block == 'res':
                h = node_feats + h
            elif self.block == 'dense':
                h = torch.cat([node_feats, h], dim=-1)
            elif self.block == 'plain':
                pass
                
            return self.dropout(h)
