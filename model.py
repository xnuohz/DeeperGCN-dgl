import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.functional import edge_softmax

class MLP(nn.Sequential):
    def __init__(self, channels, norm=None, dropout=0., bias=True):
        layers = []
        
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                if norm == 'batch':
                    layers.append(nn.BatchNorm1d(channels[i], affine=True))
                elif norm == 'layer':
                    layers.append(nn.LayerNorm(channels[i], elementwise_affine=True))
                elif norm == 'instance':
                    layers.append(nn.InstanceNorm1d(channels[i], affine=False))
                else:
                    raise NotImplementedError(f'Normalization layer {norm} is not supported.')
        
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        super(MLP, self).__init__(*layers)


class MessageNorm(nn.Module):
    r"""
    
    Description
    -----------
    Message normalization was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """
    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=learn_scale)
    
    def reset_parameters(self):
        self.scale.data.fill_(1.0)

    def forward(self, feats, msg):
        feats = feats.norm(p=2, dim=-1, keepdim=True)
        msg = F.normalize(msg, p=2, dim=-1)
        return msg * feats * self.scale


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
        Type of aggregator scheme ('softmax', 'softmax_sg', 'power', 'sum', 'mean', 'max'), default is 'softmax'.
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
    num_layers: int
        The number of MLP layers. Default is 2.
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
                 num_layers=2,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for i in range(num_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels, norm=norm)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

    def reset_parameters(self):
        pass

    def msg_fn(self, edge):
        return {'x': edge.data['m'] * edge.data['a']}

    def forward(self, g, feats):
        # Node and edge feature dimension need to match.
        assert feats.size()[-1] == g.edata['h'].size()[-1]

        with g.local_scope():
            if self.aggr == 'softmax':
                g.ndata['h'] = feats
                g.apply_edges(fn.u_add_e('h', 'h', 'm'))
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(self.msg_fn, fn.sum('x', 'h'))

                if self.msg_norm is not None:
                    feats = feats + self.msg_norm(feats, g.ndata['h'])
            elif self.aggr == 'power':
                pass
            elif self.aggr == 'sum':
                pass
            elif self.aggr == 'max':
                pass
            elif self.aggr == 'mean':
                pass
            else:
                raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')
            
            return self.mlp(feats)


class DeepGCNLayer(nn.Module):
    r"""

    Description
    -----------
    Graph convolution was introduced in:

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
        super(DeepGCNLayer, self).__init__()

        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.block = block
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, g, feats):
        h = feats
        if self.block == 'res+':
            if self.norm is not None:
                h = self.norm(h)
            if self.activation is not None:
                h = self.activation(h)
            h = self.dropout(h)
            if self.conv is not None:
                h = self.conv(g, h)
            return feats + h
        else:
            if self.conv is not None:
                h = self.conv(g, h)
            if self.norm is not None:
                h = self.norm(h)
            if self.activation is not None:
                h = self.activation(h)

            if self.block == 'res':
                h = feats + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass
                
            return self.dropout(h)
            