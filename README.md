# DGL Implementation of DeeperGCN

This DGL example implements the GNN model proposed in the paper [DeeperGCN: All You Need to Train Deeper GCNs](https://arxiv.org/abs/1905.00067). For the original implementation, see [here](https://github.com/lightaime/deep_gcns_torch).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0.post1
torch 1.7.0
ogb 1.3.0
```

### The graph datasets used in this example

Open Graph Benchmark(OGB). Dataset summary:

###### Node Property Prediction

| Dataset | #Nodes | #Edges | #Node Feats | #Edge Feats | #Labels | #Metric |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ogbn-proteins | 132,534 | 39,561,252 | 8(one-hot) | 8 | 2 | ROC-AUC |
| ogbn-arxiv | 169,343 | 1,166,243 | 128 | - | 40 | Accuracy |

###### Graph Property Prediction

| Dataset | #Graphs |
| :-: | :-: |
| ogbg-ppa | 158,100 |
| ogbg-molhiv | 41,127 |

### Architecture

* PlainGCN: GCN + Norm + ReLU
* ResGCN: GCN + Norm + ReLU + Addition
* ResGCN+: Norm + ReLU + GCN + Addition
* ResGEN: Norm + ReLU + GEN + Addition
* DyResGEN: Norm + ReLU + DyGEN + Addition

* ogbn-proteins
* ogbn-arxiv
    - a 28-layer ResGEN model with softmax aggregator
    - beta is fixed as 0.1
    - using batch normalization
    - hidden size is 128
    - dropout is 0.5
    - using Adam with lr(0.01) and epochs(500)
* ogbg-ppa
* ogbg-molhiv

### Usage

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'ogbn-arxiv'.
--self-loop                Add self-loop or not.               
```

###### Training options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
--epochs           int     Number of epochs to train.          Default is 500.
--lr               float   Learning rate.                      Default is 0.01.
--dropout          float   Dropout rate.                       Default is 0.5.
```

###### Model options
```
--num-layers       int     Number of GNN layers.                  Default is 3.
--mlp-layers       int     Number of MLP layers.                  Default is 1.
--hid-dim          int     Hidden channel size.                   Default is 128.
--block            str     Graph backbone block type.             Default is 'res+'.
--conv-type        str     GCNs type.                             Default is 'gen'.
--aggr             str     Type of GENConv aggregator.            Default is 'softmax'.
--aggr             str     Type of GENConv aggregator.            Default is 'softmax'.
--norm             str     Type of GENConv aggregator.            Default is 'softmax'.
--beta             float   The temperature of softmax aggregator. Default is 1.0.
--p                float   The power of power-mean aggregator.    Default is 1.0.
--learn-beta               Whether beta is a learnable weight or not.
--learn-p                  Whether p is a learnable weight or not.
--msg-norm                 Add Message Norm or not.
--learn-msg-scale          Whether s is a learnable weight or not.
```

###### Examples

Train a model which follows the original hyperparameters on different datasets.
```bash
# ogbn-arxiv
python main.py --gpu 0 --self-loop --num-layers 28 --block res+ --aggr softmax --beta 0.1
```

### Performance

| Dataset | ogbn-proteins | ogbn-arxiv | ogbg-ppa | ogbg-molhiv |
| :-: | :-: | :-: | :-: | :-: |
| Results(Table 6) | 0.858±0.0017 | 0.719±0.0016 | 0.771±0.0071 | 0.786±0.0117 |
| Results(DGL) |  |  |  |
