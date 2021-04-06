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
| ogbn-proteins | 132,534 | 39,561,252 | - | 8 | 2 | ROC-AUC |
| ogbn-arxiv | 169,343 | 1,166,243 | 128 | - | 40 | Accuracy |

###### Graph Property Prediction

| Dataset | #Graphs | #Node Feats | #Edge Feats | Metric |
| :-: | :-: | :-: | :-: | :-: |
| ogbg-ppa | 158,100 | - | 7 | Accuracy |
| ogbg-molhiv | 41,127 | 9 | 3 | ROC-AUC |

### Architecture

* PlainGCN: GCN + Norm + ReLU
* ResGCN: GCN + Norm + ReLU + Addition
* ResGCN+: Norm + ReLU + GCN + Addition
* ResGEN: Norm + ReLU + GEN + Addition
* DyResGEN: Norm + ReLU + DyGEN + Addition

* ogbn-proteins
  - node features is initialized via a sum aggregation of their connected edges
  - a 112-layer DyResGEN with softmax aggregator
  - using layer normalization
  - hidden size is 64
  - dropout is 0.1
  - using Adam with lr(0.01) and epochs(1000)
* ogbn-arxiv
  - a 28-layer ResGEN with softmax aggregator
  - beta is fixed as 0.1
  - using batch normalization
  - hidden size is 128
  - dropout is 0.5
  - using Adam with lr(0.01) and epochs(500)
* ogbg-ppa
  - node features is initialized via a sum aggregation of their connected edges
  - a 28-layer ResGEN with softmax aggregator
  - beta is fixed as 0.1
  - using layer normalization
  - hidden size is 128
  - dropout is 0.5
  - using Adam with lr(0.01) and epochs(200)
* ogbg-molhiv
  - a 7-layer DyResGEN with softmax aggregator
  - beta is learnable
  - using batch normalization
  - hidden size is 256
  - dropout is 0.5
  - using Adam with lr(0.01) and epochs(300)

### Usage

Train a model which follows the original hyperparameters on different datasets.
```bash
# ogbg-molhiv
python ogbg_molhiv.py --gpu 0 --learn-beta --batch-size 2048 --dropout 0.2
# ogbg-ppa
python ogbg_ppa.py --gpu 0
```

### Performance

| Dataset | ogbn-proteins | ogbn-arxiv | ogbg-ppa | ogbg-molhiv |
| :-: | :-: | :-: | :-: | :-: |
| Results(Table 6) | 0.858±0.0017 | 0.719±0.0016 | 0.771±0.0071 | 0.786±0.0117 |
| Results(Author) |  |  |  | 0.781 |
| Results(DGL) |  |  |  | 0.778 |
