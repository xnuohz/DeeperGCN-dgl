PlainGCN: GCN + Norm + ReLU
ResGCN: GCN + Norm + ReLU + Addition
ResGCN+: Norm + ReLU + GCN + Addition
ResGEN: Norm + ReLU + GEN + Addition
DyResGEN: Norm + ReLU + DyGEN + Addition


DeeperGCNLayer
    DeeperGCN
        - GENConv
        - MessageNorm
    DeepGCNs

dataset:
    ogbn-proteins
    ogbn-arxiv
    ogbg-ppa
    ogbg-molhiv