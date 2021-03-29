import argparse
import torch

from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from model import DeeperGCN


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load ogb dataset
    dataset = DglNodePropPredDataset(name=args.dataset)
    g, labels = dataset[0]
    g = g.to(device)
    labels = labels.to(device)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    print(g)
    num_node_feats = g.ndata['feat'].size()[-1]


if __name__ == '__main__':
    """
    DeepGCNs & DeeperGCN Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DeepGCNs & DeeperGCN')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Name of OGB dataset.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
    parser.add_argument('--hid-dim', type=int, default=64, help='Hidden channel size.')
    parser.add_argument('--num-layers', type=int, default=28, help='Number of graph convolutional layers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs.')


    args = parser.parse_args()
    print(args)

    main()
