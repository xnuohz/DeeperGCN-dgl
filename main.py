import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from ogb.nodeproppred import Evaluator
from deepergcn.arxiv import DeeperGCN


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load ogb dataset & evaluator
    dataset = DglNodePropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    g, labels = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    g = g.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)

    in_feats = g.ndata['feat'].size()[-1]
    n_classes = (labels.max() + 1).item()

    # load model
    model = DeeperGCN(in_dim=in_feats,
                      hid_dim=args.hid_dim,
                      out_dim=n_classes,
                      conv_type=args.conv_type,
                      aggr=args.aggr,
                      num_layers=args.num_layers,
                      beta=args.beta, learn_beta=args.learn_beta,
                      p=args.p, learn_p=args.learn_p,
                      msg_norm=args.msg_norm, learn_msg_scale=args.learn_msg_scale,
                      norm=args.norm,
                      activation='relu',
                      mlp_layers=args.mlp_layers,
                      dropout=args.dropout,
                      block=args.block).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # training & validation & testing
    best_acc = 0
    best_model = copy.deepcopy(model)

    for i in range(args.epochs):
        model.train()
        logits = model(g, g.ndata['feat'])
        y_pred = logits.argmax(dim=-1, keepdim=True)
        train_loss = F.nll_loss(logits[train_idx], labels.squeeze(1)[train_idx])
        
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            valid_acc = evaluator.eval({
                'y_true': labels[valid_idx],
                'y_pred': y_pred[valid_idx]
            })['acc']

            print(f'Epoch {i} | Train Loss: {train_loss:.4f} | Valid Acc: {valid_acc:.4f}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = copy.deepcopy(model)

    best_model.eval()
    with torch.no_grad():
        logits = best_model(g, g.ndata['feat'])
        y_pred = logits.argmax(dim=-1, keepdim=True)
        test_acc = evaluator.eval({
            'y_true': labels[test_idx],
            'y_pred': y_pred[test_idx]
        })['acc']
        print(f'Test Acc: {test_acc}')


if __name__ == '__main__':
    """
    DeepGCNs & DeeperGCN Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DeepGCNs & DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Name of OGB dataset.')
    parser.add_argument('--self-loop', action='store_true', help='Add self-loop or not.')
    # training
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    # model
    parser.add_argument('--num-layers', type=int, default=3, help='Number of graph convolutional layers.')
    parser.add_argument('--mlp-layers', type=int, default=1, help='Number of MLP layers.')
    parser.add_argument('--hid-dim', type=int, default=128, help='Hidden channel size.')
    parser.add_argument('--block', type=str, default='res+', help='Graph backbone block type {res+, res, dense, plain}.')
    parser.add_argument('--conv-type', type=str, default='gen', help='GCNs type.')
    parser.add_argument('--aggr', type=str, default='softmax', help='Type of GENConv aggregator {mean, max, sum, softmax, power}.')
    parser.add_argument('--norm', type=str, default='batch', help='Type of normalization layer.')
    # learnable parameters in aggr
    parser.add_argument('--beta', type=float, default=1.0, help='The temperature of softmax aggregator.')
    parser.add_argument('--p', type=float, default=1.0, help='The power of power-mean aggregator.')
    parser.add_argument('--learn-beta', action='store_true')
    parser.add_argument('--learn-p', action='store_true')
    # message norm
    parser.add_argument('--msg-norm', action='store_true')
    parser.add_argument('--learn-msg-scale', action='store_true')

    args = parser.parse_args()
    print(args)

    main()
