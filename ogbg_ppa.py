import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import dgl

from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader
from ogb.graphproppred import Evaluator
from models import DeeperGCN


def train(model, device, data_loader, opt):
    model.train()
    
    train_loss = []
    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(device)
        logits = model(g, g.edata['feat'])
        loss = F.nll_loss(logits, labels.squeeze(1))
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader, evaluator):
    model.eval()
    y_true, y_pred = [], []

    for g, labels in data_loader:
        g = g.to(device)
        logits = model(g, g.edata['feat'])
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.argmax(dim=-1, keepdim=True).detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred
    })['acc']


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load ogb dataset & evaluator
    dataset = DglGraphPropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    g, _ = dataset[0]
    edge_feat_dim = g.edata['feat'].size()[-1]
    n_classes = int(dataset.num_classes)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]],
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]],
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_dgl)

    # load model
    model = DeeperGCN(dataset=args.dataset,
                      node_feat_dim=edge_feat_dim,
                      edge_feat_dim=edge_feat_dim,
                      hid_dim=args.hid_dim,
                      out_dim=n_classes,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      norm=args.norm,
                      beta=args.beta,
                      mlp_layers=args.mlp_layers).to(device)

    print(model)
    
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # training & validation & testing
    best_acc = 0
    best_model = copy.deepcopy(model)

    print('---------- Training ----------')
    for i in range(args.epochs):
        train_loss = train(model, device, train_loader, opt)

        if i % args.eval_steps == 0:
            train_acc = test(model, device, train_loader, evaluator)
            valid_acc = test(model, device, valid_loader, evaluator)

            print(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = copy.deepcopy(model)
        else:
            print(f'Epoch {i} | Train Loss: {train_loss:.4f}')
    
    print('---------- Testing ----------')
    test_acc = test(best_model, device, test_loader, evaluator)
    print(f'Test Acc: {test_acc}')


if __name__ == '__main__':
    """
    DeeperGCN Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbg-ppa', help='Name of OGB dataset.')
    # training
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--eval-steps', type=int, default=5, help='The interval of evaluation.')
    # model
    parser.add_argument('--num-layers', type=int, default=18, help='Number of GNN layers.')
    parser.add_argument('--hid-dim', type=int, default=128, help='Hidden channel size.')
    parser.add_argument('--norm', type=str, default='layer', help='Type of norm layer.', choices=['batch', 'layer', 'instance'])
    parser.add_argument('--beta', type=float, default=0.01, help='Inverse temperature.')
    parser.add_argument('--mlp-layers', type=int, default=2, help='Number of MLP layers.')

    args = parser.parse_args()
    print(args)

    main()
