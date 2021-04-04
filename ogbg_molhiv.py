import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader
from ogb.graphproppred import Evaluator
from models import DeeperMolhiv


def train(model, device, data_loader, opt, loss_fn, grad_clip=0.):
    model.train()
    
    train_loss = []
    for g, labels in data_loader:
        g = g.to(device)
        labels = labels.to(torch.float32).to(device)
        logits = model(g, g.ndata['feat'], g.edata['feat'])
        loss = loss_fn(logits, labels)
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
        opt.step()

    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader, evaluator):
    model.eval()
    y_true, y_pred = [], []

    for g, labels in data_loader:
        g = g.to(device)
        logits = model(g, g.ndata['feat'], g.edata['feat'])
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred
    })['rocauc']


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load ogb dataset & evaluator
    dataset = DglGraphPropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    g, _ = dataset[0]
    node_feat_dim = g.ndata['feat'].size()[-1]
    edge_feat_dim = g.edata['feat'].size()[-1]
    n_classes = dataset.num_tasks

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
    model = DeeperMolhiv(node_feat_dim=node_feat_dim,
                         edge_feat_dim=edge_feat_dim,
                         hid_dim=args.hid_dim,
                         out_dim=n_classes,
                         num_layers=args.num_layers,
                         learn_beta=args.learn_beta,
                         msg_norm=args.msg_norm,
                         learn_msg_scale=args.learn_msg_scale,
                         dropout=args.dropout).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # training & validation & testing
    best_auc = 0
    best_model = copy.deepcopy(model)

    print('---------- Training ----------')
    for i in range(args.epochs):
        train_loss = train(model, device, train_loader, opt, loss_fn, args.grad_clip)
        train_auc = test(model, device, train_loader, evaluator)
        valid_auc = test(model, device, valid_loader, evaluator)

        print(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_auc:.4f} | Valid Acc: {valid_auc:.4f}')

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_model = copy.deepcopy(model)
    
    print('---------- Testing ----------')
    test_auc = test(best_model, device, test_loader, evaluator)
    print(f'Test AUC: {test_auc}')


if __name__ == '__main__':
    """
    DeeperGCN Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv', help='Name of OGB dataset.')
    # training
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--grad-clip', type=float, default=0., help='Grad clip.')
    # model
    parser.add_argument('--num-layers', type=int, default=7, help='Number of GNN layers.')
    parser.add_argument('--hid-dim', type=int, default=256, help='Hidden channel size.')
    # learnable parameters in aggr
    parser.add_argument('--learn-beta', action='store_true')
    # message norm
    parser.add_argument('--msg-norm', action='store_true')
    parser.add_argument('--learn-msg-scale', action='store_true')

    args = parser.parse_args()
    print(args)

    main()
