from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset


def main():
    dataset = DglNodePropPredDataset(name='ogbn-proteins')
    print(dataset)


if __name__ == '__main__':
    main()
