import os
import pickle as pkl
import sys
import time

import joblib
import numpy as np
import torch
from dgl import DGLGraph


class Graph:
    def __init__(self, pyg_data=None, dgl_graph=None):
        self.pyg_data = pyg_data
        self.dgl_graph = dgl_graph


class Item:
    def __init__(self, x):
        self.x = x


def from_scipy_sparse_matrix(A):
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    # edge_weight = torch.from_numpy(A.data)
    return edge_index


def create_graph(node_features, edges, edge_attr=None):

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index = from_scipy_sparse_matrix(edges)

    assert len(edge_index) == 2
    dgl_graph = DGLGraph((edge_index[0], edge_index[1]))
    dgl_graph = dgl_graph.to(torch.device('cuda'))
    return Graph(Item(node_features.cuda()), dgl_graph)


if __name__ == '__main__':
    sys.path.append('./')
    print(sys.argv)

    st = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    adj = pkl.load(open(sys.argv[1], 'rb'))
    features = np.load(sys.argv[2])

    model = joblib.load('model.pkl')
    g = create_graph(features, adj)

    predictions = model.predict(g).argmax(axis=1)

    assert len(predictions) == len(features)
    with open(sys.argv[3], 'w') as f2:
        for p in predictions:
            print(int(p), file=f2)

    print('Execution time: {}'.format(time.time() - st))
    os.system('head {}'.format(sys.argv[3]))
