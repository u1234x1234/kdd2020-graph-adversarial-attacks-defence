from functools import partial

import joblib
import numpy as np
from dgl.nn import pytorch as dgl_layers
from scipy import sparse

from data_utils import get_validation_split
from utils import load
from defender.uxils.graph.node_classifier import (ConvolutionalNodeClassifier,
                                                  create_graph)

adj, features, labels = load()
cv = [get_validation_split(len(labels), test_size=0.01)]
train_idx, val_idx = cv[0]


def train_defender(features, labels, adj):
    n_classes = np.max(labels) + 1
    conv_class = partial(dgl_layers.SGConv, k=4)
    model = ConvolutionalNodeClassifier(
        n_classes=n_classes, conv_class=conv_class, n_hiddens=[140, 120, 100], lr=0.01,
        in_normalization='bn', hidden_normalization='ln', wd=0,
        optimizer='adamw', activation='tanh', criterion='ce'
    )
    g = create_graph(features, labels, adj)

    for i in range(30):
        model.fit(g, train_idx, n_epochs=20)
        r = model.predict(g, val_idx)
        acc = (r.argmax(axis=1) == labels[val_idx]).mean()
        print(i, acc)

    joblib.dump(model, 'model.pkl')

    return acc


def create_attacker(adj, test_idxs):
    adj = adj.copy()
    add_adj = sparse.dok_matrix((500, 659574+500), dtype=np.int64)

    ####################################

    idxs = []
    for i in test_idxs:
        r = adj[i].getnnz()
        idxs.append((i, r))

    idxs = list(sorted(idxs, key=lambda x: x[1]))[:50000 // 5]

    for i, (idx, r) in enumerate(idxs[:-500]):
        for j in range(5):
            row = (i+j) % 500
            add_adj[row, idx] = 1

    for i in range(500):
        add_adj[i, i] = 1

    add_adj = add_adj.tocsr()

    ####################################

    assert add_adj.getnnz(axis=1).max() <= 100
    assert (add_adj[:, 659574:].transpose(copy=True).todense() == add_adj[:, 659574:].todense()).all()

    return add_adj


if __name__ == '__main__':
    # Defender
    nlabels = np.pad(labels, (0, adj.shape[0] - len(labels)), constant_values=-1)
    train_defender(features, nlabels, adj)

    # Attacker
    add_adj = create_attacker(adj, np.arange(len(labels), adj.shape[0]))
    joblib.dump(add_adj, 'adj.pkl')

    nfeatures = np.random.choice([-1.9], size=(500, 100)).astype(np.float32)
    np.save('features.npy', nfeatures)
