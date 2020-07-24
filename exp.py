# %%
import pickle
import sys
from collections import Counter, defaultdict
from functools import partial

import joblib
import numpy as np
from dgl.nn import pytorch as dgl_layers
from scipy import sparse
from scipy.special import softmax
from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric import nn as pyg_layers

from data_utils import get_validation_split
from utils import load
from uxils.automl.model_specific import optimize_graph_node_classifier
from uxils.graph.node_classifier import (ConvolutionalNodeClassifier,
                                         create_graph)
from uxils.random_ext import fix_seed
from uxils.timer import Timer

fix_seed()
adj, features, labels = load(0)
cv = [get_validation_split(len(labels), test_size=0.01)]
train_idx, val_idx = cv[0]

# %%

# aadj = adj.tocoo()
# dd = defaultdict(list)
# for f, t in zip(aadj.row, aadj.col):
#     if f == 22 or t == 22:
#         print(f, t)
#     dd[f].append(t)


nlabels = np.pad(labels, (0, adj.shape[0] - len(labels)), constant_values=0)
# for v in val_idx:  # mask validation labels
    # nlabels[v] = 0

# # %%
# print('classes', Counter(labels).most_common())

# x_per_class = defaultdict(list)
# for idx, l in enumerate(labels):
#     x_per_class[l].append(idx)

# X_cl = np.array([features[idx] for idx in x_per_class[6]])
# km = KMeans(n_clusters=10)
# km.fit(X_cl)
# print(km.cluster_centers_.shape)

# # %%
# items = list(d.items())
# np.random.shuffle(items)
# idx = 0

# counter = Counter()

# nei_to_root = defaultdict(list)
# for u, vs in items:
#     u_label = nlabels[u]
#     v_labels = [nlabels[idx] for idx in vs]
#     nei_to_root[tuple(v_labels)].append(u_label)
# print(f'unique nei {len(nei_to_root)}')

# for u, vs in items:
#     v_labels = [nlabels[idx] for idx in vs]
#     u_label = nlabels[u]

#     if len(v_labels) != 2:
#         continue

#     counter[tuple(v_labels)] += 1
#     # print(u_label, v_labels)
#     idx += 1
#     # if idx == 30:
#         # break

# print(idx)
# for x, cnt in counter.most_common(30):
#     print(x, cnt)

# # %%

# s = nei_to_root[(6, 6)]
# print(Counter(s).most_common())

# %%
# optimize_graph_node_classifier('accuracy', features, nlabels, adj, cv)
# qwe

def func(features, labels, adj):
    n_classes = np.max(labels) + 1
    # conv_class = partial(pyg_layers.ChebConv, K=7)
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


def create_adj(adj, test_idxs):
    adj = adj.copy()
    add_adj = sparse.dok_matrix((500, 659574+500), dtype=np.int64)

    ####################################

    idxs = []
    for i in test_idxs:
        r = adj[i].getnnz()
        idxs.append((i, r))

    idxs = list(sorted(idxs, key=lambda x: x[1]))[:50000//5]
    # idxs = list(filter(lambda x: nlabels[x[0]] == 6, idxs))
    # print(len(idxs))

    for i, (idx, r) in enumerate(idxs[:-500]):
        for j in range(5):
            row = (i+j) % 500
            add_adj[row, idx] = 1

    for i in range(500):
        add_adj[i, i] = 1

    add_adj = add_adj.tocsr()

    ####################################

    # assert add_adj.getnnz(axis=1).max() <= 100
    # assert (add_adj[:, 659574:].transpose(copy=True).todense() == add_adj[:, 659574:].todense()).all()

    nadj = adj.todok()
    return
    nadj.resize((659574+500, 659574+500))
    cx = add_adj.tocoo()

    for i, j, v in zip(cx.row, cx.col, cx.data):
        nadj[659574+i, j] = v
        nadj[j, i+659574] = v

    nadj = nadj.tocsr()
    nadj = nadj.astype(np.int64)

    # print(nadj[29].nonzero(), adj[29].nonzero())
    return nadj, add_adj


def evaluate_attacker(features, adj):
    model = joblib.load('model.pkl')
    g = create_graph(features, np.zeros(len(features)), adj)
    predictions = model.predict(g, val_idx).argmax(axis=1)
    score = (predictions == labels[val_idx]).mean()
    return score


# nlabels = np.pad(labels, (0, adj.shape[0] - len(labels)), constant_values=-1)
# score = func(features, nlabels, adj)
# print(score)
# qwe

# nfeatures = km.cluster_centers_[:2]
# nfeatures = np.array(X_cl[:2])
# nfeatures = np.repeat(nfeatures, 250, axis=0)
# assert nfeatures.shape == (500, 100)

# test_idx = list(range(len(labels), adj.shape[0]))

with Timer():
    nadj, add_adj = create_adj(adj, val_idx)

qwe
print(add_adj.shape, add_adj.dtype)

# with open('adj.pkl', 'wb') as out_file:
    # pickle.dump(add_adj, out_file)

print('NNZ diff', nadj.getnnz() - adj.getnnz())
nlabels = np.pad(labels, (0, nadj.shape[0] - len(labels)), constant_values=-1)

# for _ in range(10):
    # nfeatures = np.random.uniform(-200, 200, size=(500, 100)).astype(np.float32)
nfeatures = np.random.choice([-1.9], size=(500, 100)).astype(np.float32)
nfeatures = np.vstack((features, nfeatures))
print(evaluate_attacker(nfeatures, nadj))
