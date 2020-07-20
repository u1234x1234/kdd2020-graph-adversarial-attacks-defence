# %%
import pickle
from functools import partial
import joblib

import numpy as np
from dgl.nn import pytorch as dgl_layers
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric import nn as pyg_layers

from data_utils import get_validation_split
from uxils.automl.model_specific import optimize_graph_node_classifier
from uxils.graph.feature_extraction import pooled_features
from uxils.graph.node_classifier import (ConvolutionalNodeClassifier,
                                         create_graph)
from uxils.profiling import Profiler
from uxils.timer import Timer
from sklearn import ensemble


def load(d):
    if d:
        with open('data/experimental_adj.pkl', 'rb') as in_file:
            adj = pickle.load(in_file)
        with open('data/experimental_features.pkl', 'rb') as in_file:
            features = pickle.load(in_file)
        with open('data/experimental_train.pkl', 'rb') as in_file:
            labels = pickle.load(in_file)
    else:
        with open('data/kdd_cup_phase_two/adj_matrix_formal_stage.pkl', 'rb') as in_file:
            adj = pickle.load(in_file)
        features = np.load('data/kdd_cup_phase_two/feature_formal_stage.npy')
        labels = np.load('data/kdd_cup_phase_two/train_labels_formal_stage.npy')

    return adj, features, labels


adj, features, labels = load(0)

# X = pooled_features(features, adj)
X = features
y = labels
X = X[:len(labels)]
print(X.shape, len(y))


# features = StandardScaler().fit_transform(features)

cv = [get_validation_split(len(labels), test_size=0.1)]

# %%

# optimize_graph_node_classifier('accuracy', features, labels, adj, cv)
from scipy import sparse
from uxils.random_ext import fix_seed
fix_seed()


def func(features, labels, adj):
    n_classes = np.max(labels) + 1
    conv_class = partial(dgl_layers.TAGConv, k=3)
    model = ConvolutionalNodeClassifier(
        n_classes=n_classes, conv_class=conv_class, hidden_size=40, n_layers=3, lr=0.01,
        in_normalization='bn', hidden_normalization=None, wd=0.01,
        optimizer='adamw', activation='tanh'
    )
    train_idx, val_idx = cv[0]
    g = create_graph(features, labels, adj)

    for i in range(10):
        model.fit(g, train_idx, n_epochs=20)

        r = model.predict(g, val_idx)
        acc = (r.argmax(axis=1) == labels[val_idx]).mean()
        print(acc)

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

    idxs = list(sorted(idxs, key=lambda x: x[1]))[:10000]

    for i, (idx, r) in enumerate(idxs):
        for j in range(5):
            row = (i+j) % 500
            add_adj[row, idx] = 1

    add_adj = add_adj.tocsr()
    print(add_adj.getnnz(axis=1))

    ####################################

    assert add_adj.getnnz(axis=1).max() <= 100
    assert (add_adj[:, 659574:].transpose(copy=True).todense() == add_adj[:, 659574:].todense()).all()

    nadj = adj.copy().todok()
    nadj.resize((659574+500, 659574+500))
    cx = add_adj.tocoo()

    for i, j, v in zip(cx.row, cx.col, cx.data):
        nadj[659574+i, j] = v
        nadj[j, i+659574] = v

    nadj = nadj.tocsr()
    nadj = nadj.astype(np.int64)
    return nadj, add_adj


nlabels = np.pad(labels, (0, adj.shape[0] - len(labels)), constant_values=-1)
score = func(features, nlabels, adj)
print(score)
qwe

val_idx = cv[0][1]
# test_idx = list(range(len(labels), adj.shape[0]))
# print('test', test_idx[:3], len(test_idx))

nadj, add_adj = create_adj(adj, val_idx)
print(add_adj.shape, add_adj.dtype)

# with open('adj.pkl', 'wb') as out_file:
    # pickle.dump(add_adj, out_file)

print('NNZ diff', nadj.getnnz() - adj.getnnz())
nlabels = np.pad(labels, (0, nadj.shape[0] - len(labels)), constant_values=-1)
# nfeatures = np.random.uniform(-2, 2, size=(500, 100)).astype(np.float32)
nfeatures = np.random.choice([-1.9], size=(500, 100)).astype(np.float32)
# np.save('feature.npy', nfeatures)

# score = func(np.vstack((features, nfeatures)), nlabels, nadj)


qwe


# %%
# print(nfeatures.shape, len(nlabels), nadj.shape)
import sys
sys.path.append('/home/u1234x1234/libs/pygmo2/build/')

import pygmo as pg


def obj(w):
    w = w.reshape((500, 100))
    nfeatures = np.vstack((features, w))
    score = func(nfeatures, nlabels, nadj)
    return -score


class problem:
    def __init__(self):
        self.dim = 100*500

    def fitness(self, x):
        score = obj(x)
        print(score)
        return [score]

    def get_bounds(self):
        return ([-2]*self.dim, [2] * self.dim)


prob = pg.problem(problem())
print(prob)

algo = pg.algorithm(pg.sga(gen=10))
algo.set_verbosity(5)
print(algo)

pop = pg.population(prob, 20)
pop = algo.evolve(pop)
print(pop.champion_f) 
qwe


w = np.random.normal(0, 1, size=(500, 100)).flatten()
alpha = 0.5
import noisyopt

noisyopt.minimizeSPSA(obj, w, paired=False)

qwe

for it in range(100):

    delta = np.random.uniform(-0.2, 0.2, size=w.shape)
    w[w > 2] = 2
    w[w < -2] = -2

    score1 = func(np.vstack((features, w+delta)), nlabels, nadj)
    score2 = func(np.vstack((features, w-delta)), nlabels, nadj)

    grad = (score1 - score2) / (2. * delta)
    w = w - alpha * grad

    print(score1, score2)

# Train on FUll
