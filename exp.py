# %%
import pickle
from functools import partial

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


X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
cv = [get_validation_split(len(labels))]

# %%
# import pickle
# with Timer('fit'):
    # model = ensemble.RandomForestClassifier(n_jobs=40, n_estimators=100)
    # model = LogisticRegression(solver='sag', n_jobs=20, max_iter=5)
    # model.fit(X_train, y_train)
# with Timer('predict'):
    # y_pred = model.predict(X_test)
    # print((y_pred == y_test).mean())


labels = np.pad(labels, (0, len(features) - len(labels)), constant_values=-1)
# optimize_graph_node_classifier('accuracy', features, labels, adj, cv)

n_classes = np.max(labels) + 1
conv_class = partial(dgl_layers.TAGConv, k=2)
model = ConvolutionalNodeClassifier(
    n_classes=n_classes, conv_class=conv_class, hidden_size=32, n_layers=1, lr=0.01, normalization=None, wd=0.01,
    optimizer='adamw', activation='tanh',
)

train_idx, val_idx = cv[0]
g = create_graph(features, labels, adj)
import joblib

for i in range(10):
    model.fit(g, train_idx, n_epochs=20)

    joblib.dump(model, 'model.pkl')
    model = joblib.load('model.pkl')

    r = model.predict(g, val_idx)
    acc = (r.argmax(axis=1) == labels[val_idx]).mean()
    print(acc)
