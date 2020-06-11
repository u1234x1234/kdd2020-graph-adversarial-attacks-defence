# %%
import pickle
import numpy as np
from functools import partial
from data_utils import get_validation_split
from uxils.graph.node_classifier import ConvolutionalNodeClassifier, create_graph
from dgl.nn import pytorch as dgl_layers
from torch_geometric import nn as pyg_layers


with open('data/experimental_adj.pkl', 'rb') as in_file:
    adj = pickle.load(in_file)
with open('data/experimental_features.pkl', 'rb') as in_file:
    features = pickle.load(in_file)
with open('data/experimental_train.pkl', 'rb') as in_file:
    labels = pickle.load(in_file)

n_classes = np.max(labels) + 1
conv_class = partial(pyg_layers.ChebConv, K=4)
model = ConvolutionalNodeClassifier(
    n_classes=n_classes, conv_class=conv_class, hidden_size=64, n_layers=1,
)

train_idx, val_idx = get_validation_split(len(labels))
labels = np.pad(labels, (0, len(features) - len(labels)), constant_values=-1)

g = create_graph(features, labels, adj)

for i in range(10):
    model.fit(g, train_idx, n_epochs=20)
    r = model.predict(g, val_idx)
    acc = (r.argmax(axis=1) == labels[val_idx]).mean()
    print(acc)
