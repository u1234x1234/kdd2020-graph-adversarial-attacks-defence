import pickle

import numpy as np


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
