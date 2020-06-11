from sklearn.model_selection import train_test_split
import numpy as np


def get_validation_split(n, test_size=0.1):
    train_idx, val_idx = train_test_split(range(n), random_state=1, shuffle=True, test_size=test_size)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    return train_idx, val_idx
