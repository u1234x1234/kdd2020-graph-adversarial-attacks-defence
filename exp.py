# %%
import pickle


# with open('data/experimental_adj.pkl', 'rb') as in_file:
    # adj = pickle.load(in_file)
with open('data/experimental_features.pkl', 'rb') as in_file:
    features = pickle.load(in_file)
with open('data/experimental_train.pkl', 'rb') as in_file:
    labels = pickle.load(in_file)


print(labels)
