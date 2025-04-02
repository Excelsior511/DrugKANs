import torch
import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(adj, drugs_feature, targets_feature):
    pos_data = [[drugs_feature[i], targets_feature[j]] for i in range(len(adj)) for j in range(len(adj[0])) if adj[i][j] == 1]

    neg_data = []
    while len(neg_data) < len(pos_data):
        x = np.random.randint(len(drugs_feature))
        y = np.random.randint(len(targets_feature))
        if adj[x][y] == 0:
            neg_data.append([drugs_feature[x], targets_feature[y]])

    return pos_data, neg_data


def split_data(pos_data, neg_data, train_ratio=0.8, val_ratio=0., test_ratio=0.2):
    data = pos_data + neg_data
    labels = [1] * len(pos_data) + [0] * len(neg_data)

    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data, val_labels, test_labels = temp_data, temp_data, temp_labels, temp_labels

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def data_to_tensor(data):
    drug = []
    target = []
    for x, y in data:
        drug.append(x)
        target.append(y)
    return torch.Tensor(drug).cuda(), torch.Tensor(target).cuda()
