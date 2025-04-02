import tqdm
import torch
import numpy as np
import pandas as pd
from model import my_model
from utils import generate_data, split_data, data_to_tensor
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

dataset_name = "gpcr"
if dataset_name in ["BindingDB", "davis"]:
    adj = pd.read_excel(f"dataset/{dataset_name}/dti_mat.xlsx", header=None).values
    drugs_feature = np.loadtxt(f"dataset/{dataset_name}/drugs_feature.txt", delimiter=",")
    targets_feature = np.loadtxt(f"dataset/{dataset_name}/targets_feature.txt", delimiter=",")
else:   # YAM
    adj = pd.read_csv("dataset/YAM/{}/{}_admat_dgc.txt".format(dataset_name, dataset_name), sep="\t",
                      index_col=0).values.tolist()
    drugs_feature = pd.read_csv("dataset/YAM/{}/{}_simmat_dg.txt".format(dataset_name, dataset_name), sep="\t",
                                index_col=0).values.tolist()
    targets_feature = pd.read_csv("dataset/YAM/{}/{}_simmat_dc.txt".format(dataset_name, dataset_name), sep="\t",
                                  index_col=0).values.tolist()

pos_data, neg_data = generate_data(adj, drugs_feature, targets_feature)
(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = split_data(pos_data, neg_data)
train_drug, train_target = data_to_tensor(train_data)
val_drug, val_target = data_to_tensor(val_data)
test_drug, test_target = data_to_tensor(test_data)
train_labels = torch.Tensor(train_labels).cuda()
val_labels = torch.Tensor(val_labels).cuda()
test_labels = torch.Tensor(test_labels).cuda()

model = my_model(len(train_drug[0]), len(train_target[0])).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss().cuda()

min_val_loss = float('inf')
best_AUC = float('-inf')
early_stop = 100
count = 0
model.train()
for epoch in range(10000):
    count += 1
    opt.zero_grad()
    out = model(train_drug, train_target)
    loss = loss_fn(out.squeeze(-1), train_labels)
    loss.backward()
    opt.step()

    test_out = model(test_drug, test_target)
    preds = (test_out >= 0.5).float()
    test_auc = roc_auc_score(test_labels.cpu().detach().numpy(), test_out.cpu().detach().numpy())
    if test_auc > best_AUC:
        best_AUC = test_auc
        print(epoch, best_AUC)
        count = 0
    else:
        count += 1

