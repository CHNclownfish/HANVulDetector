import json

from baseline_LGCN.utils_LGCN import contract2runtimbin
from baseline_LGCN.seqGenerator_LGCN import dataGenerator
from baseline_LGCN.L_GCN import LGCNModel
from sklearn.model_selection import KFold
import torch as th
from baseline_LGCN.utils_LGCN import evaluate
from baseline_LGCN.showResults_LGCN import showRes
from baseline_LGCN.utils_LGCN import readJson
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

labels_path = 'files/front_running_trueBalanceForLGCN.json'

info = readJson(labels_path)

dG = dataGenerator(info)
data = dG.get_data()
graph_input_dim = 64
graph_hidden_dim1 = 32
graph_hidden_dim2 = 16
seq_input_dim = 256
seq_hidden_dim = 64
seq_layer_dim = 1
output_dim = 2

np.random.shuffle(data)
#kf = KFold(n_splits=5, random_state=1, shuffle=True)
kf = KFold(n_splits=5)
losses = []
scores = []
cnt = 0
pca = PCA(n_components=2)
print('data prepared')
for train_idx, test_idx in kf.split(data):
    model = LGCNModel(graph_input_dim, graph_hidden_dim1,graph_hidden_dim2, seq_input_dim, seq_hidden_dim, seq_layer_dim, output_dim)
    loss_fcn= th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.002,
                              weight_decay=0.001)
    epoch_losses = []
    cnt += 1
    train_set = data[train_idx]
    test_set = data[test_idx]
    for epoch in range(10):
        epoch_loss = 0
        model.train()

        for i, (inputs, l) in enumerate(train_set):
            if i % 50 == 0:
                print(i)

            dg, graph_seq_feature = inputs
            logits = model(dg,graph_seq_feature)
            loss = loss_fcn(logits, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (i + 1)
        epoch_losses.append(epoch_loss)
        print(epoch,epoch_loss)

    losses.append(epoch_losses)
    score = evaluate(test_set, model)
    #em = em.tolist()
    scores.append(score)
    print('this is fold ', cnt, ' mat:', scores)


new_showResult = showRes(losses,scores)
new_showResult.showLoss()
new_showResult.showScores()
# with open('true_front_running.json','w') as f:
#     json.dump(em, f)