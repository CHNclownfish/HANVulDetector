import json
import random

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
import os

#labels_path = 'files/front_running_trueBalanceForLGCN.json'
graph_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/unchecked_low_level_calls/clean_95_buggy_curated_0/'
graphdata = os.listdir(graph_path)

with open('files/la.json') as f:
    balance_label = json.load(f)
f.close()
info = []

for obj in balance_label:
    graph_name = obj['contract_name'].split('.')[0]+'.dot'
    d = {}
    if graph_name in graphdata:
        d['path'] = graph_path + graph_name
        d['target'] = obj['targets']
        info.append(d)
random.shuffle(info)
for x in info :
    print(x)
# info = readJson(labels_path)
# with open(labels_path) as f:
#     info = json.load(f)
# print(info)
# graph_clean_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/clean_57_buggy_curated_0/'
# graph_buggy_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/buggy/'
# graphdata_clean = os.listdir(graph_clean_path)
# graphdata_buggy = os.listdir(graph_buggy_path)
# info = []
# name_list = set()
# d_info = {}
# for obj in graphdata_clean:
#     if '.dot' in obj:
#         file_name = obj[:obj.find('-')]
#         if file_name not in d_info:
#             d_info[file_name] = {}
#             d_info[file_name]['path'] = []
#             d_info[file_name]['target'] = 0
#         d_info[file_name]['path'].append(graph_clean_path+obj)
# for obj1 in graphdata_buggy:
#     if '.dot' in obj1:
#         file_name = obj1[:obj1.find('-')]
#         if file_name not in d_info:
#             d_info[file_name] = {}
#             d_info[file_name]['path'] = []
#             d_info[file_name]['target'] = 1
#         d_info[file_name]['path'].append(graph_buggy_path+obj1)
# for x in d_info:
#     info.append(d_info[x])
# random.shuffle(info)


dG = dataGenerator(info)
data = dG.get_data()
graph_input_dim = 64
graph_hidden_dim1 = 32
graph_hidden_dim2 = 16
seq_input_dim = 256
seq_hidden_dim = 64
seq_layer_dim = 1
output_dim = 2


kf = KFold(n_splits=5, random_state=1, shuffle=True)
#kf = KFold(n_splits=5)
losses = []
scores = []
cnt = 0
pca = PCA(n_components=2)
print('data prepared')
for train_idx, test_idx in kf.split(data):
    model = LGCNModel(graph_input_dim, graph_hidden_dim1,graph_hidden_dim2, seq_input_dim, seq_hidden_dim, seq_layer_dim, output_dim)
    loss_fcn= th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.005,
                              weight_decay=0.001)
    epoch_losses = []
    cnt += 1
    train_set = data[train_idx]
    test_set = data[test_idx]
    model.train()
    for epoch in range(10):
        epoch_loss = 0

        for i, (inputs, l) in enumerate(train_set):
            if i % 50 == 0:
                print(i)
            dg, graph_seq_feature = inputs
            logits, emb = model(dg,graph_seq_feature)
            loss = loss_fcn(logits, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (i + 1)
        epoch_losses.append(epoch_loss)
        print(epoch,epoch_loss)

    losses.append(epoch_losses)
    score, em, colors = evaluate(test_set, model)

    scores.append(score)
    print('this is fold ', cnt, ' mat:', scores)
    compressed_embedding = pca.fit_transform(em)
    #fig = plt.figure(figsize=(12, 12))
    plt.subplot(1, 5, cnt)
    plt.scatter(compressed_embedding[:,0],compressed_embedding[:,1],c=np.array(colors))



new_showResult = showRes(losses,scores)
new_showResult.showLoss()
new_showResult.showScores()
plt.show()
# with open('true_front_running.json','w') as f:
#     json.dump(em, f)