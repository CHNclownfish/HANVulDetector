import json
import random
from new_L_HAN.L_HANGraphGenerator import graphGenerator_ethersolve
from new_L_HAN.utils_LHAN import model_select
from baseline_LGCN.seqGenerator_LGCN import dataGenerator
from sklearn.model_selection import KFold
import torch
from new_L_HAN.utils_LHAN import evaluate_embeddings
from new_L_HAN.utils_LHAN import evaluate
from new_L_HAN.showResults_LHAN import showRes
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from utils_LHAN import orderedsourcefile
from utils_LHAN import smalldataset

#labels_path = 'files/front_running_trueBalanceForLGCN.json'
graph_paths = {'access_control': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/clean_57_buggy_curated_1/',
               'arithmetic': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/arithmetic/clean_60_buggy_curated_0/',
               'denial_of_service': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/denial_of_service/clean_46_buggy_curated_0/',
               'front_running': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/front_running/clean_44_buggy_curated_0/',
               'reentrancy': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/reentrancy/clean_71_buggy_curated_0/',
               'time_manipulation': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/time_manipulation/clean_50_buggy_curated_0/',
               'unchecked_low_level_calls': '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/unchecked_low_level_calls/clean_95_buggy_curated_0/'
               }
# graph_clean_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/clean_57_buggy_curated_0/'
# graph_buggy_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/buggy/'

graph_path = graph_paths['time_manipulation']
graphdata = os.listdir(graph_path)

with open('label_files/time_manipulation_labels.json') as f:
    balance_label = json.load(f)
f.close()
info = []
# d_info = orderedsourcefile(graph_clean_path,graph_buggy_path)
# for x in d_info:
#     info.append(d_info[x])
# after this, info is a list of obj, where obj = {'path': path/name.dot, target: 0 or 1}
for obj in balance_label:
    graph_name = obj['contract_name'].split('.')[0]+'.dot'
    d = {}
    if graph_name in graphdata:
        d['path'] = graph_path + graph_name
        d['target'] = obj['targets']
        info.append(d)

#info = smalldataset(balance_label,graphdata,graph_path)

random.shuffle(info)
info_data = np.array(info)
def get_data_LHAN(info):
    counter = 0
    data_LHAN = []

    for obj in info:
        counter += 1
        Gg = graphGenerator_ethersolve(obj['path'], False)
        dg, gr = Gg.reflectHeteroGraph()
        l = torch.LongTensor([obj['target']])
        data_LHAN.append(([dg, gr], l))
        if counter % 10 == 0:
            print(l, counter)
    return np.array(data_LHAN)
data_LHAN = get_data_LHAN(info)

dG_LGCN = dataGenerator(info)
data_LGCN = dG_LGCN.get_data()
datas = {'parallel':data_LHAN, 'LGCN':data_LGCN}

kf = KFold(n_splits=5, random_state=2, shuffle=True)
#kf = KFold(n_splits=5)
losses = {}
scores = {}
cnt = 0
pca = PCA(n_components=2)
print('data prepared')
types = ['parallel','LGCN']#'origin',, 'series', 'combine'
#fig = plt.figure(figsize=(12, 12))
for t in types:
    losses[t] = []
    scores[t] = []
for train_idx, test_idx in kf.split(info_data):
    cnt += 1
    for j in range(len(types)):
        model = model_select(types[j])
        loss_fcn= torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                  weight_decay=0.001)
        epoch_losses = []
        train_set = datas[types[j]][train_idx]
        test_set = datas[types[j]][test_idx]
        model.train()
        # if cnt > 1:
        #     break
        for epoch in range(10):
            epoch_loss = 0

            for i, (inputs, l) in enumerate(train_set):
                if i % 50 == 0:
                    print(i)

                dg, graph_seq_feature = inputs
                if types[j] == 'LGCN':
                    logits, emb = model(dg,graph_seq_feature)
                else:
                    logits, emb = model(dg, dg.ndata['f'], graph_seq_feature)
                loss = loss_fcn(logits, l)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (i + 1)
            epoch_losses.append(epoch_loss)
            print(epoch,epoch_loss)

        losses[types[j]].append(epoch_losses)
        score, em, colors = evaluate_embeddings(test_set, model,types[j])
        #score = evaluate(test_set, model)

        scores[types[j]].append(score)
        print('this is fold ', cnt, ' mat:', score)
        compressed_embedding = pca.fit_transform(em)
        plt.subplot(5, 4, (cnt - 1) * 4 + j + 1)
        plt.scatter(compressed_embedding[:, 0],compressed_embedding[:, 1], c=np.array(colors))
        plt.title("fold "+ str(cnt) + " " + types[j])


new_showResult = showRes(losses,scores)
new_showResult.showScores()
new_showResult.showLoss()
plt.show()


for i in range(4):
    plt.subplot(1, 4, i+1)
    for t in losses:
        y = np.array(losses[t][i])
        x = np.array([j for j in range(len(losses[t][i]))])
        plt.plot(x, y)
    plt.title(str(i+1))
plt.show()