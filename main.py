import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import pydot
import os
import collections
import random
from model import HAN
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch
import json
from GraphGenerator import graphGenerator_ethersolve
from GraphGenerator import graphGenerator_slither
from utils import readfile
from utils import evaluate
from utils import model_select
from utils import fullset
from DataLoader import dataloader

# can add 'type': model_select('type') to the dict to compare different type of codes,
# where types are 'source_code', 'runtime_code' and 'creation_code'
models = {'source_code':model_select('source_code'),'runtime_code': model_select('runtime_code')}

files_uncheck = {'source_code': 'uncheck_low_level_calls_sourcecode.json',
                 'runtime_code': 'uncheck_low_level_calls_runtimecode.json',
                 'creation_code': 'uncheck_low_level_calls_creationcode.json'}

files_reentrancy = {'source_code':'reentrancy_sourcecode.json',
                    'runtime_code':'reentrancy_runtimecode.json'}

# for detecting uncheck_low_lwvel_calls use this
graphinfos = {key:readfile(files_uncheck[key]) for key in models}

# for detecting reentrancy use this
# graphinfos = {key:readfile(files_reentrancy[key]) for key in models}


sets = {key:set(graphinfos[key].keys()) for key in models}
name_list = set(fullset())
for key in sets:
    name_list &= sets[key]
name_list = list(name_list)
random.shuffle(name_list)
print(len(name_list))

dataloaders = {key:dataloader(name_list,graphinfos[key]) for key in models}

datas = {key:dataloaders[key].createdata(key) for key in models}

print('dataset prepare finish')

matrixs = {key:[] for key in models}
# kf = KFold(n_splits=5,random_state=1,shuffle=True)
kf = KFold(n_splits=5)


mask = np.array(name_list)
cnt = 0
for train_idx, test_idx in kf.split(mask):
    print('ready for training')

    cnt += 1
    for type in models:
        train_set = datas[type][train_idx]
        test_set = datas[type][test_idx]
        model = models[type]
        loss_fcn= th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001,
                              weight_decay=0.001)

        for epoch in range(20):
            epoch_loss = 0
            model.train()

            for i,(g,l) in enumerate(train_set):

                features = g.ndata['f']

                logits = model(g,features)
                loss = loss_fcn(logits, l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (i + 1)
            print(epoch,epoch_loss)

        scores = evaluate(test_set,model)
        matrixs[type].append(scores)
        print('this is fold ', cnt, type,' mat:',scores)




for k in matrixs:
    print(matrixs[k])