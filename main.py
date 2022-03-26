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

# can add 'type'to the list to compare different type of codes,
# where types are 'source_code', 'runtime_code' and 'creation_code'
types = ['source_code','runtime_code','creation_code']

files_uncheck = {'source_code': 'uncheck_low_level_calls_sourcecode.json',
                 'runtime_code': 'uncheck_low_level_calls_runtimecode.json',
                 'creation_code': 'uncheck_low_level_calls_creationcode.json'}

files_reentrancy = {'source_code':'reentancy_sourcecode_path.json',
                    'runtime_code':'reentancy_runtimecode_path.json',
                    'creation_code':'reentancy_creationcode_path.json'}

files_timestampel_dependency = {'source_code':'timestampledependency_sourcecode.json',
                    'runtime_code':'timestampledependency_runtime.json'}

files_uncheckedsend = {'source_code': 'uncheckedsend_sourcecode.json',
                       'runtime_code':'uncheckedsend_runtimecode.json'}

files_new_reentrancy = {'source_code': 'new_re_sourcecode.json',
                        'runtime_code': 'new_re_runtimecode.json',
                        'creation_code': 'new_re_creationcode.json'}

# for detecting uncheck_low_level_calls use this
# graphinfos = {key:readfile(files_uncheck[key]) for key in types}

# for detecting reentrancy use this
graphinfos = {key:readfile(files_reentrancy[key]) for key in types}

# for detecting timestample dependency use this
# graphinfos = {key:readfile(files_timestampel_dependency[key]) for key in types}

# for detecting unchecked_send use this
# graphinfos = {key:readfile(files_uncheckedsend[key]) for key in types}
# graphinfos = {key:readfile(files_new_reentrancy[key]) for key in types}


sets = {key:set(graphinfos[key].keys()) for key in types}
# name_list = sets['source_code'] & sets['runtime_code'] & sets['creation_code']
name_list = set(fullset())
for key in sets:
    name_list &= sets[key]
name_list = list(name_list)
random.shuffle(name_list)
print(len(name_list))

dataloaders = {key:dataloader(name_list,graphinfos[key]) for key in types}

datas = {key:dataloaders[key].createdata(key) for key in types}

print('dataset prepare finish')

matrixs = {key:[] for key in types}
# kf = KFold(n_splits=5,random_state=1,shuffle=True)
kf = KFold(n_splits=5)


mask = np.array(name_list)
cnt = 0
for train_idx, test_idx in kf.split(mask):
    print('ready for training')

    cnt += 1
    for type in types:
        train_set = datas[type][train_idx]
        test_set = datas[type][test_idx]
        model = model_select(type)
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
    print(k)
    for obj in matrixs[k]:
        print(obj)