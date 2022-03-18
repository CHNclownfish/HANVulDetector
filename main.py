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
from DataLoader import dataloader


# file = '/Users/xiechunyao/creationcode_uncheck_call_label.json'
file_runtime = 'reentrancy_runtimecode_label.json'
file_sourcecode = 'reentrancy_sourcecode_label.json'
# file = '/Users/xiechunyao/sourcecode_unchecked_low_level_calls_label.json'
graphinfo_source = readfile(file_sourcecode)
graphinfo_runtime = readfile(file_runtime)
set_for_runtime = graphinfo_runtime.keys()
set_for_source = graphinfo_source.keys()

name_list = list(set(set_for_runtime) & set(set_for_source))
random.shuffle(name_list)

runtime = 'bytecode'
source = 'sourcecode'
dataloader_runtime = dataloader(name_list,graphinfo_runtime)
dataloader_sourcecode = dataloader(name_list,graphinfo_source)
data_runtime = dataloader_runtime.createdata(runtime)
data_sourcecode = dataloader_sourcecode.createdata(source)
print('dataset prepare finish')

matrix_runtime = []
matrix_source = []
# kf = KFold(n_splits=5,random_state=1,shuffle=True)
kf = KFold(n_splits=5)
metapaths_runtime = [['CF', 'FC'],['ELSE','ESLE'],['IF','FI']]
metapaths_source = [['False', 'eslaF'], ['True', 'eurT'], ['CF', 'FC']]
print('ready for training')

cnt = 0
for train_idx, test_idx in kf.split(data_runtime):

    cnt += 1

    train_set_runtime = data_runtime[train_idx]
    test_set_runtime = data_runtime[test_idx]
    model_runtime = HAN(meta_paths=metapaths_runtime,
                in_size=8,
                hidden_size=32,
                out_size=2,
                num_heads=[8],
                dropout=0)
    loss_fcn_runtime = th.nn.CrossEntropyLoss()
    optimizer_runtime = th.optim.Adam(model_runtime.parameters(), lr=0.001,
                              weight_decay=0.001)

    for epoch in range(20):
        epoch_loss_runtime = 0
        model_runtime.train()

        for i,(g,l) in enumerate(train_set_runtime):

            features = g.ndata['f']

            logits = model_runtime(g,features)
            loss_runtime = loss_fcn_runtime(logits, l)

            optimizer_runtime.zero_grad()
            loss_runtime.backward()
            optimizer_runtime.step()
            epoch_loss_runtime += loss_runtime.detach().item()
        epoch_loss_runtime /= (i + 1)
        print(epoch,epoch_loss_runtime)

    scores_runtime = evaluate(test_set_runtime,model_runtime)
    matrix_runtime.append(scores_runtime)
    print('this is fold ', cnt, 'runtime code mat:',scores_runtime)


    train_set_source = data_sourcecode[train_idx]
    test_set_source = data_sourcecode[test_idx]
    model_source = HAN(meta_paths=metapaths_source,
                        in_size=15,
                        hidden_size=32,
                        out_size=2,
                        num_heads=[8],
                        dropout=0)
    loss_fcn_source = th.nn.CrossEntropyLoss()
    optimizer_source = th.optim.Adam(model_source.parameters(), lr=0.001,
                              weight_decay=0.001)

    for epoch in range(20):
        epoch_loss_source = 0
        model_source.train()

        for i,(g,l) in enumerate(train_set_source):

            features = g.ndata['f']

            logits = model_source(g,features)
            loss_source = loss_fcn_source(logits, l)

            optimizer_source.zero_grad()
            loss_source.backward()
            optimizer_source.step()
            epoch_loss_source += loss_source.detach().item()
        epoch_loss_source /= (i + 1)
        print(epoch,epoch_loss_source)

    scores_source = evaluate(test_set_source,model_source)
    matrix_source.append(scores_source)
    print('this is fold ', cnt, 'source code mat:',scores_source)

print(matrix_runtime)
print(matrix_source)