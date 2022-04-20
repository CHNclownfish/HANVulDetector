import numpy as np
import torch as th
import random
from sklearn.model_selection import KFold
from utils import readfile
from utils import evaluate
from utils import model_select
from utils import fullset
from DataLoader import dataloader
from sklearn import metrics
import matplotlib.pyplot as plt

# can add 'type'to the list to compare different type of codes,
# where types are 'source_code', 'runtime_code' and 'creation_code'
types = ['runtime_code']#,'creation_code']#['source_code'

files = {'source_code': None,
         'runtime_code': 'files/access_control_runtime.json',
         'creation_code': None}

graphinfos = {key:readfile(files[key]) for key in types}

sets = {key:set(graphinfos[key].keys()) for key in types}
# name_list = sets['source_code'] & sets['runtime_code'] & sets['creation_code']
#name_list = set(fullset())
name_list = set()
for key in sets:
    name_list |= sets[key]
for key in sets:
    name_list &= sets[key]
name_list = list(name_list)
random.shuffle(name_list)
print(len(name_list))

dataloaders = {key:dataloader(name_list,graphinfos[key]) for key in types}

datas = {key:dataloaders[key].createdata(key) for key in types}

print('dataset prepare finish')

matrixs = {key:[] for key in types}
kf = KFold(n_splits=5,random_state=1,shuffle=True)
# kf = KFold(n_splits=5)

mask = np.array(name_list)
cnt = 0
epoch_losses = []
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
            epoch_losses.append(epoch_loss)
            print(epoch,epoch_loss)

        scores = evaluate(test_set,model)
        matrixs[type].append(scores)
        print('this is fold ', cnt, type, ' mat:', scores)

print(epoch_losses)
cnt1 = 0
for k in matrixs:
    print(k)
    for obj in matrixs[k]:
        print(obj)