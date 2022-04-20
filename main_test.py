import numpy as np
import torch as th
import random
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from utils import readfile
from utils import evaluate2
from utils import model_select
from utils import fullset
from DataLoader import dataloader
from model_modify import get_embedding
import matplotlib.pyplot as plt
pca = PCA(n_components=3)
types = ['runtime_code']

files_new_uncheck = {'source_code': 'uncheck_low_level_call_sourcecode_path.json',
                     'runtime_code': 'uncheck_low_level_call_runtimecodecode_path.json',
                     'creation_code': 'uncheck_low_level_call_creationcode_path.json'}
files = {'runtime_code':'new_re.json'}
graphinfos = {key:readfile(files[key]) for key in types}
sets = {key:set(graphinfos[key].keys()) for key in types}
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

metapaths= [['False', 'eslaF'], ['True', 'eurT'], ['CF', 'FC']]
mask = np.array(name_list)
cnt = 0
epoch_losses = []
for train_idx, test_idx in kf.split(mask):
    print('ready for training')
    if cnt > 0:
        break

    cnt += 1
    for type in types:
        train_set = datas[type][train_idx]
        test_set = datas[type][test_idx]
        model =  get_embedding(meta_paths=metapaths,
                     in_size=8,
                     hidden_size=32,
                     out_size=2,
                     num_heads=[8],
                     dropout=0.1)
        loss_fcn= th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001,
                                  weight_decay=0.001)

        for epoch in range(20):
            epoch_loss = 0
            model.train()

            for i,(g,l) in enumerate(train_set):

                features = g.ndata['f']

                logits,embedding = model(g,features)
                loss = loss_fcn(logits, l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (i + 1)
            epoch_losses.append(epoch_loss)
            print(epoch,epoch_loss)

        scores = evaluate2(train_set,model)
        #print('this is fold ', cnt, type,' mat:',scores)


print(epoch_losses)
cnt1 = 0
for obj in scores:
    if obj != 'embeddings':
        print(scores[obj])

embeddings = np.array(scores['embeddings'])
print(embeddings)
colors = scores['colors']
compressed_embedding = pca.fit_transform(embeddings)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(compressed_embedding[:,0],compressed_embedding[:,1],compressed_embedding[:,2],c=np.array(colors))
#plt.scatter(compressed_embedding[:,0],compressed_embedding[:,1],c=np.array(colors))
plt.show()