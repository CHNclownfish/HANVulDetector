import json
import os
import networkx as nx
import torch
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model import HAN
import GraphGenerator
def readfile(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
    return data
def evaluate(data,m):
    y_true = []
    y_pred = []
    for g,l in data:
        logit = m(g,g.ndata['f'])
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    micro = metrics.precision_score(y_true, y_pred, average='micro')
    macro = metrics.precision_score(y_true, y_pred, average='macro')
    recall_micro = metrics.recall_score(y_true, y_pred, average='micro')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    d = {'acc': acc,'micro': micro, 'macro': macro, 'recall_micro': recall_micro, 'recall_macro': recall_macro, 'f1_micro': f1_micro,'f1_macro':f1_macro}
    return d

def fullset():
    with open('fullset.json') as f:
        data = json.load(f)
    return data['fullset']

def model_select(type):

    metapaths= [['False', 'eslaF'], ['True', 'eurT'], ['CF', 'FC']]
    if type == 'source_code':
        model = HAN(meta_paths=metapaths,
                    in_size=15,
                    hidden_size=32,
                    out_size=2,
                    num_heads=[8],
                    dropout=0)
    if type == 'runtime_code' or type == 'creation_code':
        model = HAN(meta_paths=metapaths,
                    in_size=8,
                    hidden_size=32,
                    out_size=2,
                    num_heads=[8],
                    dropout=0)
    return model

# path = '/Users/xiechunyao/Downloads/SolidiFI-benchmark-master/buggy_contracts/TOD/'
# data = os.listdir(path)
# for x in data:
#     if '.sol' in x:
#         print('slither '+ path +x + ' --print cfg')
#
# def readname(type,file):
#     if type == 'creation' or type == 'source':
#         name = file[:file.find('.')]
#     if type == 'runtime':
#         name = file[:file.find('_')]
#     return name
# path_clean = 'dataset/clean/clean_creationcode_cfg/'
# path_re= 'dataset/reentrancy/re_creationcode_cfg/'
# data_clean = os.listdir(path_clean)
# data_re = os.listdir(path_re)
# d = {}
# for x in data_clean:
#     if '.dot' in x:
#         a = x.split('_')
#         name = a[0] + '_' + a[1] # for runtimecode
#         #name = x[:x.find('.')] # for sourcecode
#         print(name)
#         if name not in d:
#             d[name] = {}
#             d[name]['path'] = []
#             d[name]['label'] = 0
#         d[name]['path'].append(path_clean+x)
#
# for y in data_re:
#     if '.dot' in y:
#         b = y.split('_')
#         name = b[0] + '_' + b[1]
#         # name = y[:y.find('.')]
#         print(name)
#         if name not in d:
#             d[name] = {}
#             d[name]['path'] = []
#             d[name]['label'] = 1
#         d[name]['path'].append(path_re+y)
# for name in d:
#     print(name,d[name])
# with open('reentancy_creationcode_path.json','w') as f:
#     json.dump(d,f)
# def generategraph(dotfilepaths):
#     g = nx.drawing.nx_pydot.read_dot(dotfilepaths[0])
#     for i in range(1,len(dotfilepaths)):
#         new_g = nx.drawing.nx_pydot.read_dot(dotfilepaths[i])
#         g = nx.algorithms.operators.binary.disjoint_union(g,new_g)
#     return g
#
# file1 = '/Users/xiechunyao/sourcecode_unchecked_low_level_calls_label.json'
# file2 = '/Users/xiechunyao/sourcecode_reentrancy_label.json'
# data1 = readfile(file1)
# data2 = readfile(file2)
# a = set()
# for k in data1:
#     print(data1[k]['path'])
#     g = generategraph(data1[k]['path'])
#     nodes = g.nodes()
#
#     for idx,node_idx in enumerate(nodes):
#         node_type = g._node[node_idx]['label'].split(' ')[2]
#         a.add(node_type)
# for k2 in data2:
#     print(data2[k2]['path'])
#     if k2 == '0xfca47962d45adfdfd1ab2d972315db4ce7ccf094':
#         continue
#     g = generategraph(data2[k2]['path'])
#     nodes = g.nodes()
#
#     for idx,node_idx in enumerate(nodes):
#         node_type = g._node[node_idx]['label'].split(' ')[2]
#         a.add(node_type)
# print(len(a),a)
# # 15
# {'IF_LOOP', 'ENTRY_POINT', 'OTHER_ENTRYPOINT', '_', 'END_IF', 'BREAK', 'EXPRESSION', 'CONTINUE', 'INLINE', 'END_LOOP', 'BEGIN_LOOP', 'THROW', 'IF', 'RETURN', 'NEW'}
