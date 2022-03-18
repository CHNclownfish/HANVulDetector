import json
import os
import networkx as nx
import torch
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
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
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    d = {'acc': acc,'micro': micro, 'macro': macro, 'recall_micro': recall_micro, 'recall_macro': recall_macro, 'f1': f1}
    return d


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
