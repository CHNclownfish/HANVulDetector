import networkx as nx
from opcodes import int2op
from baseline_LGCN.seqGenerator_LGCN import dataGenerator
from baseline_LGCN.L_GCN import LGCNModel
import torch
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
from pathfinder import JsonPathFinder
import json
import os
path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/unchecked_low_level_calls/clean_95_buggy_curated_0/'
graphdata = os.listdir(path)
for x in graphdata:
    print(x)
with open('la.json') as f:
    balance_label = json.load(f)
tobecreatedata = []
for obj in balance_label:
    graph_name = obj['contract_name'].split('.')[0]+'.dot'
    d = {}
    if graph_name in graphdata:
        d['path'] = path + graph_name
        d['target'] = obj['targets']
        tobecreatedata.append(d)
b,c = 0,0,
for obj in tobecreatedata:
    if obj['target'] == 1:
        b += 1
    else:
        c += 1
print(b,c)
with open('unchecked_low_level_calls_trueBalanceForLGCN.json','w') as f1:
    json.dump(tobecreatedata,f1)

