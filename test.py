import torch
import json
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
from utils import readfile
import random
file_runtime = '/Users/xiechunyao/reentrancy_runtimecode_label.json'
file_sourcecode = '/Users/xiechunyao/reentrancy_sourcecode_label.json'

with open(file_sourcecode) as f:
    data = json.load(f)
d = {}
for k in data:
    d[k] = {}
    d[k]['path'] = []
    d[k]['label'] = data[k]['label']
    for i in range(len(data[k]['path'])):
        a = data[k]['path'][i].split('/')
        d[k]['path'].append(a[3] +'/'+  a[4])
# with open('reentrancy_runtimecode_label.json','w') as f1:
#     json.dump(d,f1)


