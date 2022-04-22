import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import dgl
import json
import os
label_path = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/unchecked_low_level_calls/contract_labels.json'
# creation_path = '/Users/xiechunyao/Downloads/crytic_byte_code/creation/access_control/clean_57_buggy_curated_0/'
runtime_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/unchecked_low_level_calls/clean_95_buggy_curated_0/'
d_label = {}

d = {}
with open(label_path) as f:
    data_label = json.load(f)
data_runtime_graph = os.listdir(runtime_path)
for obj in data_label:

    a = obj['contract_name']
    name = a[:a.find('-')]
    if name not in d_label:
        d_label[name] = obj['targets']
    else:
        d_label[name] |= obj['targets']
for file in data_runtime_graph:
    if '.dot' in file:
        name = file[:file.find('-')]
        if name not in d:
            d[name] = {}
            d[name]['path'] = []
            d[name]['label'] = d_label[name]
        d[name]['path'].append(runtime_path+file)
# print(d)
# with open('un_from_minh.json','w') as f:
#     json.dump(d,f)
# for con in d_label:
#     if d_label[con] == 0:
#         d_clean[con] = 0
#     else:
#         d_buggy[con] = 1
# count = 0
# for name in d_clean:
#     count += 1
#     print(count,name+'.sol')
path_re = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/reentrancy/contract_labels.json'
path_uncheck = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/unchecked_low_level_calls/contract_labels.json'
with open(path_re) as fre:
    data_re = json.load(fre)
with open(path_uncheck) as fun:
    data_un = json.load(fun)
re_label = {}
un_label = {}
for obj in data_re:
    a = obj['contract_name']
    name = a[:a.find('-')]
    if name not in re_label:
        re_label[name] = obj['targets']
    else:
        re_label[name] |= obj['targets']
for obj in data_un:
    a = obj['contract_name']
    name = a[:a.find('-')]
    if name not in un_label:
        un_label[name] = obj['targets']
    else:
        un_label[name] |= obj['targets']
l_re = [k for k in re_label if re_label[k] == 0]
l_un = [k for k in un_label if un_label[k] == 0]
count = 0
for x in l_re:
    if x in l_un:
        count += 1
print(count)