import torch
from baseline_lstm.model_LSTM import LSTMModel
import json
import random
import os
from pathfinder import JsonPathFinder
def contract2runtimbin(path_base, labels_path, labels_path_clean):
    info = []
    with open(labels_path_clean) as flc:
        labels_clean = json.load(flc)
    with open(labels_path) as fl:
        labels_buggy = json.load(fl)
    labels = {}
    for c in labels_clean:
        labels[c['contract_name']] = c['targets']
    for b in labels_buggy:
        labels[b['contract_name']] = b['targets']

    files_ = os.listdir(path_base)
    files = []
    for x in files_:
        files.append(x)
    random.shuffle(files)
    for file in files:
        if '.json' not in file:
            continue

        with open(path_base+file) as f:
            data = f.read()
        finder = JsonPathFinder(data)
        path_list = finder.find_all('bin-runtime')
        d = {}
        blockchain_name = file[:file.find('.')]

        for i,path in enumerate(path_list):
            with open(path_base+file) as f1:
                obj = json.load(f1)
            contract_name = path[4]
            for key in path:
                obj = obj[key]
            if obj != '':
                name = blockchain_name + '-' + contract_name + '.sol'
                d[name] = {}
                d[name]['bin-runtime'] = obj
                d[name]['label'] = labels[name]
        info.append(d)
    return info
path_base = '/Users/xiechunyao/dataset_08_Apr/front_running/front_running_export/'
labels_path = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/front_running/contract_labels.json'
labels_path_clean = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/clean_labels.json'

info = contract2runtimbin(path_base, labels_path, labels_path_clean)
cnt = 0
ones = 0
for file in info:
    for contract in file:
        cnt += 1
        if file[contract]['label'] == 1:
            ones += 1
print(cnt, ones)