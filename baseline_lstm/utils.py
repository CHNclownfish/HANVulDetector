import json
from pathfinder import JsonPathFinder
import os
import random
import torch
from sklearn import metrics
from sklearn.metrics import classification_report
def contract2runtimbin(path_base,labels_path,labels_path_clean):
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
def evaluate(data,m):
    y_true = []
    y_pred = []
    for seq,l in data:
        logit = m(seq['feature'])
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
    print(y_true)
    print(y_pred)
    target_names = ['class 0', 'class 1']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_buggy = report['class 0']['f1-score']
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

    return [f1_buggy,f1_macro]