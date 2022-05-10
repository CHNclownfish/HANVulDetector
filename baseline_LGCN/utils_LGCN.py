import json
from pathfinder import JsonPathFinder
import os
import random
import torch
from sklearn import metrics
from sklearn.metrics import classification_report

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
def evaluate(data,m):
    y_true = []
    y_pred = []
    embeddings = []
    colors = []
    for inputs, l in data:
        dg, graph_seq_feature = inputs
        logit, embedding = m(dg,graph_seq_feature)
        embeddings.append(embedding)
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
        if l == predict == torch.LongTensor([1]):
            colors.append('red')
        elif l == predict == torch.LongTensor([0]):
            colors.append('green')
        elif l == torch.LongTensor([1]) and l != predict:
            colors.append('blue')
        else:
            colors.append('yellow')

    print(y_true)
    print(y_pred)
    embeddings = torch.stack([eb for eb in embeddings], 0)
    embeddings = embeddings.view(-1, 80)
    print(embeddings.size())
    embeddings = embeddings.detach().numpy()
    target_names = ['class 0', 'class 1']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_buggy = report['class 0']['f1-score']
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_buggy_score = "%.{}f".format(2) % (100*(f1_buggy))
    f1_macro_score = "%.{}f".format(2) % (100*(f1_macro))
    acc_score = "%.{}f".format(2) % (100*(acc))

    return {'f1_buggy': str(f1_buggy_score)+'%', 'f1_macro': str(f1_macro_score)+'%', 'acc':str(acc_score)+'%'},embeddings,colors
def readJson(path):
    with open(path) as f:
        info = json.load(f)
    #random.shuffle(info)
    return info

# given a list of obj, where obj is a dict with {'contract_name': source_file_name-contract_name.sol,'targets':0 or 1}
# return a dict where {source_file_name:[(contract_name1.sol, targets), (contract_name2.sol, targets)]}
def readlabel(file):
    res = {}
    with open(file) as f:
        # here data is a list of obj
        data = json.load(f)
    for obj in data:
        source_file_name = obj['contract_name'][:obj['contract_name'].find('-')]
        if source_file_name not in res:
            res[source_file_name] = []
        res[source_file_name].append((obj['contract_name'],obj['targets']))
    return res
