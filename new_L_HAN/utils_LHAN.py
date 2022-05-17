import json
from pathfinder import JsonPathFinder
import os
import random
import torch
from sklearn import metrics
from sklearn.metrics import classification_report
from model_LHAN import LHAN_parallel
from model_L2HAN import LHAN_Series
from model_L3HAN import LHAN_Combine
from model_Orig import HAN_org
from baseline_LGCN.L_GCN import LGCNModel


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

def evaluate_embedding(data,m):
    y_true = []
    y_pred = []
    embeddings = []
    colors = []
    for inputs, l in data:
        dg, graph_seq_feature = inputs
        logit, embedding = m(dg, dg.ndata['f'], graph_seq_feature)
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
    embeddings = embeddings.view(-1, 256 + 64)
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
def evaluate_embeddings(data, m, type_):
    in_size = 8
    hidden_size = 32
    out_size = 2
    num_heads = 8
    dropout = 0

    seq_input_dim = 256
    seq_hidden_dim = 64
    seq_layer_dim = 1
    y_true = []
    y_pred = []
    embeddings = []
    colors = []
    for inputs, l in data:
        dg, graph_seq_feature = inputs
        if type_ == 'LGCN':
            logit, embedding = m(dg,graph_seq_feature)
        else:
            logit, embedding = m(dg, dg.ndata['f'], graph_seq_feature)
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
    if type_ == 'parallel':
        embeddings = embeddings.view(-1, hidden_size * num_heads + seq_hidden_dim)
    elif type_ == 'LGCN' :
        embeddings = embeddings.view(-1, 80)
    else:
        embeddings = embeddings.view(-1, hidden_size * num_heads)
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

def evaluate(data,m):
    y_true = []
    y_pred = []
    for inputs, l in data:
        dg, graph_seq_feature = inputs
        logit, em = m(dg, dg.ndata['f'], graph_seq_feature)
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
    print(y_true)
    print(y_pred)
    target_names = ['class 0', 'class 1']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_buggy = report['class 0']['f1-score']
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_buggy_score = "%.{}f".format(2) % (100*(f1_buggy))
    f1_macro_score = "%.{}f".format(2) % (100*(f1_macro))
    acc_score = "%.{}f".format(2) % (100*(acc))

    return {'f1_buggy': str(f1_buggy_score)+'%', 'f1_macro': str(f1_macro_score)+'%', 'acc':str(acc_score)+'%'}

def model_select(type_):
    in_size = 8
    hidden_size = 32
    out_size = 2
    num_heads = [8]
    dropout = 0

    seq_input_dim = 256
    seq_hidden_dim = 64
    seq_layer_dim = 1

    metapaths= [['False', 'eslaF'], ['True', 'eurT'], ['CF', 'FC']]

    if type_ == 'parallel':
        model = LHAN_parallel(metapaths, in_size, hidden_size, out_size, num_heads,
                     seq_input_dim, seq_hidden_dim, seq_layer_dim, dropout)
    elif type_ == 'series':
        model = LHAN_Series(metapaths, hidden_size, out_size, num_heads,
                              seq_input_dim, seq_hidden_dim, seq_layer_dim, dropout)
    elif type_ == 'combine':
        model = LHAN_Combine(metapaths, in_size, hidden_size, out_size, num_heads,
                              seq_input_dim, seq_hidden_dim, seq_layer_dim, dropout)
    elif type_ == 'origin':
        model = HAN_org(metapaths, in_size, hidden_size, out_size, num_heads, dropout)

    elif type_ == 'LGCN':
        model = LGCNModel(graph_input_dim=64, graph_hidden_dim1=32,graph_hidden_dim2=16, seq_input_dim=256,
                          seq_hidden_dim=64, seq_layer_dim=1, output_dim=2)

    return model

def smalldataset(balance_label, graphdata, graph_path):
    info = []
    c = 0
    for obj in balance_label:
        c += 1
        if c <= 80:
            graph_name = obj['contract_name'].split('.')[0]+'.dot'
            d = {}
            if graph_name in graphdata:
                d['path'] = graph_path + graph_name
                d['target'] = obj['targets']
                info.append(d)
    return info

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

def orderedsourcefile(graph_clean_path,graph_buggy_path):
    graphdata_clean = os.listdir(graph_clean_path)
    graphdata_buggy = os.listdir(graph_buggy_path)
    name_list = set()
    d_info = {}
    for obj in graphdata_clean:
        if '.dot' in obj:
            file_name = obj[:obj.find('-')]
            if file_name not in d_info:
                d_info[file_name] = {}
                d_info[file_name]['path'] = []
                d_info[file_name]['target'] = 0
            d_info[file_name]['path'].append(graph_clean_path+obj)
    for obj1 in graphdata_buggy:
        if '.dot' in obj1:
            file_name = obj1[:obj1.find('-')]
            if file_name not in d_info:
                d_info[file_name] = {}
                d_info[file_name]['path'] = []
                d_info[file_name]['target'] = 1
            d_info[file_name]['path'].append(graph_buggy_path+obj1)
    return d_info
