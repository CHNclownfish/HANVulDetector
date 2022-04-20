import json
import os
import networkx as nx
import torch
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
# from model import HAN
from model_modify import HAN
import numpy as np
import matplotlib.pyplot as plt
import GraphGenerator
def readfile(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
    return data

# def evaluate(data,m):
#     y_true = []
#     y_pred = []
#     for g,l in data:
#         logit = m(g,g.ndata['f'])
#         predict = torch.max(logit,1)[1]
#         y_pred.append(predict)
#         y_true.append(l)
#     print(y_true)
#     print(y_pred)
#     acc = accuracy_score(y_true, y_pred)
#     micro = metrics.precision_score(y_true, y_pred, average='micro')
#     macro = metrics.precision_score(y_true, y_pred, average='macro')
#     recall_micro = metrics.recall_score(y_true, y_pred, average='micro')
#     recall_macro = metrics.recall_score(y_true, y_pred, average='macro')
#     f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
#     f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
#     d = {'acc': acc,'micro': micro, 'macro': macro, 'recall_micro': recall_micro, 'recall_macro': recall_macro, 'f1_micro': f1_micro,'f1_macro':f1_macro}
#     return d

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
    target_names = ['class 0', 'class 1']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_buggy = report['class 0']['f1-score']
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

    return [f1_buggy,f1_macro]


def evaluate2(data,m):
    y_true = []
    y_pred = []
    embeddings = []

    colors = []
    for g,l in data:
        logit,embedding = m(g,g.ndata['f'])
        predict = torch.max(logit,1)[1]

        y_pred.append(predict)
        y_true.append(l)

        embeddings.append(np.ndarray.tolist(embedding.cpu().detach().numpy())[0])
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
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    d = {'f1_micro': f1_micro,'f1_macro':f1_macro,'embeddings':embeddings,'colors':colors}
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
                    dropout=0.1)
    if type == 'runtime_code' or type == 'creation_code':
        model = HAN(meta_paths=metapaths,
                    in_size=8,
                    hidden_size=32,
                    out_size=2,
                    num_heads=[8],
                    dropout=0)
    return model

def createfile(path_clean,path_buggy,type,isAddBug):
    data_clean = os.listdir(path_clean)
    data_buggy = os.listdir(path_buggy)
    d = {}
    for x in data_clean:
        if '.dot' in x:
            if type == 'source_code':
                name = x[:x.find('.')]
            else:
                a = x.split('_')
                name = a[0] + '_' + a[1]
            if name not in d:
                d[name] = {}
                d[name]['path'] = []
                d[name]['label'] = 0
            d[name]['path'].append(path_clean+x)

    for y in data_buggy:
        if '.dot' in y:
            if type == 'source_code':
                name = y[:y.find('.')]
            else:
                if isAddBug:
                    name = y[:y.rfind('_')]
                else:
                    b = y.split('_')
                    name = b[0] + '_' + b[1]
            if name not in d:
                d[name] = {}
                d[name]['path'] = []
                d[name]['label'] = 1
            d[name]['path'].append(path_buggy+y)

    return d
def showLoss(losses):
    print(len(losses))
    epochs = np.array([i for i in range(len(losses))])
    plt.plot(epochs,losses)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss per epoch")
    plt.show()
# l_reentrancy = [0.6710577673110806, 0.5691893248528731, 0.5017760518755092, 0.47306075124222724, 0.46180878481904014, 0.45571812765947617, 0.4512815716233654, 0.4475759717284656, 0.4441603803549145, 0.44189338358577157, 0.43964685614938376, 0.4377819888034194, 0.4362727016836527, 0.4349076290172143, 0.43360649485171576, 0.43247500184892873, 0.4315224930216543, 0.4306058286780827, 0.42968887135722356, 0.4289345834709582]#, 0.6406884370524375, 0.52467081683581, 0.46653234573905583, 0.44897505109671687, 0.4406863214127475, 0.43536796076528606, 0.43101153249440133, 0.4260477719706346, 0.4224850509895897, 0.4199100546347985, 0.41724517947200257, 0.4154791802961807, 0.41377152373144005, 0.41243632065636093, 0.41093894482407045, 0.40975161912835767, 0.4083280185367301, 0.40744015219102264, 0.4062201510771315, 0.40508216468342506, 0.6561917635749598, 0.567344669802267, 0.5109188839426784, 0.49170429070220617, 0.4849484319203213, 0.4810240590853281, 0.47789855533447423, 0.4751688040731872, 0.4726137731102158, 0.47044377282383987, 0.4685595858781064, 0.4668004312170822, 0.4652751799611772, 0.4638190179391474, 0.4623012898459298, 0.46179565979686915, 0.46060471371060513, 0.4593549375345961, 0.45918727882939286, 0.45842221980822867, 0.6773006756131242, 0.5847794287573032, 0.5266930654766114, 0.5042694082589654, 0.4964476082686002, 0.4921874969713087, 0.48950115900214125, 0.4863965707278349, 0.4840632946328904, 0.48230479409297305, 0.4808042274439723, 0.4800876676430547, 0.47895976743562435, 0.47817230085289575, 0.4774391612083447, 0.4767925293464971, 0.47618633962985946, 0.4756744208980382, 0.47514818018166033, 0.47468719888872246, 0.6594714168610611, 0.553623718338284, 0.49236594837128633, 0.470951411237077, 0.4631420291384788, 0.45893860167664724, 0.45537892223252513, 0.45247292494386193, 0.4503905704562984, 0.4482457576001563, 0.4459609191834442, 0.44480807428074076, 0.44333953060573195, 0.4420685703162013, 0.44108027290946583, 0.4403482740669231, 0.4394380453731713, 0.4385168776218969, 0.43786739312657497, 0.43733849538475034]


# path_clean_main = 'dataset/clean/clean_runtimecode_cfg/'
# path_buggy_main = 'dataset/access_control/access_control_runtime_cfg/'
# # path_clean_add = '/Users/xiechunyao/dataset_08_Apr/clean_addition_runtimecfg/'
# # path_buggy_add = 'dataset/uncheck_low_level_call/uncheck_low_level_call_addition/uncheck_low_level_call_runtimecfg/'
# type = ['source_code', 'runtime_code', 'creation_code']
# d_main = createfile(path_clean_main,path_buggy_main,type[1],isAddBug=False)
# # d_add = createfile(path_clean_add,path_buggy_add,type[1],isAddBug=True)
# # d = dict(d_main,**d_add)
# #
# with open('files/access_control_runtime.json', 'w') as f:
#     json.dump(d_main,f)
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
