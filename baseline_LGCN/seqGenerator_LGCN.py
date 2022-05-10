import json
import torch
import numpy as np
import networkx as nx
from opcodes import int2op
import dgl

# file is a json file, which is a list of obj, where obj = {'path':'graph/path.dot','target': 0 or 1}
class dataGenerator:
    def __init__(self, file):
        s = '0123456789abcdef'
        self.file = file
        self.str2int = {s[i]: i for i in range(len(s))}
        #self.str2bit = {s[i]: [int(x) for x in (4 - len(bin(i)[2:])) * '0' + bin(i)[2:]] for i in range(len(s))}
        self.data = []

    # op is a operation code like 'PUSH1' , "RETURN" returns a 256 dim onehot vec for this operation
    def op2onehot(self, op):
        onehot_line = [0 for _ in range(256)]
        if op != 'EXIT BLOCK':
            op2int = {int2op[key]:key for key in int2op}
            hexstr = op2int[op]
            number = self.str2int[hexstr[0]] * 16 + self.str2int[hexstr[1]]
            onehot_line[number] = 1
        return onehot_line

    def encoder(self, nx_g):
        nodes = nx_g.nodes()
        graph_seq_feature = []
        vec_u, vec_v = [], []
        mapping = {n: i for i, n in enumerate(nodes)}
        for idx, node_idx in enumerate(nodes):
            node_seq_feature = []
            obj = nx_g._node[node_idx]
            seq = obj['label'].split(':')[1:]
            #print(obj)
            for unit in seq:
                op = unit[1:unit.find('\\')]
                if 'PUSH' in op:
                    op = op.split(' ')[0]
                node_seq_feature.append(self.op2onehot(op))
            graph_seq_feature.append(torch.tensor([node_seq_feature]).float())
        for u, v in nx_g.edges():
            vec_u.append(mapping[u])
            vec_v.append(mapping[v])
        dg = dgl.graph((vec_u,vec_v))
        dg = dgl.add_self_loop(dg)
        dg.ndata['f'] = torch.zeros(dg.num_nodes(), 64)
        return dg, graph_seq_feature


    def get_data(self):
        counter = 0
        for obj in self.file:
            counter += 1
            nx_g = nx.drawing.nx_pydot.read_dot(obj['path'])
            dg, graph_seq_feature = self.encoder(nx_g)
            l = torch.LongTensor([obj['target']])
            self.data.append(([dg, graph_seq_feature], l))
            if counter % 10 == 0:
                print(l, counter)
        return np.array(self.data)

class dataGenerator_sourcefile:
    def __init__(self, file):
        s = '0123456789abcdef'
        self.file = file
        self.str2int = {s[i]: i for i in range(len(s))}
        #self.str2bit = {s[i]: [int(x) for x in (4 - len(bin(i)[2:])) * '0' + bin(i)[2:]] for i in range(len(s))}
        self.data = []

    # op is a operation code like 'PUSH1' , "RETURN" returns a 256 dim onehot vec for this operation
    def op2onehot(self, op):
        onehot_line = [0 for _ in range(256)]
        if op != 'EXIT BLOCK':
            op2int = {int2op[key]:key for key in int2op}
            hexstr = op2int[op]
            number = self.str2int[hexstr[0]] * 16 + self.str2int[hexstr[1]]
            onehot_line[number] = 1
        return onehot_line

    def encoder(self, nx_g):
        nodes = nx_g.nodes()
        graph_seq_feature = []
        vec_u, vec_v = [], []
        mapping = {n: i for i, n in enumerate(nodes)}
        for idx, node_idx in enumerate(nodes):
            node_seq_feature = []
            obj = nx_g._node[node_idx]
            seq = obj['label'].split(':')[1:]
            #print(obj)
            for unit in seq:
                op = unit[1:unit.find('\\')]
                if 'PUSH' in op:
                    op = op.split(' ')[0]
                node_seq_feature.append(self.op2onehot(op))
            graph_seq_feature.append(torch.tensor([node_seq_feature]).float())
        for u, v in nx_g.edges():
            vec_u.append(mapping[u])
            vec_v.append(mapping[v])
        dg = dgl.graph((vec_u,vec_v))
        dg = dgl.add_self_loop(dg)
        dg.ndata['f'] = torch.zeros(dg.num_nodes(), 64)
        return dg, graph_seq_feature


    def get_data(self):
        counter = 0
        for obj in self.file:
            nx_g = nx.drawing.nx_pydot.read_dot(obj['path'][0])
            for i in range(1, len(obj['path'])):
                new_g = nx.drawing.nx_pydot.read_dot(obj['path'][i])
                nx_g = nx.algorithms.operators.binary.disjoint_union(nx_g, new_g)
            counter += 1
            #nx_g = nx.drawing.nx_pydot.read_dot(obj['path'])
            dg, graph_seq_feature = self.encoder(nx_g)
            l = torch.LongTensor([obj['target']])
            self.data.append(([dg, graph_seq_feature], l))
            if counter % 10 == 0:
                print(l, counter)
        return np.array(self.data)