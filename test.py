import networkx as nx
from opcodes import int2op
from baseline_LGCN.seqGenerator_LGCN import dataGenerator
from baseline_LGCN.L_GCN import LGCNModel
import torch as th
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
from pathfinder import JsonPathFinder
from new_L_HAN.L_HANGraphGenerator import graphGenerator_ethersolve
a = ['/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/buggy/buggy_2-CareerOnToken.dot']
Gg = graphGenerator_ethersolve(a)
nx_g = nx.drawing.nx_pydot.read_dot(a[0])
dg = Gg.reflectHeteroGraph()
print(len(nx_g.nodes()))
print(dg.num_nodes())
