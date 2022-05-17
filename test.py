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
# a = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/buggy/buggy_2-CareerOnToken.dot'
# Gg = graphGenerator_ethersolve(a)
# nx_g = nx.drawing.nx_pydot.read_dot(a)
# dg, gr = Gg.reflectHeteroGraph()
# # print(len(nx_g.nodes()))
# # print(dg.num_nodes())
#
# print(type(dg.ndata['f']))
import matplotlib.pyplot as plt
import numpy as np

# #plot 1:
# xpoints = np.array([0, 6])
# ypoints = np.array([0, 100])
#
# plt.subplot(1, 2, 1)
# plt.scatter(xpoints,ypoints)
# plt.title("plot 1")
#
# #plot 2:
# x = np.array([1, 2, 3, 4])
# y = np.array([1, 4, 9, 16])
#
# plt.subplot(1, 2, 2)
# plt.scatter(x,y)
# plt.title("plot 2")
#
# plt.suptitle("RUNOOB subplot Test")
# plt.show()

losses = {'1': [np.array([1, 2, 3, 4.0]),np.array([2, 3, 4, 5])],
          '2': [np.array([2, 3, 4, 5]),np.array([4, 5, 6, 7])]}
for i in range(2):
    plt.subplot(1, 2, i+1)
    for t in losses:
        y = losses[t][i]
        x = np.array([j for j in range(4)])
        plt.plot(x, y)
    plt.title(str(i))
plt.show()