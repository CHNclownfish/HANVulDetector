from GraphGenerator import graphGenerator_slither
from GraphGenerator import graphGenerator_ethersolve
import torch as th
import numpy as np


class dataloader:
    def __init__(self,name_list,graphinfo):
        self.name_list = name_list
        self.graphinfo = graphinfo
        self.data = []

    def createdata(self,type):
        for con in self.name_list:
            print(con)
            if type == 'bytecode':
                gg = graphGenerator_ethersolve(self.graphinfo[con]['path'])
            elif type == 'sourcecode':
                gg = graphGenerator_slither(self.graphinfo[con]['path'])
            g = gg.reflectHeteroGraph()
            l = th.LongTensor([self.graphinfo[con]['label']])
            self.data.append((g,l))
        return np.array(self.data)
