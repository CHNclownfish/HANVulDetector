import json
import torch
import numpy as np
class dataGenerator:
    def __init__(self):
        s = '0123456789abcdef'
        self.str2int = {s[i]: i for i in range(len(s))}
        self.str2bit = {s[i]: [int(x) for x in (4 - len(bin(i)[2:])) * '0' + bin(i)[2:]] for i in range(len(s))}
        self.data = []

    def bts2int(self, bts):
        return self.str2int[bts[0]] * 16 + self.str2int[bts[1]]

    def jump(self, pos, bts):
        number = self.bts2int(bts)
        pos += 2
        if 96 <= number <= 127:
            pos += ((number - 96) + 1) * 2
        return pos

    def read(self, byte_seq):
        oneHotEncoder = []
        bitMaskEncoder = []
        pos = 0
        n = len(byte_seq)
        while pos < n:
            bts = byte_seq[pos] + byte_seq[pos+1]
            fe = [0 for _ in range(256)]
            fe[self.bts2int(bts)] = 1
            oneHotEncoder.append(fe)
            bitMaskEncoder.append(self.str2bit[bts[0]] + self.str2bit[bts[1]])
            pos = self.jump(pos, bts)
        return oneHotEncoder,bitMaskEncoder

    def encodeSelect(self, encode_type, info):
        for block in info:
            for contract in block:
                oneHotEncoder, bitMaskEncoder = self.read(block[contract]['bin-runtime'])
                if encode_type == 'onehot':
                    seq = {'feature': torch.tensor([oneHotEncoder]).float()}
                    l = torch.LongTensor([block[contract]['label']])
                    self.data.append((seq, l))
                elif encode_type == 'bitmask':
                    seq = {'feature': torch.tensor([bitMaskEncoder]).float()}
                    l = torch.LongTensor([block[contract]['label']])
                    self.data.append((seq, l))
        return np.array(self.data)