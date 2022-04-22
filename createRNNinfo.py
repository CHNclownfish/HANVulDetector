import json
from pathfinder import JsonPathFinder
import os

path = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/access_control/contract_labels.json'
with open(path) as f:
    data = json.load(f)
d = {}
for contract in data:
    x = contract['contract_name']
    block = x[:x.find('-')]
    if block not in d:
        d[block] = {}
        d[block]['number'] = 0
        d[block]['contracts'] = [contract]
    else:
        d[block]['contracts'].append(contract)
    d[block]['number'] += contract['targets']
a = []
for y in d:
    if d[y]['number'] != 0:
        a += d[y]['contracts']

