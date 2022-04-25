import json
from pathfinder import JsonPathFinder
from baseline_lstm.utils import readlabel
from baseline_lstm.utils import read_exportJsonName
from baseline_lstm.utils import contract2runtimbin
path_with_buggy = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/denial_of_service/contract_labels.json'

path_clean = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/clean_labels.json'
path2 = '/Users/xiechunyao/dataset_08_Apr/denial_of_service/denial_of_service_export/'

info = contract2runtimbin(path2,path_with_buggy,path_clean)
cnt_buggy = 0
cnt_clean = 0
for x in info:
    for k in x:
        if x[k]['label'] == 1:
            cnt_buggy += 1
        else:
            cnt_clean += 1
print(cnt_clean)
print(cnt_buggy)
