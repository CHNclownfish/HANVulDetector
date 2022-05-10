import json
import os

graph_clean_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/clean_57_buggy_curated_0/'
graph_buggy_path = '/Users/xiechunyao/Downloads/crytic_byte_code/runtime/access_control/buggy/'
graphdata_clean = os.listdir(graph_clean_path)
graphdata_buggy = os.listdir(graph_buggy_path)
info = []
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
for x in d_info:
    print(d_info[x])