import json
from pathfinder import JsonPathFinder

file = '/Users/xiechunyao/dataset_08_Apr/denial_of_service/buggy_9.sol.json'
x = file.split('/')[4]
t = x[:x.find('.')]
with open(file) as f:
    data = f.read()
finder = JsonPathFinder(data)
#path_list = finder.find_all('bin')
path_list = finder.find_all('name')
for i,path in enumerate(path_list):
    print(path)