import json
from pathfinder import JsonPathFinder
# with open('results_wild.json') as f:
#     data = json.load(f)
# count = 0
# for contract in data:
#
#     obj = data[contract]
#     a = set()
#     for tool in obj['tools']:
#         for vul in obj['tools'][tool]['categories']:
#             a.add(vul)
#     if len(a) == 1 and 'reentrancy' in a:
#         count += 1
#         print(count,contract)
a = []
# base = '/Users/xiechunyao/uncheck_low_level_call/'
# file = base +  '0x610495793564aed0f9c7fc48dc4c7c9151d34fd6.sol.json'
file = '/Users/xiechunyao/clean_from_minh_json/clean_106.sol.json'
x = file.split('/')[4]
t = x[:x.find('.')]
with open(file) as f:
    data = f.read()
finder = JsonPathFinder(data)
#path_list = finder.find_all('bin')
path_list = finder.find_all('bin-runtime')

for i,path in enumerate(path_list):
    with open(file) as f1:
        obj = json.load(f1)
    print('java -jar "/Users/xiechunyao/Downloads/EtherSolve.jar" -r -d -o ' + t + '_'+ path[4]+'.dot')
    for key in path:
        obj = obj[key]
    print(obj)
# for i in range(1,53):
#     print('slither /Users/xiechunyao/clean_contract/clean_'+str(i)+'.sol --print cfg')
#     print('crytic-compile /Users/xiechunyao/clean_contract/clean_'+str(i)+'.sol --export-format standard')