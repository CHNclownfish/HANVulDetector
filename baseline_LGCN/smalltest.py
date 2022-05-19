import json
import os

files = ['/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/denial_of_service/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/access_control/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/arithmetic/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/front_running/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/reentrancy/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/time_manipulation/contract_labels.json',
         '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/unchecked_low_level_calls/contract_labels.json'
         ]
def createfile(files):
    for file in files:
        buggy_type = file.split('/')[-2]
        res = {}
        with open(file) as f:
            # here data is a list of obj
            data = json.load(f)
        for obj in data:
            source_file_name = obj['contract_name'][:obj['contract_name'].find('-')]
            contract_name = source_file_name + '_' + obj['contract_name'][obj['contract_name'].find('-')+1:obj['contract_name'].find('.')]
            res[contract_name] = obj['targets']
        path = '/Users/xiechunyao/dataset_08_Apr/' + buggy_type +'/'+ buggy_type + '_runtime_cfg/'
        dire = os.listdir(path)
        summe = 0
        c = 0
        b = 0
        con = []
        for dot_file in dire:
            if '.dot' in dot_file:
                contractName = dot_file.split('.')[0]
                if contractName in res:
                    d = {}
                    d['path'] = path + dot_file
                    d['target'] = res[contractName]
                    con.append(d)
                    summe += 1
                    if res[contractName] == 1:
                        b += 1
                    else:
                        c += 1

        #print(buggy_type,summe,b,c)

        path2 = '/Users/xiechunyao/dataset_08_Apr/'+ buggy_type +'/clean_balance_for_'+ buggy_type + '/'
        dire2 = os.listdir(path2)
        for dot_file2 in dire2:
            if '.dot' in dot_file2:
                contractName = dot_file2.split('.')[0]
                d = {}
                d['path'] = path2 + dot_file2
                d['target'] = 0
                c += 1
                summe += 1
                con.append(d)

        print(summe,b,c)
        tobewriten = buggy_type+ '_forLGCN.json'

        # with open(tobewriten,'w') as f1:
        #     json.dump(con,f1)
#denial_of_service_losses = [[0.46349728498772874, 0.8569148720689411, 0.651534612077897, 0.42774752586141346, 0.3410328329562789, 0.36556698899261475, 0.28965160722191646, 0.27903699556981604, 0.26181141095766514, 0.5685025266423573]
#[0.43439860650022905, 0.7752329826805975, 0.46959665680201507, 0.35345339103359874, 0.2859923105547571, 0.2500480404288662, 0.26334341831219304, 0.2286469336960274, 0.21183145087025326, 0.20669101630053374]
#[0.5065429190780572, 0.8758623595358658, 0.6068099342345252, 0.3984032273035878, 0.34561905443785695, 0.31056623428502655, 0.28727287597455614, 0.2801192675783174, 0.2824198019665145, 0.2698507122402116]
#[0.6649516923004747, 0.7762323002528957, 0.540418580790439, 0.39646855684539933, 0.29596914434585514, 0.27373790642334606, 0.27869097484591027, 0.26088452094975184, 0.2494125670357424, 0.2428320457454859]
#[0.5527734931065424, 0.8719320390373468, 0.6387748994166031, 0.4868552198749967, 0.3574260914028855, 0.3056278196454514, 0.28942659976746654, 0.2676853842858691, 0.27527005049159925, 0.27374658036933397]]
