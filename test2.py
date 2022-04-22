import json
import os
reentrancy_loss_me = [0.6710577673110806, 0.5691893248528731, 0.5017760518755092, 0.47306075124222724, 0.46180878481904014, 0.45571812765947617, 0.4512815716233654, 0.4475759717284656, 0.4441603803549145, 0.44189338358577157, 0.43964685614938376, 0.4377819888034194, 0.4362727016836527, 0.4349076290172143, 0.43360649485171576, 0.43247500184892873, 0.4315224930216543, 0.4306058286780827, 0.42968887135722356, 0.4289345834709582, 0.6406884370524375, 0.52467081683581, 0.46653234573905583, 0.44897505109671687, 0.4406863214127475, 0.43536796076528606, 0.43101153249440133, 0.4260477719706346, 0.4224850509895897, 0.4199100546347985, 0.41724517947200257, 0.4154791802961807, 0.41377152373144005, 0.41243632065636093, 0.41093894482407045, 0.40975161912835767, 0.4083280185367301, 0.40744015219102264, 0.4062201510771315, 0.40508216468342506, 0.6561917635749598, 0.567344669802267, 0.5109188839426784, 0.49170429070220617, 0.4849484319203213, 0.4810240590853281, 0.47789855533447423, 0.4751688040731872, 0.4726137731102158, 0.47044377282383987, 0.4685595858781064, 0.4668004312170822, 0.4652751799611772, 0.4638190179391474, 0.4623012898459298, 0.46179565979686915, 0.46060471371060513, 0.4593549375345961, 0.45918727882939286, 0.45842221980822867, 0.6773006756131242, 0.5847794287573032, 0.5266930654766114, 0.5042694082589654, 0.4964476082686002, 0.4921874969713087, 0.48950115900214125, 0.4863965707278349, 0.4840632946328904, 0.48230479409297305, 0.4808042274439723, 0.4800876676430547, 0.47895976743562435, 0.47817230085289575, 0.4774391612083447, 0.4767925293464971, 0.47618633962985946, 0.4756744208980382, 0.47514818018166033, 0.47468719888872246, 0.6594714168610611, 0.553623718338284, 0.49236594837128633, 0.470951411237077, 0.4631420291384788, 0.45893860167664724, 0.45537892223252513, 0.45247292494386193, 0.4503905704562984, 0.4482457576001563, 0.4459609191834442, 0.44480807428074076, 0.44333953060573195, 0.4420685703162013, 0.44108027290946583, 0.4403482740669231, 0.4394380453731713, 0.4385168776218969, 0.43786739312657497, 0.43733849538475034]

base1 = 'crytic-compile '
path = '/Users/xiechunyao/Downloads/MA_TH/smartbugs-master/dataset/reentrancy/'
data = os.listdir(path)
base2 = ' --export-format standard'

for x in data:
    if '.sol' in x:
        print(base1 + path + x + base2)