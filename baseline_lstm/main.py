from baseline_lstm.utils import contract2runtimbin
from baseline_lstm.seqGenerator import dataGenerator
from baseline_lstm.model_LSTM import LSTMModel
from sklearn.model_selection import KFold
import torch as th
from baseline_lstm.utils import evaluate
from baseline_lstm.showResults import showRes
path_base = '/Users/xiechunyao/dataset_08_Apr/front_running/front_running_export/'
labels_path = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/front_running/contract_labels.json'
labels_path_clean = '/Users/xiechunyao/Downloads/crytic_byte_code/contract_labels/clean_labels.json'

info = contract2runtimbin(path_base, labels_path, labels_path_clean)

dG = dataGenerator()
encode_type = 'onehot'
data = dG.encodeSelect(encode_type, info)

input_dim = 256
hidden_dim = 100
num_layer = 1
out_dim = 2
model = LSTMModel(input_dim,hidden_dim,num_layer,out_dim)
loss_fcn= th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001,
                          weight_decay=0.001)
kf = KFold(n_splits=5,random_state=1,shuffle=True)
losses = []
scores = []
cnt = 0
print('data prepared')
for train_idx, test_idx in kf.split(data):
    epoch_losses = []
    cnt += 1
    train_set = data[train_idx]
    test_set = data[test_idx]
    for epoch in range(15):
        epoch_loss = 0
        model.train()

        for i,(seq, l) in enumerate(train_set):
            if i % 20 == 0:
                print(i)

            features = seq['feature']
            logits = model(features)
            loss = loss_fcn(logits, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (i + 1)
        epoch_losses.append(epoch_loss)
        print(epoch,epoch_loss)

    losses.append(epoch_losses)
    score = evaluate(test_set, model)
    scores.append(score)
    print('this is fold ', cnt, ' mat:', score)

new_showResult = showRes(losses,scores)
new_showResult.showLoss()
new_showResult.showScores()