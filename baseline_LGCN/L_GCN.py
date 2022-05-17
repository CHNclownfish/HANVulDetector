import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl
import torch.nn.functional as F
class LGCNModel(nn.Module):
    def __init__(self, graph_input_dim, graph_hidden_dim1,graph_hidden_dim2, seq_input_dim, seq_hidden_dim, seq_layer_dim, output_dim):
        super(LGCNModel, self).__init__()

        self.seq_hidden_dim = seq_hidden_dim
        self.layer_dim = seq_layer_dim
        self.lstm = nn.LSTM(seq_input_dim, seq_hidden_dim, seq_layer_dim,batch_first=True)
        self.GCN1 = GraphConv(graph_input_dim, graph_hidden_dim1, norm='both', weight=True, bias=True)
        self.GCN2 = GraphConv(graph_hidden_dim1, graph_hidden_dim2, norm='both', weight=True, bias=True)
        self.fc = nn.Linear(seq_hidden_dim + graph_hidden_dim2, output_dim)
        # self.fc = nn.Linear(graph_hidden_dim2+, output_dim)

    def forward(self, dg, graph_seq_feature):

        gcn_hidden_embedding = F.relu(self.GCN1(dg, dg.ndata['f'])) # gcn_embedding = size(num_nodes * graph_hidden_dim)
        gcn_embedding = F.relu(self.GCN2(dg, gcn_hidden_embedding))
        #gcn_embedding = self.GCN(dg, dg.ndata['f'])

        lstm_embedding = []

        x = graph_seq_feature[0]

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, 1, self.seq_hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, 1, self.seq_hidden_dim).requires_grad_()

        for i, node_seq_feature in enumerate(graph_seq_feature):
            out, (hi, ci) = self.lstm(node_seq_feature, (h0.detach(), c0.detach()))
            hi = hi.view(self.seq_hidden_dim)
            lstm_embedding.append(hi)

        lstm_embeddings = torch.stack([hi for hi in lstm_embedding], 0)
        cat_embedding = torch.cat((gcn_embedding, lstm_embeddings), 1)
        #print(cat_embedding.size())
        # out.size() --> 100, 10
        h = cat_embedding
        #h = gcn_embedding
        with dg.local_scope():
            dg.ndata['h'] = h
            hg = dgl.mean_nodes(dg, 'h')

        return self.fc(hg), hg