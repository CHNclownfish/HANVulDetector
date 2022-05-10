import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.ed_d = {'CF':0,'True':1,'False':2}

    def forward(self, g, h):
        #semantic_embeddings = []
        semantic_embeddings = {}
        for n in g.ntypes:
            semantic_embeddings[n] = []
        result = {}
        possible_metapath = []
        # for s,ed,e in g.canonical_etypes:
        #     if ed in ['FW','DF','FB','CF']:
        #         meta_p = [(s,ed,e),(e,ed[1]+ed[0],s)]
        #         possible_metapath.append(meta_p)
        for s,ed,e in g.canonical_etypes:
            if ed in ['CF','False''True'] and [(s,ed,e),(e,ed[::-1],s)] not in possible_metapath:
                meta_p = [(s,ed,e),(e,ed[::-1],s)]
                possible_metapath.append(meta_p)

        #print(possible_metapath)


        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in possible_metapath:

                self._cached_coalesced_graph[meta_path[0]] = dgl.metapath_reachable_graph(g, meta_path)

                #new_g = self._cached_coalesced_graph[meta_path[0]]
                #print('this is new_g',new_g)


        for i, meta_path in enumerate(possible_metapath):
            new_g = self._cached_coalesced_graph[meta_path[0]]
            s,ed,e = meta_path[0]
            nodetype = s
            number = self.ed_d[ed]

            #print(h[nodetype])
            semantic_embeddings[nodetype].append(self.gat_layers[number](new_g, h[nodetype]).flatten(1))
            #print(self.gat_layers[i](new_g, h[nodetype]).flatten(1))
            #print(False)
        for nodetype in semantic_embeddings:
            # if semantic_embeddings[nodetype]  == []:
            #     l = nn.Linear(57,64)
            #     en_h = l(h[nodetype])
            #     semantic_embeddings[nodetype].append(en_h)
            #     semantic_embeddings[nodetype].append(h[nodetype])
            if semantic_embeddings[nodetype] != []:

                info = semantic_embeddings[nodetype]
                semantic_embeddings[nodetype] = torch.stack(info, dim=1)
                #print(semantic_embeddings[nodetype])
                result[nodetype] = self.semantic_attention(semantic_embeddings[nodetype])# (N, M, D * K)


        return result                           # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads,
                 seq_input_dim, seq_hidden_dim, seq_layer_dim, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.seq_hidden_dim = seq_hidden_dim
        self.layer_dim = seq_layer_dim
        self.lstm = nn.LSTM(seq_input_dim, seq_hidden_dim, seq_layer_dim,batch_first=True)
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h, graph_seq_feature):
        # h = g.ndata['f']
        lstm_embedding = []
        x = graph_seq_feature[0]
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.seq_hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.seq_hidden_dim).requires_grad_()
        for i, node_seq_feature in enumerate(graph_seq_feature):
            out, (hi, ci) = self.lstm(node_seq_feature, (h0.detach(), c0.detach()))
            hi = hi.view(self.seq_hidden_dim)
            lstm_embedding.append(hi)

        lstm_embeddings = torch.stack([hi for hi in lstm_embedding], 0)

        for gnn in self.layers:
            h = gnn(g, h)


        with g.local_scope():
            for ntype in h.keys():
                g.nodes[ntype].data['h'] = h[ntype]
            hg = 0
            for ntype in h.keys():
                hg = hg + dgl.max_nodes(g, 'h', ntype=ntype)
            #print(hg)
            return self.predict(hg)

