import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNConv_dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, x, adj):

        x = self.linear(x)
        x = torch.matmul(adj, x)

        return x


class GCNConv_dgl(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        return g.ndata['h']


class GCN_CLA(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, dropout_cla, dropout_adj, sparse):
        super(GCN_CLA, self).__init__()

        self.layers = nn.ModuleList()
        if sparse == 1:
            self.layers.append(GCNConv_dgl(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hid_dim, hid_dim))
            self.layers.append(GCNConv_dgl(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = dropout_adj
            
        else:
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv_dense(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hid_dim, hid_dim))
            self.layers.append(GCNConv_dense(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.sparse = sparse

    def forward(self, x, adj):
        if self.sparse == 1:
            adj.edata['w'] = F.dropout(adj.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            adj = self.dropout_adj(adj)

        for _, conv in enumerate(self.layers[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_cla, training=self.training)
        x = self.layers[-1](x, adj)

        return x


class GSL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, k, sparse):
        super(GSL, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.layers.append(nn.Linear(hid_dim, out_dim))

        self.k = k
        self.sparse = sparse
        self.in_dim = in_dim
        self.mlp_knn_init()

    def mlp_knn_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.in_dim))

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        
        if self.sparse == 1:
            rows, cols, values = knn_fast(h, self.k, 1000)

            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = F.relu(torch.cat((values, values)))

            adj = dgl.graph((rows_, cols_), num_nodes=h.shape[0], device='cuda')
            adj.edata['w'] = values_
        else:
            embeddings = F.normalize(h, dim=1, p=2)
            adj = torch.mm(embeddings, embeddings.t())

            adj = top_k(adj, self.k + 1)
            adj = F.relu(adj)

        return adj
    

class GCN_DAE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, dropout_cla, dropout_adj, mlp_dim, k, sparse):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()
        if sparse == 1:
            self.layers.append(GCNConv_dgl(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hid_dim, hid_dim))
            self.layers.append(GCNConv_dgl(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = dropout_adj
            
        else:
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv_dense(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hid_dim, hid_dim))
            self.layers.append(GCNConv_dense(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.sparse = sparse
        self.graph_generator = GSL(in_dim, math.floor(math.sqrt(in_dim * mlp_dim)), mlp_dim, nlayers, k, sparse)

    def get_adj(self, features):
        if self.sparse == 1:
            return self.graph_generator(features)
        else:
            adj = self.graph_generator(features)
            adj = (adj + adj.T) / 2

            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + 1e-10)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

    def forward(self, features, x):
        adj = self.get_adj(features)
        if self.sparse == 1:
            adj_dropout = adj
            adj_dropout.edata['w'] = F.dropout(adj_dropout.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            adj_dropout = self.dropout_adj(adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, adj_dropout)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_cla, training=self.training)
        x = self.layers[-1](x, adj_dropout)
        
        return x, adj

    def get_loss_homophily(self, g, logits, labels, train_mask, nclasses, num_hop, sparse):

        logits = torch.argmax(logits, dim=-1, keepdim=True)
        logits[train_mask, 0] = labels[train_mask]

        preds = torch.zeros(logits.shape[0], nclasses).to(device)
        preds = preds.scatter(1, logits, 1).detach()

        if sparse == 1:
            g.ndata['l'] = preds
            for _ in range(num_hop):
                g.update_all(fn.u_mul_e('l', 'w', 'm'), fn.sum(msg='m', out='l'))
            q_dist = F.log_softmax(g.ndata['l'], dim=-1)
        else:
            q_dist = preds
            for _ in range(num_hop):
                q_dist = torch.matmul(g, q_dist)
            q_dist = F.log_softmax(q_dist, dim=-1)

        loss_hom = F.kl_div(q_dist, preds)

        return loss_hom


