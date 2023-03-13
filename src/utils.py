import dgl
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SetSeed(seed):
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def get_random_mask(features, r, scale, dataset):

    if dataset == 'ogbn-arxiv' or dataset == 'minist' or dataset == 'cifar10' or dataset == 'fashionmnist':
        probs = torch.full(features.shape, 1 / r)
        mask = torch.bernoulli(probs)
        return mask

    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * scale

    probs = torch.zeros(features.shape).to(device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r

    mask = torch.bernoulli(probs)

    return mask


def top_k(raw_graph, k):
    _, indices = raw_graph.topk(k=int(k), dim=-1)

    mask = torch.zeros(raw_graph.shape).to(device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask

    return sparse_graph


def knn_fast(X, k, b):

    X = torch.nn.functional.normalize(X, dim=1, p=2)

    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()

    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b

        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)

        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b

    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))

    return rows, cols, values


def get_homophily(adj, labels, sparse):
    if sparse == 1:
        src, dst = adj.edges()
    else:
        src, dst = adj.detach().nonzero().t()
    homophily_ratio = 1.0 * torch.sum((labels[src] == labels[dst])) / src.shape[0]

    return homophily_ratio