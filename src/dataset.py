import sys
import torch
import pickle
import warnings
import numpy as np
import pickle as pkl
import scipy.sparse as sp

from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, num_samples=20):

    if dataset_str == 'cornell' or dataset_str == 'texas' or dataset_str == 'wisconsin' or dataset_str == 'actor':
        features = pickle.load(open(f'../data/graphs/{dataset_str}_features.pkl', 'rb'))
        labels = pickle.load(open(f'../data/graphs/{dataset_str}_labels.pkl', 'rb'))
        data_mask = pickle.load(open(f'../data/graphs/{dataset_str}_tvt_nids.pkl', 'rb'))

        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        else:
            features = torch.FloatTensor(features)

        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
            nclasses = labels.shape[1]
        else:
            labels = torch.LongTensor(labels)
            nclasses = len(torch.unique(labels))
        
        train_mask = sample_mask(data_mask[0], labels.shape[0])
        val_mask = sample_mask(data_mask[1], labels.shape[0])
        test_mask = sample_mask(data_mask[2], labels.shape[0])

        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        return features.to(device), labels.to(device), features.shape[1], nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    for i in range(labels.shape[0]):
        if np.sum(labels[i]) != 1:
            labels[i] = np.array([1, 0, 0, 0, 0, 0])

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    if num_samples < 20:
        index_train = np.argwhere(train_mask==True)[:, 0]
        p_labels = labels.argmax(axis=1)
        n_class = int(p_labels.max() + 1)

        for i in range(n_class):
            index_label = np.argwhere(p_labels==i)[:, 0]
            index = np.intersect1d(index_train, index_label)
            np.random.shuffle(index)
            train_mask[index[num_samples:]] = False

    if num_samples > 20:
        train_mask = ~(val_mask | test_mask)
        index_train = np.argwhere(train_mask==True)[:, 0]
        p_labels = labels.argmax(axis=1)
        n_class = int(p_labels.max() + 1)

        for i in range(n_class):
            index_label = np.argwhere(p_labels==i)[:, 0]
            index = np.intersect1d(index_train, index_label)
            np.random.shuffle(index)
            train_mask[index[num_samples:]] = False

    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    return features.to(device), labels.to(device), nfeats, nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_ogb_data(dataset_str):
    
    dataset = PygNodePropPredDataset(dataset_str)

    data = dataset[0]
    features = data.x
    nfeats = data.num_features
    nclasses = dataset.num_classes
    labels = data.y

    split_idx = dataset.get_idx_split()

    train_mask = sample_mask(split_idx['train'], data.x.shape[0])
    val_mask = sample_mask(split_idx['valid'], data.x.shape[0])
    test_mask = sample_mask(split_idx['test'], data.x.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features.to(device), labels.to(device), nfeats, nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_mnist_data(dataset_str, num_samples=10):

    data_file = np.load('../data/{}/mnist.npz'.format(dataset_str))
    features, labels = data_file['x_test'][5000:], data_file['y_test'][5000:]
    data_file.close()

    features = features.reshape((features.shape[0], -1)) / 255.
    nclasses = int(labels.max() + 1)
    nrange = torch.arange(labels.shape[0])
    train_mask = torch.zeros(labels.shape[0], dtype=bool)

    for y in range(nclasses):
        label_mask = (labels == y)
        train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:num_samples]]] = True

    val_mask = ~train_mask
    val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
    test_mask = ~(train_mask | val_mask)
    test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features.to(device), labels.to(device), features.shape[1], nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_cifar10_data(dataset_str, num_samples=10):

    with open('../data/{}/test_batch'.format(dataset_str), 'rb') as fo:
        cifar_test_data_dict = pickle.load(fo, encoding='bytes')
    features = np.array(cifar_test_data_dict[b'data'])[5000:]
    labels = np.array(cifar_test_data_dict[b'labels'])[5000:]

    features = features.reshape((features.shape[0], -1)) / 255.
    nclasses = int(labels.max() + 1)
    nrange = torch.arange(labels.shape[0])
    train_mask = torch.zeros(labels.shape[0], dtype=bool)

    for y in range(nclasses):
        label_mask = (labels == y)
        train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:num_samples]]] = True

    val_mask = ~train_mask
    val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
    test_mask = ~(train_mask | val_mask)
    test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features.to(device), labels.to(device), features.shape[1], nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_fashionmnist_data(dataset_str, num_samples=10):
    features = np.load("../data/{}/data.npy".format(dataset_str)).astype(np.float32)[-5000:]
    labels = np.load("../data/{}/labels.npy".format(dataset_str)).astype(np.int32)[-5000:]

    features = features.reshape((features.shape[0], -1)) / np.max(features)
    nclasses = int(labels.max() + 1)
    nrange = torch.arange(labels.shape[0])
    train_mask = torch.zeros(labels.shape[0], dtype=bool)

    for y in range(nclasses):
        label_mask = (labels == y)
        train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:num_samples]]] = True

    val_mask = ~train_mask
    val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
    test_mask = ~(train_mask | val_mask)
    test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features.to(device), labels.to(device), features.shape[1], nclasses, train_mask.to(device), val_mask.to(device), test_mask.to(device)