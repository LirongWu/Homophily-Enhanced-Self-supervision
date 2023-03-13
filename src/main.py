import os
import csv
import nni
import time
import json
import argparse

import torch
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loss_classification(model, features, adj, mask, labels):

    logits = model(features, adj)
    logits = F.log_softmax(logits, 1)

    loss = F.nll_loss(logits[mask], labels[mask], reduction='mean')
    return loss


def get_loss_reconstruction(model, features, mask, dataset):

    if dataset == 'ogbn-arxiv' or dataset == 'minist' or dataset == 'cifar10' or dataset == 'fashionmnist':
        masked_features = features * (1 - mask)
        logits, adj = model(features, masked_features)

        indices = mask > 0
        loss = F.mse_loss(logits[indices], features[indices], reduction='mean')

        return loss, adj

    logits, adj = model(features, features)

    indices = mask > 0
    loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')

    return loss, adj


def main():


    model_dae = GCN_DAE(nfeats, param['hid_dim_dae'], nfeats, param['nlayers'], param['dropout_cla'], param['dropout_adj'], param['mlp_dim'], param['k'], param['sparse']).to(device)
    model_cla = GCN_CLA(nfeats, param['hid_dim_cla'], nclasses, param['nlayers'], param['dropout_cla'], param['dropout_adj'], param['sparse']).to(device)

    optimizer_dat = torch.optim.Adam(model_dae.parameters(), lr=float(param['lr']), weight_decay=float(0.0))
    optimizer_cla = torch.optim.Adam(model_cla.parameters(), lr=float(param['lr_cla']), weight_decay=float(param['w_decay']))

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    hom_ratio_val = 0

    for epoch in range(1, param['epochs'] + 1):
        model_dae.train()
        model_cla.train()

        optimizer_dat.zero_grad()
        optimizer_cla.zero_grad()

        mask = get_random_mask(features, param['ratio'], param['scale'], param['dataset']).to(device)

        if epoch < param['epochs_pre']:
            loss_dae, adj = get_loss_reconstruction(model_dae, features, mask, param['dataset'])
            loss_cla = torch.tensor(0).to(device)
            loss_hom = torch.tensor(0).to(device)
        elif epoch < param['epochs_pre'] + param['epochs_hom']:
            loss_dae, adj = get_loss_reconstruction(model_dae, features, mask, param['dataset'])
            loss_cla = get_loss_classification(model_cla, features, adj, train_mask, labels)
            loss_hom = torch.tensor(0).to(device)
        else:
            loss_dae, adj = get_loss_reconstruction(model_dae, features, mask, param['dataset'])
            loss_cla = get_loss_classification(model_cla, features, adj, train_mask, labels)
            loss_hom = model_dae.get_loss_homophily(adj, logits, labels, train_mask, nclasses, param['num_hop'], param['sparse'])

        loss = loss_dae * param['alpha'] + loss_cla + loss_hom * param['beta']
        loss.backward()

        optimizer_dat.step()
        optimizer_cla.step()

        model_dae.eval()
        model_cla.eval()
        adj = model_dae.get_adj(features)
        logits = model_cla(features, adj)

        train_acc = ((logits[train_mask].max(dim=1).indices == labels[train_mask]).sum() / train_mask.sum().float()).item()
        val_acc = ((logits[val_mask].max(dim=1).indices == labels[val_mask]).sum() / val_mask.sum().float()).item()
        test_acc = ((logits[test_mask].max(dim=1).indices == labels[test_mask]).sum() / test_mask.sum().float()).item()
        hom_ratio = get_homophily(adj, labels, param['sparse']).item()

        if epoch >= param['epochs_pre']:
            if test_acc > test_best:
                test_best = test_acc

            if val_acc > val_best:
                val_best = val_acc
                test_val = test_acc
                hom_ratio_val = hom_ratio
                es = 0
            else:
                es += 1
                if es >= 200:
                    print("Early stopping!")
                    break

        if epoch % 10 == 0:
            print("\033[0;30;46m [{}] DAE: {:.5f}, CLA: {:.5f}, Hom: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Ratio: {:.4f}, {:.4f}\033[0m".format(
                                        epoch, loss_dae.item() * param['alpha'], loss_cla.item(), loss_hom.item() * param['beta'], loss.item(), train_acc, val_acc, test_acc, val_best, test_val, test_best, hom_ratio, hom_ratio_val))
     
    return test_acc, test_val, test_best, hom_ratio_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'cornell', 'texas', 'wisconsin', 'actor', 'mnist', 'cifar10', 'fashionmnist'])

    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--hid_dim_cla', type=int, default=64)
    parser.add_argument('--hid_dim_dae', type=int, default=512)
    parser.add_argument('--mlp_dim', type=int, default=1433)

    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--num_hop', type=float, default=5)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=1)

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--epochs_pre', type=int, default=400)
    parser.add_argument('--epochs_hom', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--w_decay', type=float, default=0.0005)
    parser.add_argument('--dropout_cla', type=float, default=0.4)
    parser.add_argument('--dropout_adj', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--save_mode', type=int, default=0)
    parser.add_argument('--data_mode', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--lr_cla', type=float, default=0.01)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
        param['mlp_dim'] = 1433
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
        param['mlp_dim'] = 3703
    if param['data_mode'] == 2:
        param['dataset'] = 'pubmed'
        param['mlp_dim'] = 500
    if param['data_mode'] == 3:
        param['dataset'] = 'ogbn-arxiv'
        param['mlp_dim'] = 128
    if param['data_mode'] == 4:
        param['dataset'] = 'cornell'
        param['mlp_dim'] = 1703
    if param['data_mode'] == 5:
        param['dataset'] = 'texas'
        param['mlp_dim'] = 1703
    if param['data_mode'] == 6:
        param['dataset'] = 'wisconsin'
        param['mlp_dim'] = 1703
    if param['data_mode'] == 7:
        param['dataset'] = 'actor'
        param['mlp_dim'] = 932
    if param['data_mode'] == 8:
        param['dataset'] = 'mnist'
        param['mlp_dim'] = 784
    if param['data_mode'] == 9:
        param['dataset'] = 'cifar10'
        param['mlp_dim'] = 3072
    if param['data_mode'] == 10:
        param['dataset'] = 'fashionmnist'
        param['mlp_dim'] = 784


    if os.path.exists("../param/best_parameters.json"):
        param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']]

    if param['dataset'] == 'ogbn-arxiv':
        features, labels, nfeats, nclasses, train_mask, val_mask, test_mask = load_ogb_data(param['dataset'])
    elif param['dataset'] == 'mnist':
        features, labels, nfeats, nclasses, train_mask, val_mask, test_mask = load_mnist_data(param['dataset'], param['num_samples'])
    elif param['dataset'] == 'fashionmnist':
        features, labels, nfeats, nclasses, train_mask, val_mask, test_mask = load_fashionmnist_data(param['dataset'], param['num_samples'])
    elif param['dataset'] == 'cifar10':
        features, labels, nfeats, nclasses, train_mask, val_mask, test_mask = load_cifar10_data(param['dataset'], param['num_samples'])
    else:
        features, labels, nfeats, nclasses, train_mask, val_mask, test_mask = load_data(param['dataset'], param['num_samples'])

    if param['save_mode'] == 0:
        SetSeed(param['seed'])
        test_acc, test_val, test_best, hom_ratio_val = main()
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []
        hom_ratio_val_list = []

        for seed in range(5):
            SetSeed(seed + param['seed'] * 5)
            test_acc, test_val, test_best, hom_ratio_val = main()
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            hom_ratio_val_list.append(hom_ratio_val)
            nni.report_intermediate_result(test_val)
        nni.report_final_result(np.mean(test_val_list))

    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    
    if param['save_mode'] == 0:
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
        results.append(str(hom_ratio_val))

    else:  
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        results.append(str(hom_ratio_val_list))
        results.append(str(np.mean(hom_ratio_val_list)))
        results.append(str(np.std(hom_ratio_val_list)))
    writer.writerow(results)