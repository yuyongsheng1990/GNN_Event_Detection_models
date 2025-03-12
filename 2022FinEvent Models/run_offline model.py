# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:12
# @Author : yysgz
# @File : run_offline model.py
# @Project : FinEvent Models
# @Description :

import numpy as np
import json
import argparse
import torch

import os
import torch.optim as optim
import gc
import time
from typing import List
import os
project_path = os.getcwd()

from P2_layers.S2_TripletLoss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
from P2_layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter
from P4_model.MarGNN import MarGNN

from P3_utils.S2_gen_dataset import create_offline_homodataset, create_multi_relational_graph, MySampler, save_embeddings
from P3_utils.S4_Evaluation import AverageNonzeroTripletsMetric, evaluate

from P4_model.FinEvent_model import FinEvent

def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of initial-training/maintenance-training epochs.')
    parser.add_argument('--window_size', default=3, type=int, help='Maintain the model after predicting window_size blocks.')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stop if perfermance did not improve in the last patience epochs.')
    parser.add_argument('--margin', default=3, type=float, help='Margin for computing triplet losses')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size (number of nodes sampled to compute triplet loss in each batch)')
    parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension')
    parser.add_argument('--out_dim', default=64, type=int, help='Output dimension of tweet representation')
    parser.add_argument('--heads', default=4, type=int, help='Number of heads used in GAT')
    parser.add_argument('--validation_percent', default=0.2, type=float, help='Percentage of validation nodes(tweets)')
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False, action='store_true',
                        help='If true, use hardest negative messages to form triplets. Otherwise use random ones')
    parser.add_argument('--is_shared', default=False)
    parser.add_argument('--inter_opt', default='cat_w_avg')
    parser.add_argument('--is_initial', default=True)
    parser.add_argument('--sampler', default='RL_sampler')
    parser.add_argument('--cluster_type', default='kmeans', help='Types of clustering algorithms')  # DBSCAN

    # RL-0，第一个强化学习是learn the optimal neighbor weights
    parser.add_argument('--threshold_start0', default=[[0.2], [0.2], [0.2]], type=float,
                        help='The initial value of the filter threshold for state1 or state3')
    parser.add_argument('--RL_step0', default=0.02, type=float, help='The starting epoch of RL for state1 or state3')
    parser.add_argument('--RL_start0', default=0, type=int, help='The starting epoch of RL for state1 or state3')

    # RL-1，第二个强化学习是learn the optimal DBSCAN params.
    parser.add_argument('--eps_start', default=0.001, type=float, help='The initial value of the eps for state2')
    parser.add_argument('--eps_step', default=0.02, type=float, help='The step size of eps for state2')
    parser.add_argument('--min_Pts_start', default=2, type=int, help='The initial value of the min_Pts for state2')
    parser.add_argument('--min_Pts_step', default=1, type=int, help='The step size of min_Pts for state2')

    # other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help='Use cuda')
    parser.add_argument('--data_path', default= project_path + '/data', type=str, help='graph data path')  # 相对路径，.表示当前所在目录
    # parser.add_argument('--result_path', default='./result/offline result', type=str,
    #                     help='Path of features, labels and edges')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None, type=str,
                        help='File path that contains the training, validation and test masks')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')

    args = parser.parse_args(args=[])  # 解析参数

    return args

def offline_FinEvent_model(train_i,  # train_i=0
                           i,  # i=0
                           args,
                           metrics,
                           embedding_save_path,
                           loss_fn,
                           model=None,
                           loss_fn_dgi=None):
    # step1: make dir for graph i
    # ./incremental_0808//embeddings_0403005348/block_xxx
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)

    # step2: load data
    relation_ids: List[str] = ['entity', 'userid', 'word']
    homo_data = create_offline_homodataset(args.data_path, [train_i, i])  # (4762, 302), 包含x: feature embedding和y: label, generate train_slices (3334), val_slices (952), test_slices (476)
    # 返回entity, userid, word的homogeneous adj mx中non-zero neighbor idx。二维矩阵，node -> non-zero neighbor idx, (2,487962), (2,8050), (2, 51498)
    multi_r_data = create_multi_relational_graph(args.data_path, relation_ids, [train_i, i])
    num_relations = len(multi_r_data)  # 3

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # input dimension (300 in our paper)
    num_dim = homo_data.x.size(0)  # 4762
    feat_dim = homo_data.x.size(1)  # embedding dimension, 302

    # prepare graph configs for node filtering
    if args.is_initial:
        print('prepare node configures...')  # 计算neighbor node与node的相似度，并排序sorted
        cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
        filter_path = save_path_i
    else:
        filter_path = args.data_path + str(i)

    if model is None:  # pre-training stage in our paper
        # print('Pre-Train Stage...')
        # feat_dim=302; hidden_dim=128; out_dim=64; heads=4; inter_opt=cat_w_avg; is_shared=False
        model = MarGNN((feat_dim, args.hidden_dim, args.out_dim, args.heads),
                       num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)

    # define sampler
    sampler = MySampler(args.sampler)  # RL_sampler
    # load model to device
    model.to(device)

    # initialize RL thresholds
    # RL_threshold: [[.5],[.5],[.5]]
    RL_thresholds = torch.FloatTensor(args.threshold_start0)  # [[0.2], [0.2], [0.2]]

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # record training log
    message = '\n------Start initial training /maintaining using block' + str(i) + '------\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    # step12.0: record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # step12.1: record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # step12.2: record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # step12.3: record the time spent in mins on each epoch
    mins_train_epochs = []

    # step13: start training------------------------------------------------------------
    print('----------------------------------training----------------------------')
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0.0

        for metric in metrics:
            metric.reset()

        # Multi-Agent

        # RL_filter means extract limited sorted neighbors based on RL_threshold and neighbor similarity, return filtered node -> neighbor index
        if epoch >= args.RL_start0 and args.sampler == 'RL_sampler':
            filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds, filter_path)  # filtered 二维矩阵, (2,104479); (2,6401); (2,15072)
        else:
            filtered_multi_r_data = multi_r_data

        # step13.0: forward
        model.train()

        # data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))
        train_num_samples, valid_num_samples, test_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(
            0), homo_data.test_mask.size(0)  # 3354, 952, 476, 这里的train_mask指的是train_idx,不是bool类型
        all_num_samples = train_num_samples + valid_num_samples + test_num_samples

        torch.save(homo_data.train_mask, save_path_i + '/train_mask.pt')
        torch.save(homo_data.val_mask, save_path_i + '/valid_mask.pt')
        torch.save(homo_data.test_mask, save_path_i + '/test_mask.pt')

        # mini-batch training
        num_batches = int(train_num_samples / args.batch_size) + 1  # 34
        for batch in range(num_batches):
            start_batch = time.time()
            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, train_num_samples)
            batch_nodes = homo_data.train_mask[i_start:i_end]  # 100个train_idx
            batch_labels = homo_data.y[batch_nodes]  # 100个label

            # sampling neighbors of batch nodes
            # adjs是RL_sampler采样的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                         batch_size=args.batch_size)
            optimizer.zero_grad()  # 将参数置0

            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # x表示combined feature embedding, 302; pred, 其实是个embedding (100,192)

            loss_outputs = loss_fn(pred, batch_labels)  # (12.8063), 179
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()

            # step13.1: metrics
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
            if batch % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)] \tloss: {:.6f}'.format(batch * args.batch_size, train_num_samples,
                                                                           100. * batch / ((train_num_samples // args.batch_size) + 1),
                                                                           np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                # print(message)
                with open(save_path_i + '.log.txt', 'a') as f:
                    f.write(message)
                losses = []

            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            del pred, loss_outputs
            gc.collect()

            # step13.2: backward
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)

            del loss
            gc.collect()

        # step14: print loss
        total_loss /= (batch + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch, args.n_epochs, total_loss)

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        # step15: validation--------------------------------------------------------
        print('---------------------validation-------------------------------------')
        # inder the representations of all tweets
        model.eval()

        # we recommend to forward all nodes and select the validation indices instead
        extract_features = torch.FloatTensor([])
        num_batches = int(all_num_samples / args.batch_size) + 1

        # all mask are then splited into mini-batch in order
        all_mask = torch.arange(0, num_dim, dtype=torch.long)

        for batch in range(num_batches):
            start_batch = time.time()

            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                         batch_size=args.batch_size)

            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

            del pred
            gc.collect()

        # evaluate the model: conduct kMeans clustering on the validation and report NMI
        validation_nmi = evaluate(extract_features[homo_data.val_mask],  # val feature embedding
                                  homo_data.y,
                                  indices=homo_data.val_mask,  # 即index, validation时，index不变
                                  epoch=epoch,
                                  num_isolated_nodes=0,
                                  save_path=save_path_i,
                                  is_validation=True,
                                  cluster_type=args.cluster_type)
        all_vali_nmi.append(validation_nmi)

        # step16: early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model was at epoch ', str(best_epoch))
        else:
            wait += 1
        if wait >= args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # step17: save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')

    # step18: load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print('Best model loaded.')

    # del homo_data, multi_r_data
    torch.cuda.empty_cache()

    # test--------------------------------------------------------
    print('--------------------test----------------------------')
    model.eval()

    # we recommend to forward all nodes and select the validation indices instead
    extract_features = torch.FloatTensor([])
    num_batches = int(all_num_samples / args.batch_size) + 1

    # all mask are then splited into mini-batch in order
    all_mask = torch.arange(0, num_dim, dtype=torch.long)

    for batch in range(num_batches):
        start_batch = time.time()

        # split batch
        i_start = args.batch_size * batch
        i_end = min((batch + 1) * args.batch_size, all_num_samples)
        batch_nodes = all_mask[i_start:i_end]
        batch_labels = homo_data.y[batch_nodes]

        # sampling neighbors of batch nodes
        adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                     batch_size=args.batch_size)

        pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

        extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
        del pred
        gc.collect()

    save_embeddings(extract_features, save_path_i)

    test_nmi = evaluate(extract_features[homo_data.test_mask],
                        homo_data.y,
                        indices=homo_data.test_mask,
                        epoch=-1,
                        num_isolated_nodes=0,
                        save_path=save_path_i,
                        is_validation=False,
                        cluster_type=args.cluster_type)


if __name__ == '__main__':
    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    embedding_path = args.data_path + '/offline_embeddings/'
    if not os.path.exists(embedding_path):
        os.mkdir(embedding_path)
    print('embedding path: ', embedding_path)

    # record hyper-parameters
    with open(embedding_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)  # __dict__将模型参数保存成字典形式；indent缩进打印

    print('Batch Size:', args.batch_size)
    print('Intra Agg Mode:', args.is_shared)
    print('Inter Agg Mode:', args.inter_opt)
    print('Reserve node config?', args.is_initial)

    # load number of message in each blocks
    # e.g. data_split = [  500  ,   100, ...,  100]
    #                    block_0  block_1    block_n
    # define loss function，调用forward(embeddings, labels)方法，最终loss返回单个值
    # contrastive loss in our paper
    if args.use_hardest_neg:
        # HardestNegativeTripletSelector返回某标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))  # margin used for computing tripletloss
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]
    # define detection stage
    Streaming = FinEvent(args)
    # pre-train stage: train on initial graph
    train_i = 0
    offline_FinEvent_model(train_i=train_i,
                          args=args,
                          i=0,
                          metrics=BCL_metrics,
                          embedding_save_path=embedding_path,
                          loss_fn=loss_fn,
                          model=None)
    print('model finished')