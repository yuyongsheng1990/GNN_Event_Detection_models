# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:12
# @Author : yysgz
# @File : offline_FinEvent_model.py
# @Project : FinEvent Models
# @Description :

import numpy as np
import json
import argparse
import torch
from time import localtime, strftime  # strftime() 函数用于格式化时间，返回以可读字符串表示的当地时间

import os
import torch.optim as optim
import gc
import time
from typing import List, Any

from P2_Layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter
from P2_Layers.S4_MarGNN_model import MarGNN
from P3_FinEvent_Model.S1_gen_dataset import create_offline_homodataset, create_multi_relational_graph, MySampler, save_embeddings
from P3_FinEvent_Model.S4_Evaluation import evaluate

def offline_FinEvent_model(train_i,
                           i,
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
    homo_data = create_offline_homodataset(args.data_path, args.result_path, [train_i, i])
    multi_r_data = create_multi_relational_graph(args.data_path, relation_ids, [train_i, i])
    num_relations = len(multi_r_data)

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # input dimension (300 in our paper)
    num_dim = homo_data.x.size(0)
    feat_dim = homo_data.x.size(1)

    # prepare graph configs for node filtering
    if args.is_initial:
        print('prepare node configures...')
        cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
        filter_path = save_path_i
    else:
        filter_path = args.data_path + str(i)

    if model is None:  # pre-training stage in our paper
        # print('Pre-Train Stage...')
        model = MarGNN((feat_dim, args.hidden_dim, args.out_dim, args.heads),
                       num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)

    # define sampler
    sampler = MySampler(args.sampler)
    # load model to device
    model.to(device)

    # initialize RL thresholds
    # RL_threshold: [[.5],[.5],[.5]]
    RL_thresholds = torch.FloatTensor(args.threshold_start0)

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

        # filter neighbor in adbvance to fit with neighbor sampling, return filtered neighbor index
        if epoch >= args.RL_start0 and args.sampler == 'RL_sampler':
            filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds, filter_path)
        else:
            filtered_multi_r_data = multi_r_data

        # step13.0: forward
        model.train()

        #     data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))
        train_num_samples, valid_num_samples, test_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(
            0), homo_data.test_mask.size(0)
        all_num_samples = train_num_samples + valid_num_samples + test_num_samples

        torch.save(homo_data.train_mask, save_path_i + '/train_mask.pt')
        torch.save(homo_data.val_mask, save_path_i + '/valid_mask.pt')
        torch.save(homo_data.test_mask, save_path_i + '/test_mask.pt')

        # mini-batch training
        num_batches = int(train_num_samples / args.batch_size) + 1
        for batch in range(num_batches):
            start_batch = time.time()
            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, train_num_samples)
            batch_nodes = homo_data.train_mask[i_start:i_end]
            batch_labels = homo_data.y[batch_nodes]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                         batch_size=args.batch_size)

            optimizer.zero_grad()  # 将参数置0

            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()

            # step13.1: metrics
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
            if batch % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)] \tloss: {:.6f}'.format(batch * args.batch_size, train_num_samples,
                                                                           100. * batch / ((
                                                                                                       train_num_samples // args.batch_size) + 1),
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
        validation_nmi = evaluate(extract_features[homo_data.val_mask],
                                  homo_data.y,
                                  indices=homo_data.val_mask,
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
