# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:07
# @Author : yysgz
# @File : FinEvent_model.py
# @Project : FinEvent Models
# @Description :

from typing import List
import time
import torch.optim as optim
import gc  # garbage cleaning package
import numpy as np
import os
import torch
# args = args_register()
# print(args)

from P2_layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter
from P4_model.MarGNN import MarGNN
from P3_utils.S2_gen_dataset import create_multi_relational_graph, MySampler, save_embeddings, create_homodataset
from P3_utils.S4_Evaluation import evaluate


class FinEvent():
    def __init__(self, args) -> None:
        # register args
        self.args = args

    def inference(self,  # inference = prediction
                  train_i, i,
                  metrics,
                  embedding_save_path,
                  loss_fn,
                  model: MarGNN,
                  RL_thresholds=None,
                  loss_fn_dgi=None):
        # make dir for graph i
        # ./incremental_0808//embeddings_0403005348/block_xxx
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        relation_ids: List[str] = ['entity', 'userid', 'word']  # typing package
        homo_data = create_homodataset(self.args.data_path, [train_i, i],
                                       self.args.validation_percent)  # get training,validation,test数据
        multi_r_data = create_multi_relational_graph(self.args.data_path, relation_ids,
                                                     [train_i, i])  # load relation data
        num_relations = len(multi_r_data)

        device = torch.device('cuda' if torch.cuda.is_available() and self.args.use_cuda else 'cpu')

        # input dimension (300 in our paper)
        features = homo_data.x  # x是features embeddings
        feat_dim = features.size(1)

        # prepare graph configs for node filtering
        if self.args.is_initial:
            print('prepare node configures...')
            RL_neighbor_filter(multi_r_data, homo_data.x, save_path_i)
            filter_path = save_path_i
        else:
            filter_path = self.args.data_path + str(i)

        if model is None:
            assert 'Cannot fine pre-trained model'

        # directly predict
        message = '\n-----------------Directly predict on block' + str(i) + '-----------------\n'
        print(message)
        print('RL Threshold using in this block:', RL_thresholds)

        model.eval()

        test_indices, labels = homo_data.test_mask, homo_data.y
        test_num_samples = test_indices.size(0)

        sampler = MySampler(self.args.sampler)

        # filter neighbor in advance to fit with neighbor sampling
        filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds,
                                                   filter_path) if RL_thresholds is not None and \
                                                                   self.args.sampler == 'RL_sampler' else multi_r_data

        # batch testing
        extract_features = torch.FloatTensor([])
        num_batches = int(test_num_samples / self.args.batch_size) + 1
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，不自动反向传播求导
            for batch in range(num_batches):
                start_batch = time.time()

                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, test_num_samples)
                batch_nodes = test_indices[i_start:i_end]

                # sampling neighbors of batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                             batch_size=self.args.batch_size)
                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)
                batch_seconds_spent = time.time() - start_batch

                # for we haven't shuffle the test indices(see utils.py)
                # the output embeddings can be simply stacked together
                extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

                del pred
                gc.collect()  # 清除缓存

        save_embeddings(extract_features, save_path_i)
        # 返回评价指标nmi，ami，ari
        test_nmi = evaluate(extract_features,
                            labels,
                            indices=test_indices,
                            epoch=-1,  # just for test
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            is_validation=False,
                            cluster_type=self.args.cluster_type)
        del homo_data, multi_r_data, features, filtered_multi_r_data
        torch.cuda.empty_cache()  # 释放显存

        return model

    # train on initial/maintenance graphs, t==0 or t % window_size == 0 in this paper
    def initial_maintain(self,
                         train_i, i,
                         metrics,
                         embedding_save_path,
                         loss_fn,
                         model=None,
                         loss_fn_dgi=None):
        '''
        :param i:
        :param data_split:
        :param metrics:
        :param embedding_save_path:
        :param loss_fn:
        :param model:
        :param loss_fn_dgi:
        :return:
        '''
        # make dir for graph i
        # ./incremental_0808//embeddings_0403005348/block_xxx
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        relation_ids: List[str] = ['entity', 'userid', 'word']
        homo_data = create_homodataset(self.args.data_path, [train_i, i], self.args.validation_percent)
        multi_r_data = create_multi_relational_graph(self.args.data_path, relation_ids, [train_i, i])  # relation data
        num_relations = len(multi_r_data)

        device = torch.device('cuda' if torch.cuda.is_available() and self.args.use_cuda else 'cpu')

        # input dimension (300 in our paper)
        num_dim = homo_data.x.size(0)  # embeddings num
        feat_dim = homo_data.x.size(1)  # embeddings dimension

        # prepare graph configs for node filtering
        if self.args.is_initial:
            print('prepare node %configures...')
            cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
            filter_path = save_path_i
        else:
            filter_path = self.args.data_path + str(i)

        if model is None:  # pre-training stage in our paper
            # print('Pre-Train Stage')
            model = MarGNN((feat_dim, self.args.hidden_dim, self.args.out_dim, self.args.heads),
                           num_relations=num_relations, inter_opt=self.args.inter_opt, is_shared=self.args.is_shared)

        # define sampler
        sampler = MySampler(self.args.sampler)  # top-p neighbors
        # load model to device
        model.to(device)

        # initialize RL thresholds
        RL_thresholds = torch.FloatTensor(self.args.threshold_start0)  # [[0.2],[0.2],[0.2]]

        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # record training log
        message = '\n------------- Start initial training / maintaining using block ' + str(i) + '----------\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

        # record the highest validation nmi ever got for early stopping
        best_vali_nmi = 1e-9
        best_epoch = 0
        wait = 0
        # record validation nmi of all epochs before early stop
        all_vali_nmi = []
        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []

        # step13: start epoch training
        for epoch in range(self.args.n_epochs):  # n_epochs=50
            start_epoch = time.time()
            losses = []
            total_loss = 0.0

            for metric in metrics:
                metric.reset()

            # Multi-Agent

            # filter neighbor in advance to fit with neighbor sampling
            if epoch >= self.args.RL_start0 and self.args.sampler == 'RL_sampler':  # RL_start0=0
                filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds, filter_path)
            else:
                filtered_multi_r_data = multi_r_data

            model.train()
            train_num_samples, valid_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(0)
            all_num_samples = train_num_samples + valid_num_samples

            # mini-batch training------------------------------------------------------------------
            num_batches = int(train_num_samples / self.args.batch_size) + 1  # batch_size=100
            for batch in range(num_batches):
                start_batch = time.time()
                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, train_num_samples)
                batch_nodes = homo_data.train_mask[i_start:i_end]  # 从training data中取出mini-batch用于训练
                batch_labels = homo_data.y[batch_nodes]

                # sampling neighobrs from mini-batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_ids=batch_nodes, sizes=[-1, -1],
                                             batch_size=self.args.batch_size)

                optimizer.zero_grad()

                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

                loss_outputs = loss_fn(pred, batch_labels)

                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                losses.append(loss.item())

                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch % self.args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch * self.args.batch_size,
                                                                              train_num_samples, 100. * batch / ((
                                                                                                                             train_num_samples // self.args.batch_size) + 1),
                                                                              np.mean(losses))

                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []
                del pred, loss_outputs
                gc.collect()

                loss.backward()
                optimizer.step()  # 更新参数

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)

                del pred
                gc.collect()

            # step 14: print loss
            total_loss /= (batch + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch, self.args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)

            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            # validation-------------------------------------------------------------------------
            # infer the representation of all tweets
            model.eval()

            # we recommand to forward all nodes and select the validation indices instead
            extract_features = torch.FloatTensor([])

            num_batches = int(all_num_samples / self.args.batch_size) + 1

            # all mask are then splited into mini-batch in order
            all_mask = torch.arange(0, num_dim, dtype=torch.long)

            for batch in range(num_batches):
                start_batch = time.time()
                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, all_num_samples)
                batch_nodes = all_mask[i_start:i_end]  #
                batch_labels = homo_data.y[batch_nodes]

                # sampling neighbors of batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                             batch_size=self.args.batch_size)

                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

                extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
                del pred
                gc.collect()
            # save_embeddings(extract_reatures, save_path_i)
            # evaluate the model: conduct kMeans clustering on the validation and report NMI
            validation_nmi = evaluate(extract_features[homo_data.val_mask],
                                      homo_data.y,
                                      epoch=epoch,
                                      num_isolated_nodes=0,
                                      save_path=save_path_i,
                                      is_validation=True,
                                      cluster_type=self.args.cluster_type)
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
                torch.save(model.state_dict(), p)  # 保存模型，OrderDict存储网络结构的名字和对应的参数
                print('Best model saved after epoch ', str(epoch))
            else:
                wait += 1
            if wait >= self.args.patience:
                print('Saved all_mins_spent')
                print('Early stopping at epoch ', str(epoch))
                print('Best model was at epoch ', str(best_epoch))
                break
            # end one epoch

        # save all validation mi
        np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
        # save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs')
        # save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Best model loaded.')

        del homo_data, multi_r_data
        torch.cuda.empty_cache()  # 释放显存

        return model, RL_thresholds