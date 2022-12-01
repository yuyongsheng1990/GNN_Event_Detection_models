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
from time import localtime, strftime  # strftime() 函数用于格式化时间，返回以可读字符串表示的当地时间

import os
import torch.optim as optim
import gc
import time
from typing import List, Any

from define_parameters import args_register

from P2_Layers.S2_TripletLoss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
from P3_FinEvent_Model.S2_FinEvent_model import FinEvent
from P3_FinEvent_Model.S4_Evaluation import AverageNonzeroTripletsMetric
from P4_Run_FinEvent_Offline.offline_FinEvent_model import offline_FinEvent_model
args = args_register()
args.data_path + '/embeddings_' + strftime('%m%d%H%M%S', localtime())

if __name__ == '__main__':
    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    if not os.path.exists(args.result_path):
        os.mkdir(args.data_path)
    embedding_save_path = args.result_path + '/offline_embeddings'
    if not os.path.exists(embedding_save_path):
        os.mkdir(embedding_save_path)
    print('embedding save path: ', embedding_save_path)

    # record hyper-parameters
    with open(embedding_save_path + '/args.txt', 'w') as f:
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
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(
            args.margin))  # margin used for computing tripletloss
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]
    # define detection stage
    Streaming = FinEvent(args)
    # pre-train stage: train on initial graph
    train_i = 0
    model, RL_thresholds = offline_FinEvent_model(train_i=train_i,
                                                  args=args,
                                                  i=0,
                                                  metrics=BCL_metrics,
                                                  embedding_save_path=embedding_save_path,
                                                  loss_fn=loss_fn,
                                                  model=None)
    print('model finished')