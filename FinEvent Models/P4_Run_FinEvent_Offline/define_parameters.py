# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:11
# @Author : yysgz
# @File : define_parameters.py
# @Project : FinEvent Models
# @Description :
'''
step 1. run utils/generate_initial_features.py to generate the initial features for the messages

step 2. run utils/custom_message_graph.py to construct incremental message graphs. To construct small message graphs for test purpose, set test=True when calling construct_incremental_dataset_0922(). To use all the messages (see Appendix of the paper for a statistic of the number of messages in the graphs), set test=False.

step 3. run utils/S3_save_edge_index.py in advance to acclerate the training process.

step 4. run offline.py
'''

import numpy as np
import json
import argparse
import torch
from time import localtime, strftime  # strftime() 函数用于格式化时间，返回以可读字符串表示的当地时间

import os
project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # # 获取上上级路径

import torch.optim as optim
import gc
import time
from typing import List, Any


def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--n_epochs', default=50, type=int,
                        help='Number of initial-training/maintenance-training epochs.')
    parser.add_argument('--window_size', default=3, type=int,
                        help='Maintain the model after predicting window_size blocks.')
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
    parser.add_argument('--data_path', default=project_path + '/result/FinEvent result/offline dataset/', type=str, help='graph data path')
    parser.add_argument('--result_path', default=project_path + '/result/FinEvent result/offline result', type=str,
                        help='Path of features, labels and edges')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None, type=str,
                        help='File path that contains the training, validation and test masks')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')

    args = parser.parse_args(args=[])  # 解析参数

    return args