# -*- coding: utf-8 -*-
# @Time : 2022/12/2 20:14
# @Author : yysgz
# @File : S3_save_edge_index.py
# @Project : FinEvent Models
# @Description :

import torch
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from scipy import sparse

from torch_geometric.data import Data
from torch_sparse.tensor import SparseTensor

import os
project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # # 获取上上级路径
offline_dataset_savepath = project_path + '/result/FinEvent result/offline dataset/'

# relations_ids = ['entity', 'userid', 'word'],分别读取这三个文件
def sparse_trans(datapath = None):
    relation = sparse.load_npz(datapath)  # (11971, 11971)
    all_edge_index = torch.tensor([], dtype=int)
    for node in range(relation.shape[0]):
        neighbor = torch.IntTensor(relation[node].toarray()).squeeze()  # IntTensor是torch定义的7中cpu tensor类型之一；
                                                                        # squeeze对数据维度进行压缩，删除所有为1的维度
        # del self_loop in advance
        neighbor[node] = 0
        neighbor_idx = neighbor.nonzero()  # 返回非零元素的索引
        neighbor_sum = neighbor_idx.size(0)  # 表示第0维度的数据量
        loop = torch.tensor(node).repeat(neighbor_sum, 1)  # repeat表示沿着指定的维度重复tensor的次数
        edge_index_i_j = torch.cat((loop, neighbor_idx), dim=1).t()  # cat表示拼接；t表示对二维矩阵进行转置
        self_loop = torch.tensor([[node], [node]])
        all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
        del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
    return all_edge_index

# save edge_index_[entity, userid, word].pt 文件
relations = ['entity', 'userid', 'word']
for relation in relations:
    relation_edge_index = sparse_trans(os.path.join(offline_dataset_savepath, 's_m_tid_%s_tid_matrix.npz' % relation))
    torch.save(relation_edge_index, offline_dataset_savepath + '/edge_index_%s.pt' % relation)