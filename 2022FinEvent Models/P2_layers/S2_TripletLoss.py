# -*- coding: utf-8 -*-
# @Time : 2022/11/29 16:54
# @Author : yysgz
# @File : S2_TripletLoss.py
# @Project : P3_utils Models
# @Description :

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

# Applies an average on seq, of shape(nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):  # 鉴别器
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)  # 双向现行变换x1*A*x2
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, m.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)  # 权值初始化方法，均分分布
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)  # torch.randn(size*)生成size维数组；expand是扩展到size_new数组；expand_as是扩展到像y的数组
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        return logits


# 计算triplet_loss损失函数
class OnlineTripletLoss(nn.Module):
    '''
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels
    Triplets are generated using triplet_selector objects that take embeddings and targets and return indices of triplets
    '''

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector  # selector选择器对象，含有get_triplets方法

    def forward(self, embeddings, target):  # (100, 192); target, (100, )
        triplets = self.triplet_selector.get_triplets(embeddings, target)  # 根据embeddings和labels返回最大loss的triplets list, (179, 3), positive:0,1; negative: 2.
        # if embeddings.is_cuda():
        #     triplets = triplets.cuda()
        # embeddings矩阵索引是单个元素，取行向量，多个行向量又组成矩阵！！
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # 第一列pre embedding 到 第二列pre embedding距离distance, (179,), =20 .pow(.5);
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # (179,), =30 ().pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)  # (179, ), margin=3. 这表示positive pair distance - negative pair distance，自然是希望区分度越大越好，即loss越小越好，梯度下降adam

        return losses.mean(), len(triplets)

class TripletSelector:
    '''
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets * 3]
    '''
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError  # 如果这个方法没有被子类重写，但是调用了，就会报错。

# 矩阵计算
def distance_matrix_computation(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix


# 具体实现三元损失函数triplets_loss，返回某标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
class FunctionNegativeTripletSelector(TripletSelector):
    '''
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    '''

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn  # 返回loss_values最大元素值的index的selector

    def get_triplets(self, embeddings, labels):  # (100, 192); target, (100, )
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = distance_matrix_computation(embeddings)  # pre embedding计算distance matrix (100, 100)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()  # 100
        triplets = []

        # embedding计算的distance matrix与labels计算loss，取最大loss_index
        # 对于每个标签label
        for label in set(labels):
            label_mask = (labels == label)  # numpy array, (100,), ([True, False, True, True])
            label_indices = np.where(label_mask)[0]  # 同一标签索引, label_index, (3, ) array([0, 2, 3], dtype=int64)
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[  # (97, )
                0]  # 其他标签索引, not_label_index, array([1], dtype=int64)
            anchor_pos_list = list(combinations(label_indices, 2))  # 2个元素的标签索引组合, list: 3, [(23, 66), (23, 79), (66, 79)]
            anchor_pos_list = np.array(anchor_pos_list)  # 转换成np.array才能进行slice切片操作, (3, 2)

            # 按照anchor_positive index从距离矩阵中抽取distance；0-index，array([0, 0, 2]);
            # 从distance_matrix中提取标签label的i-element与j-element距离。
            anchor_p_distances = distance_matrix[
                anchor_pos_list[:, 0], anchor_pos_list[:, 1]]  # 同一label不同位置之间的距离distance, (3, ),tensor([-1.1761,-0.8381,0.0099])
            for anchor_positive, ap_distance in zip(anchor_pos_list, anchor_p_distances):  # 每个标签下，元素组合、元素距离
                # 0表示ith元素到各个其他标签元素的距离。
                # 同一标签下(ith,jth)距离 - ith元素到其他标签元素的距离 + self.margin边际收益
                loss_values = ap_distance - distance_matrix[  # loss_values, (97, ); ap_distance是同一label第一个位置到第二个位置的距离
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin  # anchor_positive, (2, ) [23, 66]; label第一个位置到其他label位置的距离
                loss_values = loss_values.data.cpu().numpy()  # (97, )
                hard_neg_max_index = self.negative_selection_fn(loss_values)  # hard返回最大loss的索引, 4
                if hard_neg_max_index is not None:  # if 最大loss值非空
                    hard_negative = negative_indices[hard_neg_max_index]  # 返回最大loss对应的negative label 位置
                    # 对于谋标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])  # positive label位置, negative label位置

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)  # (179, 3)
        return torch.LongTensor(triplets)

# random，随机抽取一个非零实数
def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None  # np.random.choice函数默认随机抽取一个数

# hardest,抽取最大的一个数
def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

# 硬三元损失函数
def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative, cpu=cpu)

# 随机三元损失函数
def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative, cpu=cpu)