# -*- coding: utf-8 -*-
# @Time : 2022/11/2 15:01
# @Author : yysgz
# @File : TirpletLoss.py
# @Project : data_checking.py
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


class OnlineTripletLoss(nn.Module):
    '''
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels
    Triplets are generated using triplet_selector objects that take embeddings and targets and return indices of triplets
    '''

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        # if embeddings.is_cuda():
        #     triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

# 矩阵计算
def distance_matrix_computation(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(1,-1)
    return distance_matrix

class TripletSelector:
    '''
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets * 3]
    '''
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError  # 如果这个方法没有被子类重写，但是调用了，就会报错。


class FunctionNegativeTripletSelector(TripletSelector):
    '''
    For each positive pair, takes the hardes negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin userd in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    '''

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = distance_matrix_computation(embeddings)  # 计算distance matrix
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # all anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)

# 随机-loss随机负值
def random_hard_negative(loss_values):
    hard_negatives = np.where(loss > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

# 硬-loss最大负值
def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

# 硬负值三元组选择器
def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative, cpu=cpu)

# 随机负值三元组选择器
def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative, cpu=cpu)