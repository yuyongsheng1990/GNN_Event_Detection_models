import numpy as np
import torch
from sklearn import metrics
from munkres import Munkres
import networkx as nx


def decoding_cluster_from_tree(manifold, tree: nx.Graph, num_clusters, num_nodes, height):
    root = tree.nodes[num_nodes]
    root_coords = root['coords']
    dist_dict = {}  # for every height of tree
    for u in tree.nodes():  # distance between root and cluster anchor at each layer.
        if u != num_nodes:  # u is not root
            h = tree.nodes[u]['height']
            dist_dict[h] = dist_dict.get(h, {})
            dist_dict[h].update({u: manifold.dist(root_coords, tree.nodes[u]['coords']).numpy()})

    h = 1
    sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])  # sorted according to dis_increase
    count = len(sorted_dist_list)  #  6
    group_list = [([u], dist) for u, dist in sorted_dist_list]  # [ ([u], dist_u) ]
    while len(group_list) <= 1:
        h = h + 1
        sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
        count = len(sorted_dist_list)
        group_list = [([u], dist) for u, dist in sorted_dist_list]

    while count > num_clusters:
        group_list, count = merge_nodes_once(manifold, root_coords, tree, group_list, count)

    while count < num_clusters and h <= height:  # 若是没有达到label guided num_clusters, 就从最近的sub-tree cluster 开始分解出anchor ！！！呵呵。。。
        h = h + 1   # search next level for elements nodes contained in the sub-tree cluster.
        pos = 0  # positive or position
        while pos < len(group_list):
            v1, d1 = group_list[pos]  # node to split, vertical cluster and its distance, v1=[46], d1=0.081
            sub_level_set = []
            v1_coord = tree.nodes[v1[0]]['coords']  # v1 embedding, (1,3)
            for u, v in tree.edges(v1[0]):  # get v1[0] correlated edges, (u=46, v=44, 11, 15, 36)
                if tree.nodes[v]['height'] == h:
                    v_coords = tree.nodes[v]['coords']
                    dist = manifold.dist(v_coords, v1_coord).cpu().numpy()
                    sub_level_set.append(([v], dist))    # [ ([v], dist_v) ]
            if len(sub_level_set) <= 1:
                pos += 1
                continue
            sub_level_set = sorted(sub_level_set, reverse=False, key=lambda x: x[1])  # sorted sub_level_set!!不是group_list!! according to distance increase
            count += len(sub_level_set) - 1  # 8
            if count > num_clusters:
                while count > num_clusters:
                    sub_level_set, count = merge_nodes_once(manifold, v1_coord, tree, sub_level_set, count)  # 对最近一个超出num_classes的cluster nodes进行sub_cluster.
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set    # Now count == num_clusters
                break
            elif count == num_clusters:
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set
                break
            else:
                del group_list[pos]
                group_list += sub_level_set
                # pos += 1

    cluster_dist = {}
    # for i in range(num_clusters):
    #     u_list, _ = group_list[i]
    #     group = []
    #     for u in u_list:
    #         index = tree.nodes[u]['children'].tolist()
    #         group += index
    #     cluster_dist.update({k: i for k in group})
    for i in range(len(group_list)):  #
        u_list, _ = group_list[i]
        group = []
        for u in u_list:
            index = tree.nodes[u]['children'].tolist()
            group += index
        cluster_dist.update({k: i for k in group})
    results = sorted(cluster_dist.items(), key=lambda x: x[0])
    results = np.array([x[1] for x in results])
    return results


def merge_nodes_once(manifold, root_coords, tree, group_list, count):  # 合并group_list里面离root最远的的两个点，减少count。
    # group_list should be ordered ascend
    v1, v2 = group_list[-1], group_list[-2]
    merged_node = v1[0] + v2[0]
    merged_coords = torch.stack([tree.nodes[v]['coords'] for v in merged_node], dim=0)
    merged_point = manifold.Frechet_mean(merged_coords)
    merged_dist = manifold.dist(merged_point, root_coords).cpu().numpy()
    merged_item = (merged_node, merged_dist)
    del group_list[-2:]
    group_list.append(merged_item)
    group_list = sorted(group_list, reverse=False, key=lambda x: x[1])
    count -= 1
    return group_list, count
"""
Analysis of merge_nodes_once policy:
1. 为什么合并距离root最远的两个点？
    - 减少局部结构复杂度。通过合并最远的两个点可以减少系统的整体不确定性，即减少group_list的整体变动，使得新merged_node更接近root，减少整体熵。
    - 增强全局结构一致性。合并最远的点可以减少整体方差variance，make new merged node可能离root更近，减小hierarchy SE熵。
2. problems：
    - 不一定是最有合并策略。hierarchical clustering通常根据similarity (e.g., Euclidean, cosine, geodestic)合并，而只根据root距离可能会破坏局部结构。如果v1, v2属于不同cluster，强行合并只会导致错误聚类。
    - 信息损失。合并最远两个点可能破坏已经形成的dense cluster，比如cluster1就是最远的两个点之一。 --》一个更优policy是选择局部最相似点进行合并，而不只看root的全局距离。
    - 在实际中的可行性。在数据分布均匀的情况下，合并最远的点可能不会带来太大问题，但如果group_list里cluster的大小差异较大，强行合并会导致错误聚类结果。
3. 是否有better alternative：目的是减少count，同是保持信息熵SE的最优减少。
    - case-1: 局部最近邻合并。不是合并最远的两个点，而是合并group_list里最相近的两个点。这样保证新生成的cluster不会偏离原有的局部结构，avoid destroying the overall hierarchical clustering.
"""
def merge_nodes_once1(manifold, root_coords, tree, group_list, count):  # 最近邻合并：合并group_list里面最相近的两个点。
    # group_list should be ordered ascend
    min_dist = float('inf')
    min_pair = None

    # 找到 group_list 里最近的一对节点
    for i in range(len(group_list) - 1):  # 冒泡排序，快排？？
        v1, d1 = group_list[i]
        v2, d2 = group_list[i + 1]
        merged_coords = torch.stack([tree.nodes[v]['coords'] for v in v1 + v2], dim=0)
        merged_point = manifold.Frechet_mean(merged_coords)
        merged_dist = manifold.dist(merged_point, root_coords).cpu().numpy()

        if merged_dist < min_dist:
            min_dist = merged_dist
            min_pair = (i, v1, v2, merged_point, merged_dist)

    # 合并最相近的一对
    i, v1, v2, merged_point, merged_dist = min_pair
    merged_node = v1 + v2
    merged_item = (merged_node, merged_dist)

    del group_list[i:i + 2]  # 删除这两个点
    group_list.append(merged_item)
    group_list = sorted(group_list, key=lambda x: x[1])  # 重新排序

    count -= 1
    return group_list, count

class cluster_metrics:
    def __init__(self, trues, predicts):
        self.true_label = trues
        self.pred_label = predicts

    def get_new_predicts(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        self.new_predicts = new_predict

    def clusterAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        self.new_predicts = new_predict
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluateFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ami = metrics.adjusted_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)

        return nmi, ami, ari


def cal_AUC_AP(scores, trues):
    auc = metrics.roc_auc_score(trues, scores)
    ap = metrics.average_precision_score(trues, scores)
    return auc, ap