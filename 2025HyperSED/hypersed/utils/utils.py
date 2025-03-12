import numpy as np
import torch
from collections import Counter

import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch_scatter import scatter_sum
from sklearn.cluster import KMeans
from hypersed.models.hyper_kmeans import PoincareKMeans


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')

def Frechet_mean_poincare(manifold, embeddings, weights=None, keepdim=False):
    z = manifold.from_poincare(embeddings)
    if weights is None:
        z = torch.sum(z, dim=0, keepdim=True)
    else:
        z = torch.sum(z * weights, dim=0, keepdim=keepdim)
    denorm = manifold.inner(None, z, keepdim=keepdim)
    denorm = denorm.abs().clamp_min(1e-8).sqrt()
    z = z / denorm
    z = manifold.to_poincare(z).to(embeddings.device)
    return z

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=0.2, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    sparse_adj = torch.masked_fill(dense_adj, ~mask, value=0.)
    return sparse_adj

def adjacency2index(adjacency, weight=False, topk=False, k=10):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    if topk and k:
        adj = graph_top_K(adjacency, k)
    else:
        adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight
    else:
        return edge_index

def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    adjacency = torch.zeros(N, N).to(edge_index.device)
    m = edge_index.shape[1]
    if weight is None:
        adjacency[edge_index[0], edge_index[1]] = 1
    else:
        adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    if is_sparse:
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    return adjacency

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = np.ones_like(sparse_mx.data)  # element value of adj_label for BCE_loss should be 0 and 1.
    shape = sparse_mx.shape
    return coords, values, shape

def tensor_to_sparse(sparse_tensor, size = (500, 500)):
    # 确保sparse_tensor是稀疏格式
    sparse_tensor = sparse_tensor.coalesce()  # coalesce()函数用于压缩合并重复索引，并整理sparse mx的数据结构。
    # 获取稀疏张量的索引和值
    indices_np = sparse_tensor.indices().numpy()
    values_np = sparse_tensor.values().numpy()

    # 将索引转换为行和列
    row = indices_np[0]
    col = indices_np[1]

    # 创建 csr_matrix
    csr = csr_matrix((values_np, (row, col)), shape=size)
    return csr

def getOtherByedge(edge_index, num_nodes):
    weight = torch.ones(edge_index.shape[1])
    degrees = scatter_sum(weight, edge_index[0])
    adj = index2adjacency(num_nodes, edge_index, weight, is_sparse=True)

    return adj, degrees, weight

def getNewPredict(predicts, C):
    N = predicts.shape[0]
    P = np.zeros(N)
    C = C.cpu().numpy()
    for i in range(C.shape[0]):
        j = np.where(C[i] == 1)[0][0]
        P[i] = predicts[j]

    return P

def getC(Z, M):
    N = Z.size(0)
    Z_np = Z.detach().cpu().numpy()

    # 随机选择M个数据点作为初始锚点，并且确保每个聚类簇中至少有一个数据点
    initial_indices = np.random.choice(N, M, replace=False)
    initial_anchors = Z_np[initial_indices]

    # 对得到的特征进行kmeans聚类，使用初始锚点
    kmeans = KMeans(n_clusters=M, init=initial_anchors, n_init=1, max_iter=200, tol=1e-10)
    kmeans.fit(Z_np)
    labels = kmeans.labels_
    labels = torch.tensor(labels, device=Z.device, dtype=torch.long)

    C = torch.zeros(N, M, device=Z.device)
    C[torch.arange(N, dtype=torch.long), labels] = 1

    return C

def getNewPredict(predicts, C):  # 将44 anchors predicts maps to 893 nodes.
    P = np.zeros(C.shape[0])
    C = C.cpu().numpy()
    for i in range(C.shape[0]):
        j = np.where(C[i] == 1)[0][0]
        P[i] = predicts[j]

    return P

def get_anchor(Z, A, M):
    N = Z.size(0)
    Z_np = Z.detach().cpu().numpy()

    kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
    kmeans.fit(Z_np)
    labels = kmeans.labels_
    labels = torch.tensor(labels, device=Z.device, dtype=torch.long)

    C = torch.zeros(N, M, device=Z.device)
    C[torch.arange(N, dtype=torch.long), labels] = 1

    # 计算锚点图的邻接矩阵
    A_anchor = C.T @ A @ C
    A_anchor.fill_diagonal_(0)

    # 计算锚点的表示
    X_anchor = torch.zeros(M, Z.size(1), device=Z.device)
    for i in range(M):
        cluster_points = Z[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:  # 检查簇中是否存在数据点
            X_anchor[i] = cluster_points.mean(dim=0)

    return A_anchor, X_anchor, C

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

def artanh(x):
    return Artanh.apply(x)

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()



def mobius_add(x, y, c, dim=-1, eps=1e-5):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def hyperbolic_distance1(p1, p2, c=1):
    sqrt_c = c ** 0.5
    dist_c = artanh(
        sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
    )
    dist = dist_c * 2 / sqrt_c
    return dist ** 2

def hyperbolic_distance(z_u, z_h, eps=1e-5):
    norm_z_u = torch.norm(z_u, p=2, dim=-1)
    norm_z_h = torch.norm(z_h, p=2, dim=-1)

    # Ensure norms are less than 1 to satisfy the Poincaré ball constraint
    norm_z_u = torch.clamp(norm_z_u, max=1 - eps)
    norm_z_h = torch.clamp(norm_z_h, max=1 - eps)

    # Compute the squared Euclidean distance
    euclidean_dist_sq = torch.sum((z_u - z_h) ** 2, dim=-1)

    # Compute the hyperbolic distance
    numerator = 2 * euclidean_dist_sq
    denominator = (1 - norm_z_u ** 2) * (1 - norm_z_h ** 2)
    arg_acosh = 1 + numerator / denominator

    # Ensure the argument of acosh is >= 1
    arg_acosh = torch.clamp(arg_acosh, min=1 + eps)

    return torch.acosh(arg_acosh)


def contrastive_loss(manifold, z_u, z_h, z_h_all, temperature=0.5):

    dist1 = manifold.dist2(z_u, z_h)
    dist = manifold.dist2(z_u, z_h_all)

    loss = -torch.log(torch.exp(dist1 / temperature) / torch.exp(dist / temperature).sum())

    return loss.mean()


def L_ConV(manifold, z_u, z_h, z_e, N_s, temperature=0.5):
    """
    Compute the overall contrastive loss.
    """
    loss = 0.0
    for i in range(N_s):
        loss += (contrastive_loss(manifold, z_u[i], z_h[i], z_h, temperature) +
                 contrastive_loss(manifold, z_h[i], z_e[i], z_e, temperature) +
                 contrastive_loss(manifold, z_e[i], z_u[i], z_u, temperature))
    return loss / (3 * N_s)


def get_agg_feauture(manifold, x1, x2, x3):
    
    x = torch.zeros_like(x1)

    for i in range(x1.shape[0]):
        x_chunk = torch.stack((x1[i], x2[i], x3[i]), dim=0)  # (3, hidden)
        x[i] = manifold.Frechet_mean(x_chunk, keepdim=False)  # (hidden,)
    
    return x


def get_euc_anchors(features, adj, anchor_rate, diag, true_labels):

    num_nodes = features.shape[0]  # 893
    num_anchor = int(num_nodes / anchor_rate)  # 44

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)  # n_init is run times, then return best output.
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_  # (893,)

    labels = get_cluster_labels(anchor_predictions, true_labels)  # get true anchor labels, 44

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)  # (893,)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)  # node2anchor matrix, (893, 44)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1  # node corresponding to anchor.

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)  # (44, 384), anchor embedding = node embeddings' mean.
    for i in range(num_anchor):  # num_anchor=44
        cluster_points = features[torch.where(C[:, i] == 1)]  # (43,384)
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)
    
    anchor_adj = C.T @ adj.to_dense() @ C  # (44, 44)
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C, labels


def get_cluster_labels(cluster_predictions, true_labels):
    unique_clusters = set(cluster_predictions)  # 44
    cluster_labels = {}
    labels = []

    for cluster in unique_clusters:
        cluster_indices = [i for i, pred in enumerate(cluster_predictions) if pred == cluster]  # cluster索引
        labels_in_cluster = [true_labels[idx] for idx in cluster_indices]
        most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]  # collections.Counter计算最常见的label，即多数类标签。
        cluster_labels[cluster] = most_common_label
        labels.append(most_common_label)

    unique_cluster_labels = set(cluster_labels.values())

    return labels


def get_euc_anchors_alladj(features, adj, anchor_rate, diag, thres):

    num_nodes = features.shape[0]
    num_anchor = int(num_nodes / anchor_rate)

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)
    for i in range(num_anchor):
        cluster_points = features[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)

    corr_matrix_np = np.corrcoef(features, rowvar=True)
    adjacency_matrix_np = np.where(corr_matrix_np > thres, corr_matrix_np, 0)
    adj = torch.tensor(adjacency_matrix_np, dtype=torch.float32)

    anchor_adj = C.T @ adj @ C
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C


def get_euc_anchors_alladj_as(features, adj, anchor_rate, diag, thres):

    num_nodes = features.shape[0]
    num_anchor = int(num_nodes / anchor_rate)

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)
    for i in range(num_anchor):
        cluster_points = features[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)

    corr_matrix_np = np.corrcoef(features, rowvar=True)
    adjacency_matrix_np = np.where(corr_matrix_np > thres, corr_matrix_np, 0)
    adj = torch.tensor(adjacency_matrix_np + adj, dtype=torch.float32)

    anchor_adj = C.T @ adj @ C
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C