import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from geoopt.optim import RiemannianAdam

from hypersed.models.hyper_layers import PoincareGraphConvolution
from hypersed.utils.utils import select_activation, sparse_to_tuple, tensor_to_sparse


class GraphEncoder(nn.Module):
    def __init__(self, manifold, n_layers, in_features, n_hidden, out_dim,  # manifold=Poincare, n_layers=2, in_features=385, n_hidden=128, out_dim=3.
                 dropout, nonlin=None, use_att=False, use_bias=False):
        super(GraphEncoder, self).__init__()
        self.manifold = manifold
        self.layers = nn.ModuleList([])
        self.layers.append(PoincareGraphConvolution(self.manifold, in_features,
                                                   n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=None))  # (in_dim, hid_dim)
        for i in range(n_layers - 2):
            self.layers.append(PoincareGraphConvolution(self.manifold, n_hidden,
                                                       n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))
        self.layers.append(PoincareGraphConvolution(self.manifold, n_hidden,
                                                       out_dim, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))  # (hid_dim, out_dim)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)  # (44,385) -> (44,128) -> (44,3)
        return x


class FermiDiracDecoder(nn.Module):  # decoder: Fermi-Dirac (FD) func
    def __init__(self, 
                 args,
                 manifold):
        super(FermiDiracDecoder, self).__init__()
        
        self.args = args
        self.manifold = manifold
        self.r = self.args.r  # 2.0,
        self.t = self.args.t  # t=1.0,

    def forward(self, x):
        
        N = x.shape[0]
        dist = torch.zeros((N, N), device=x.device)  # 初始化一个 N x N 的结果张量
        
        for i in range(N):
            # 计算第 i 行的所有距离
            dist[i, :] = self.manifold.dist2(x[i].unsqueeze(0), x)  # geoopt.dist2()计算双曲空间中两个点的hyperbolic distance，点x[i]与x中所有点的距离。

        probs = torch.sigmoid((self.r - dist) / self.t)  # (44,44)
        adj_pred = torch.sigmoid(probs)  # reconstruction adj probabilities, (44,44)

        return adj_pred


class HyperGraphAutoEncoder(nn.Module):
    def __init__(self, args, device, manifold, n_layers, in_features, n_hidden, out_dim, dropout, nonlin, use_att, use_bias):
        super(HyperGraphAutoEncoder, self).__init__()

        self.args = args
        self.device = device
        self.manifold = manifold
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)

        self.encoder = GraphEncoder(self.manifold, n_layers, in_features + 1, n_hidden, out_dim + 1, 
                                    dropout, nonlin, use_att, use_bias)
        self.decoder = FermiDiracDecoder(self.args, self.manifold)
        self.optimizer = RiemannianAdam(self.parameters(), lr=self.args.lr_gae, weight_decay=args.w_decay)  # lr=0.001, w_decay=0.3, L2权重衰减


    def forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        o = torch.zeros_like(x).to(x.device)  # 后面用作使劲啊，就全是0嘛？！
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)  # geoopt.exp map xi to hyperbolic space hi.
        z = self.encoder(x, adj)  # (44,3)
        z = self.normalize(z)
        adj_pred = self.decoder(z)  # FD decoder generates a probability distribution.
        
        return adj_pred, z

    # 这段hyperbolic loss主要是基于weight binary cross-entropy (BCE)，用于poincare ball training GNN，并解决类别不平衡问题。
    # 但是pos_weight计算可能出现负数，需要纠正！！！！！！！！！
    def loss(self, x, adj):
        x = x.to(self.device)
        # adj = adj.to(self.device)

        adj_pred, z = self.forward(x, adj)  # reconstruction adj probabilities, hyper poincare gcn z.
                                                                             # pos_weight用于解决类别不平衡问题，使得GNN在sparse social network上更稳定。
        adj = tensor_to_sparse(adj, (adj.shape[0], adj.shape[1]))             # NxN表示所有可能存在的edges，包括自环；sparse adj.sum()计算实际存在的边加权和, -0.936, 错了！所以，pos_weight=不存在边 num / 存在边 num, 如果pos_weight>1，就给pos_samples更高权重，防止模型边不存在.
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.nnz) / adj.nnz  # 正样本权重 pos_weight nnz, 1.26, for weighted BCE loss，平衡positive edges(存在边) and neg_edges(不存在边)比例。
        norm = adj.shape[0] * adj.shape[0] / float(
            (adj.shape[0] * adj.shape[0] - adj.nnz) * 2)  # 计算一个norm值 0.8954，对loss_func中的正负样本进行加权平衡，确保pos_edges和neg_edge有合适比例，避免模型偏向neg_samples 或 过拟合pos_samples。
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                torch.FloatTensor(adj_label[1]),
                                                torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) != 0  # (1936,), view(-1)将mx平展成一维tensor；!=0将元素返回true
        weight_tensor = torch.ones(weight_mask.size(0))  # 创建全是1的一维tensor mx，(1936,)
        weight_tensor[weight_mask] = pos_weight  # pos_weight对存在的边进行加权
        adj_label = adj_label.to(self.device)
        weight_tensor = weight_tensor.to(self.device)
        loss = norm * F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)  # BCE loss, weight为不同样本赋予不同权重；norm归一化系数，调整loss value幅度。
        return loss, adj_pred, z  # loss=0.6839; adj_pred=(44,44); z=(44,3)
    

    def normalize(self, x):
        x = self.manifold.to_poincare(x)  # (44,2)
        x = x.to(self.device)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)  # 在x dim=-1上进行L2归一化，p表示平方；缩放范围在(0,1), 单位长度缩放因子保证x方向不变，模长受scale限制。
        x = self.manifold.from_poincare(x)  # (44,3)
        return x

