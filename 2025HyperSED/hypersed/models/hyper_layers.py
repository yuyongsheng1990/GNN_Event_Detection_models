import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from geoopt.manifolds.stereographic.math import mobius_matvec, project, expmap0, mobius_add, logmap0
from geoopt.tensor import ManifoldParameter
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.utils import add_self_loops

from hypersed.utils.utils import gumbel_softmax, adjacency2index, index2adjacency, gumbel_sigmoid


class PoincareGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):  # in_features=385, hid_features=128, dropout=0.4
        super(PoincareGraphConvolution, self).__init__()
        self.linear = PoincareLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)  # hyperbolic Poincare linear.
        self.agg = PoincareAgg(manifold, out_features, dropout, use_att)  #  加权求和，wij

    def forward(self, x, edge_index):
        h = self.linear(x)
        h = self.agg(h, edge_index)
        return h


class PoincareLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,  # 385
                 out_features,  # 128
                 bias=True,
                 dropout=0.1,  # 0.4
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)  # 训练一个放缩参数scale，常用于Poincare缩放因子。优化时scale=exp(scale)，防止出现负数。
                                                                                                 # 归一化 normalization；自适应权重 adaptive scaling，控制模型尺度，避免梯度消失/爆炸。
    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))  # 没有体现出log operation in PoincareConv, (44, 128)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)  # (44, 127), narrow(dim, start, length)是torch的tensor split切片操作，用于在某个维度上裁剪tensor。
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1  # (44,1), trainable_time: sigmoid(o) * self.scale.exp; exp(scale)防止出现负数.
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (44,1), 时间缩放因子，控制缩放力度；所有特征平方x_narrow; 计算最后一维平方和，L2平方范数；clamp_min确保最小值，避免除数为0，防止梯度爆炸或溢出。-》用于双曲几何Poincare ball or Riemannian optimization。
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)  # (44,128), 归一化scale，是为了确保缩放后的x_narrow数值稳定
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)  # 0.088, weight权重初始化，均匀分布 (Uniform Distribution), from Xavier/Glorot初始化，不让权重太大或太小，防止梯度爆炸 or 消失。
        step = self.in_features  # 385
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)  # weight bias初始化为 0.


class PoincareAgg(nn.Module):
    """
    Poincare aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(PoincareAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features  # 128
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = PoincareLinear(manifold, in_features, in_features)
            self.query_linear = PoincareLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:  # attn aggregation
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            support_t = torch.matmul(adj, x)  # 表示aggregated node features, (44,128). 这是最基本的GCN信息传递方式，俗称一阶邻域聚合，在实际中通常会加上归一化和可学习参数。
        # 实现了hyperbolic GNN feature propagation and aggregation.
        denorm = (-self.manifold.inner(support_t, support_t, keepdim=True))  # (44,1), geoopt.inner计算hyperbolic内积; 取负号，可能是因为inner follow Lorentz model。
        denorm = denorm.abs().clamp_min(1e-8).sqrt()  # L2归一化因子denorm
        output = support_t / denorm
        return output  # (44,128)


class PoincareAssignment(nn.Module):  # DSI designs an assignment scheme that assigns the i-th node on the h-th layer to the j-th node on the (h-1)-th layer.
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(PoincareAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.proj = nn.Sequential(PoincareLinear(manifold, in_features, hidden_features,  # in_dim=3, hid_dim=64.
                                                     bias=bias, dropout=dropout, nonlin=None),
                                  # PoincareLinear(manifold, hidden_features, hidden_features,
                                  #               bias=bias, dropout=dropout, nonlin=nonlin)
                                  )
        self.assign_linear = PoincareGraphConvolution(manifold, hidden_features, num_assign + 1, use_att=use_att,  # hid_dim=64, out_dim=301
                                                     use_bias=bias, dropout=dropout, nonlin=nonlin)
        self.temperature = temperature
        self.key_linear = PoincareLinear(manifold, in_features, in_features)
        self.query_linear = PoincareLinear(manifold, in_features, in_features)
        self.bias = nn.Parameter(torch.zeros(()) + 20)
        self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(hidden_features))

    def forward(self, x, adj):
        ass = self.assign_linear(self.proj(x), adj).narrow(-1, 1, self.num_assign)  # (44,3) -> (44,64) -> (44,300)
        # 这是attn aggregation from PoincareAgg，corresponding to A*H operation。
        query = self.query_linear(x)  # (44,3)
        key = self.key_linear(x)  # (44,3)
        att_adj = 2 + 2 * self.manifold.cinner(query, key)  # lorentz inner product, 用于计算hyper points distance；2* inner + 2, 调整minkowski inner product值域。
        att_adj = att_adj / self.scale + self.bias
        att = torch.sigmoid(att_adj)  # sigmoid 避免attn value出现负值, (44,44)
        # att = torch.mul(adj.to_dense(), att)
        att = torch.mul(adj, att)  # (44,44). adj_ij * w_ij * H_ij
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})  (44,300)
        logits = torch.log_softmax(ass, dim=-1)  # probability distribution, (44,300)
        return logits


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        self.assignor = PoincareAssignment(manifold, in_features, hidden_features, num_assign, use_att=use_att, bias=bias,  # 3, 64, 300
                                          dropout=dropout, nonlin=nonlin, temperature=temperature)
        self.temperature = temperature

    def forward(self, x, adj):
        ass = self.assignor(x, adj)  # (44,3) * (44,44) = (44,300), log_softmax assignment mx for anchor nodes.
        support_t = ass.exp().t() @ x  # (300,3), 按照soft assignment weight将原始node features加权聚合新特征 z=(300,3)
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))  # 归一化因子，(300, 1)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_assigned = support_t / denorm  # partition tree mapped 后的归一化节点特征, (300,3)
        adj = ass.exp().t() @ adj @ ass.exp()  # 生成 new adj matrix, (300, 300), A^h-1 = C^T*A*C
        adj = adj - torch.eye(adj.shape[0]).to(adj.device) * adj.diag()  # 去掉对角线元素
        adj = gumbel_sigmoid(adj, tau=self.temperature)  # 二值化邻接矩阵，(300,300),使得adj更接近0/1表示；temperature控制gumbel-sigmoid平滑度。
        return x_assigned, adj, ass.exp()  # (300,3), (300,300), (44,300)