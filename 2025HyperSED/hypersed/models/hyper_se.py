import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project, dist0, dist
from torch_scatter import scatter_sum
from torch_geometric.utils import negative_sampling
from geoopt.optim import RiemannianAdam

from hypersed.models.hyper_layers import PoincareGraphConvolution, PoincareLinear
from hypersed.manifold.poincare import Poincare
from hypersed.models.hyper_gae import GraphEncoder
from hypersed.models.hyper_layers import LSENetLayer, PoincareGraphConvolution
from hypersed.utils.utils import gumbel_softmax
from hypersed.utils.decode import construct_tree
from hypersed.utils.namedtuples import DSIData
from hypersed.utils.utils import select_activation

MIN_NORM = 1e-15
EPS = 1e-6


class LSENet(nn.Module):  # Lorentz structural entropy neural network for assignment scheme
    def __init__(self, args, manifold, n_layers, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3,
                 temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
        super(LSENet, self).__init__()
        if max_nums is not None:
            assert len(max_nums) == height - 1, "length of max_nums must equal height-1."
        self.args = args
        self.manifold = manifold
        self.nonlin = select_activation(nonlin) if nonlin is not None else None
        self.temperature = temperature  # 0.05
        self.num_nodes = num_nodes
        self.height = height  # 2
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = GraphEncoder(self.manifold, n_layers, in_features + 1, hidden_dim_enc, embed_dim + 1,
                                        # n_layers=3, in_dim=3, hid_dim=64, out_dim=3.
                                        use_att=use_att, use_bias=use_bias, dropout=dropout, nonlin=self.nonlin)

        self.layers = nn.ModuleList([])
        if max_nums is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            max_nums = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        for i in range(height - 1):  # height=2
            self.layers.append(
                LSENetLayer(self.manifold, embed_dim + 1, hidden_features, max_nums[i],  # in_dim=3, hid_dim=64
                            bias=use_bias, use_att=use_att, dropout=dropout,
                            nonlin=self.nonlin, temperature=self.temperature))

    def forward(self, z, edge_index):  # z=(44,3), edge_index=(44,44)

        if not self.args.hgae:
            o = torch.zeros_like(z).to(z.device)
            z = torch.cat([o[:, 0:1], z], dim=1)
            z = self.manifold.expmap0(z)
        z = self.embed_layer(z, edge_index)  # GraphEncoder: (44,3) + (44,44) -> (44,64)->(44,3)
        z = self.normalize(z)  # Poincare normalization

        self.tree_node_coords = {self.height: z}  # height=2: (44,3)
        self.assignments = {}

        edge = edge_index.clone()  # (44,44)
        ass = None  # assignment scheme.
        for i, layer in enumerate(
                self.layers):  # 1 layer LSENet for anchor assignment scheme; initial anchor number M=N/ε≈44.
            z, edge, ass = layer(z, edge)  # (300,3); (300,300), (44,300)
            self.tree_node_coords[self.height - i - 1] = z
            self.assignments[self.height - i] = ass

        self.tree_node_coords[0] = self.manifold.Frechet_mean(z)  # (1,3), 这里用的是算术平均 as paper description.
        self.assignments[1] = torch.ones(ass.shape[-1], 1).to(z.device)

        return self.tree_node_coords, self.assignments  # (44,3) -> (300,3) -> (1,3); assignments: (44,300) -> (300,1)

    def normalize(self, x):
        x = self.manifold.to_poincare(x).to(self.scale.device)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x

class HyperSE(nn.Module):
    # def __init__(self, args, manifold, n_layers, device, in_features, hidden_features, num_nodes, height=3, temperature=0.2,
    #              embed_dim=2, out_dim = 2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
    def __init__(self, args, manifold, n_layers, device, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
        
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes  # 893
        self.height = height  # 2
        self.tau = temperature  # 0.05
        self.manifold = manifold
        self.device = device
        self.encoder = LSENet(args, self.manifold, n_layers, in_features, hidden_dim_enc, hidden_features,
                              num_nodes, height, temperature, embed_dim, dropout,
                              nonlin, decay_rate, max_nums, use_att, use_bias)
        self.optimizer_pre = RiemannianAdam(self.parameters(), lr=args.lr_pre, weight_decay=args.w_decay)
        self.optimizer = RiemannianAdam(self.parameters(), lr=args.lr, weight_decay=args.w_decay)

    def forward(self, features, adj):
        features = features.to(self.device)
        adj = adj.to(self.device)
        adj = adj.to_dense()
        embeddings, clu_mat = self.encoder(features, adj)  # embeddings = self.tree_node_coords = {2:(44,3), 1:(300,3), 0:(1,3)}; assignments = clu_mat = {2:(44,300), 1:(300,1)}
        self.embeddings = {}
        self.num_nodes = features.shape[0]  # 44
        for height, x in embeddings.items():
            self.embeddings[height] = x.detach()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(self.device)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
        for k, v in ass_mat.items():  # 2:(44,44); 1:(44,300)
            idx = v.max(1)[1]  # 在dim=1方向上，取出最大值对应的index，并赋值给idx=(44,)
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.  # (44,44)
            ass_mat[k] = t  # 将probability ass_mat 转换成 0、1 ass_mat
        self.ass_mat = ass_mat
        return self.embeddings[self.height]  # (44,3)

    def loss(self, input_data: DSIData):  # 计算SE loss, and return hyper ball space的原点距离 during pretrain.

        device = input_data.device
        weight = input_data.weight.to(self.device)  # (855)
        adj = input_data.adj.to(self.device)  # (44,44)
        degrees = input_data.degrees.to(self.device)  # (44,)
        features = input_data.feature.to(self.device)  # (44,3)
        edge_index = input_data.edge_index.to(self.device)  # (2,855)
        neg_edge_index = input_data.neg_edge_index.to(self.device)
        pretrain = input_data.pretrain  # true
        self.num_nodes = features.shape[0]  # 44

        embeddings, clu_mat = self.encoder(features, adj.to_dense())  # LSENet for assignment scheme, (44,3) -> (300,3) -> (1,3); assignments: (44,300) -> (300,1)

        se_loss = 0
        vol_G = weight.sum()  # 30322
        ass_mat = {self.height: torch.eye(self.num_nodes).to(self.device)}  # {2: (44,44)}, 单位矩阵代表原始图节点。
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}  # 存储每层的volume，即node degree sum, {2: (44,), 0: (1,)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]  # clu_mat denotes assignment matrix, (44,300)
            vol_dict[k] = torch.einsum('ij, i->j', ass_mat[k], degrees)  # 计算new layer degree volumes, (300), the contribution weight from original degrees.
        """
        1. we already have hierarchical node cluster assignments dict, clu_mat={2:(44,300), 1:(300,1)}, why still need to create ass_mat={2:(44,44), 1:(44,300)} ?
            - 因为为了计算SE，我们需要从最底层directly map to 任意层，而不仅仅是相邻层之间的map。
            - 因此，我们构建ass_mat，它的作用是从最底层直接map to 某个 layer。
        2.  在计算SE过程中，为什么需要凭空创建一个单位矩阵 (44,44) for self.height=2 ?
            - 单位矩阵 (44,44)表示最底层node不变，即每个anchor (node cluster)最初属于自己，即每个node cluster在最低height=2 layer映射到自己的概率是100%。
            - 从这个意义上说，对于height=1这个特殊情况：ass_mat[2]= I * clu_mat[2] = clu_mat[2], (44,300).
            - 这样SE计算从最底层原始图结构开始，可以逐步向上聚合。
        3. why ass_mat doesn't have 0-th layer such as 0:(44,1) ?
            - 因为vol_dict[0]直接存储了root node volume，不需要ass_mat[0]去做重复计算，因为root node layer只剩一个点了，vol(G)是所有edge weight sum。
        """
        if pretrain:  # true
            return self.manifold.dist0(embeddings[0])  # dist0 computes the distance form the origin (zero point at hyper space) to a given point，衡量node embeddings'偏移程度。
        # designs a 层次化图结构 (hierarchical graph structure)的结构熵 (structural entropy)
        for k in range(1, self.height + 1):  # (1,2), 不包含leaf nodes.
            vol_parent = torch.einsum('ij, j->i', clu_mat[k], vol_dict[k - 1])  # (N_k, )  # (300,1), vol(k-1)=parent degree vol; ij->j,i, 表示从parent degree分解到当前层每个cluster的vol。
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (N_k, ) # if vol_dict[k]≈vol_parent, 说明层次聚类稳定，熵贡献低；vol_parent<<vol_dict[k]，结构剧变，熵贡献大。
            ass_i = ass_mat[k][edge_index[0]]   # (E, N_k), (44,300) * 855 edge_index
            ass_j = ass_mat[k][edge_index[1]]
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # (N_k, ), 计算两个nodes在同一cluster中的匹配度，weight是边权重-》第k层中cluster内部edge weight sum。
            delta_vol = vol_dict[k] - weight_sum    # (N_k, )
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss  # 归一化结构熵
        return se_loss + self.manifold.dist0(embeddings[0])