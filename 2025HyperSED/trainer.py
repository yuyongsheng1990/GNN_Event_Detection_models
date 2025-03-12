import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from torch_scatter import scatter_sum
from datetime import datetime

from hypersed.manifold.poincare import Poincare
from hypersed.manifold.lorentz import Lorentz
from hypersed.datasets.dataset import TwitterDataSet, mask_edges
from hypersed.models.hyper_gae import HyperGraphAutoEncoder
from hypersed.models.hyper_se import HyperSE
from hypersed.models.hyper_kmeans import PoincareKMeans
from hypersed.utils.namedtuples import EdgeIndexTypes, DSIData, Views, DataPartition
from hypersed.utils.utils import getOtherByedge, adjacency2index, get_anchor, getC, getNewPredict, L_ConV, get_agg_feauture
from hypersed.utils.decode import construct_tree, to_networkx_tree
from hypersed.utils.eval_utils import decoding_cluster_from_tree, cluster_metrics
from hypersed.utils.plot_utils import plot_leaves


class Trainer(nn.Module):
    def __init__(self, args, block=None):
        super(Trainer, self).__init__()

        self.args = args
        self.block = block
        self.data = TwitterDataSet(args, args.dataset_name, block)
        self.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.devices else "cpu"
        self.save_model_path = f"{args.save_model_path}/{args.algorithm}/{args.dataset_name}/best_model"
        self.manifold = Lorentz()  # Poincare()

        self.in_dim = self.data.data.feature.shape[1]  # 384
        self.num_nodes = self.data.data.num_nodes  # 893
        
        if self.args.hgae:  # hyperbolic graph auto-encoder = true
            # Part 1  Hyper Graph Encoder
            self.gae = HyperGraphAutoEncoder(self.args, self.device, self.manifold, args.num_layers_gae,   # gae layers=2;
                                            self.in_dim, args.hidden_dim_gae, args.out_dim_gae, args.dropout,  # hidden_dim=128; out_dim=2; dropout=0.4;
                                            args.nonlin, args.use_attn, args.use_bias).to(self.device)  # use_attn=False, use_bias=True
            if self.args.dsi:  # differentiable structural information
                self.in_dim = args.out_dim_gae

        if self.args.dsi:
            # Part 2  Hyper differentiable Structure Entropy
            self.hyperSE = HyperSE(args=self.args, manifold=self.manifold, n_layers=args.num_layers, device=self.device,   # SE_layers=3
                                   in_features=self.in_dim, hidden_dim_enc=args.hidden_dim, hidden_features=args.hidden_dim,   # SE_in_dim=2, SE_hid_dim=64, SE_out_dim=2
                                   num_nodes=self.num_nodes, height=args.height, temperature=args.temperature, embed_dim=args.out_dim,  # num_nodes=893, height=2, temperature=0.5,
                                   dropout=args.dropout, nonlin=args.nonlin, decay_rate=args.decay_rate,                    # dropout=0.4,
                                   max_nums=args.max_nums, use_att=args.use_attn, use_bias=args.use_bias).to(self.device)  # SE_max_nums=300.

        self.patience = self.args.patience  # 100


    def forward(self, data, mode="val"):
        # for testing, with no loss
        with torch.no_grad():
            if self.args.hgae:
                loss, feature = self.getGAEPre(data, mode)  # hyper_gae loss and feature
            else:
                feature = data.anchor_feature if self.args.pre_anchor else data.feature
            adj = data.anchor_edge_index_types.adj if self.args.pre_anchor else data.edge_index_types.adj
            if self.args.dsi:
                feature = self.hyperSE(feature, adj)  # (44,3)

        return feature.detach().cpu()


    def train_model(self, mode="train"):
        # online training
        self.train()
        time1 = datetime.now().strftime("%H:%M:%S")
        epochs = self.args.num_epochs  # 200
        data = self.data.data
        best_cluster = {'block_id': self.block, 'nmi': 0, 'ami': 0, 'ari': 0}

        # training the new block
        for epoch in tqdm(range(epochs), desc="Training Epochs"):  # tqdm用于显示training进度条，进度条前加上"Training Epochs"描述。
            if self.args.hgae:
                self.gae.optimizer.zero_grad()
            if self.args.dsi:
                self.hyperSE.optimizer_pre.zero_grad()
                if epoch > 0:
                    self.hyperSE.optimizer.zero_grad()

            # Part 1  Hyper Graph AutoEncoder
            if self.args.hgae:
                loss, feature = self.getGAEPre(data, mode)  # loss_gae=0.6839; feature=(44,3)
            else:
                feature = None

            # Part 2  Hyper Structural Entropy
            if self.args.dsi:
                input_data = self.getOtherByedge(data, feature, epoch)
                hse_loss = self.hyperSE.loss(input_data)  # 1.0782
                if self.args.hgae:
                    loss = loss + hse_loss
                else:
                    loss = hse_loss
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # 梯度裁剪，方式梯度爆炸。如果参数梯度的L2范数超过1.0，就需要进行缩放。
            loss.backward()
            if self.args.hgae:
                self.gae.optimizer.step()
            if self.args.dsi:
                self.hyperSE.optimizer_pre.step()
                if epoch > 0:
                    self.hyperSE.optimizer.step()

            best_cluster = self.evaluate(data, best_cluster, self.block)  # data.anchor_features=(44,384), anchor_adj=(44,44), edge_index=(2,855) for cluster detection
            print(f"VALID {self.block} : Epoch: {epoch}, NMI: {best_cluster['nmi']}, AMI: {best_cluster['ami']}, ARI: {best_cluster['ari']}")

            torch.cuda.empty_cache() 

            if self.patience < 0:
                print("Run out of patience")
                break

        time2 = datetime.now().strftime("%H:%M:%S")
        self.time = {'t3': time1,
                     't4': time2}
            
        if self.args.plot:
            nmi, ami, ari = self.test(data, self.block)

        self.result = {'block_id': self.block,
                  'nmi': best_cluster['nmi'],
                  'ami': best_cluster['ami'],
                  'ari': best_cluster['ari']}
        print(self.result)
        # print(f"TEST {self.block} : NUM_MSG: {data.num_nodes}, NMI: {nmi}, AMI: {ami}, ARI: {ari}")

    
    def getGAEPre(self, data, mode="train"):
        # get hgae latent representation
        if self.args.pre_anchor:
            features = data.anchor_feature.clone()  # (44, 384)
            adj_ori = data.anchor_edge_index_types.adj.clone()  # (44,44)
        else:
            features = data.feature.clone()  # (893, 384)
            adj_ori = data.edge_index_types.adj.clone()
        
        if mode == "train":
            loss, adj, feature = self.gae.loss(features, adj_ori)
        else:
            adj, feature = self.gae.forward(features, adj_ori)
            loss = None
        
        return loss, feature
    

    def getOtherByedge(self, data, gae_feature, epoch):
        if self.args.hgae:
            feature = gae_feature  # hyper space z, (44,3)
            if self.args.pre_anchor:
                edge_index_types = data.anchor_edge_index_types
            else:
                edge_index_types = data.edge_index_types
        else:
            if self.args.pre_anchor:
                feature = data.anchor_feature
                edge_index_types = data.anchor_edge_index_types
            else:
                feature = data.feature
                edge_index_types = data.edge_index_types
        return DSIData(feature=feature, adj=edge_index_types.adj, weight=edge_index_types.weight, degrees=edge_index_types.degrees,  # feat=(44,3), adj=(44,44), weight=(855,), degrees=(44,)
                        neg_edge_index=edge_index_types.neg_edge_index, edge_index=edge_index_types.edge_index, device=self.device,  # neg_edge_idx=(2,855),
                        pretrain=True if epoch == 0 else False)


    def evaluate(self, data, best_cluster, block_id):
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(data, "val")  # (44,384) -> (44,3). trainer.forward() = gae_loss + se_loss.
                                                    # Note: hyperSE.ass_mat已经转成成0-1矩阵了！！！！！
            if self.args.dsi:
                manifold = self.hyperSE.manifold.cpu()
                tree = construct_tree(torch.tensor([i for i in range(embeddings.shape[0])]).long(),
                                    manifold,
                                    self.hyperSE.embeddings, self.hyperSE.ass_mat, height=self.args.height,
                                    num_nodes=embeddings.shape[0])
                tree_graph = to_networkx_tree(tree, manifold, height=self.args.height)  # height=2 -> {Graph: 51}
                predicts = decoding_cluster_from_tree(manifold, tree_graph,
                                                    data.num_classes, embeddings.shape[0],
                                                    height=self.args.height)  # num_classes=34, num_nodes=44, height=2 -》detection clusters of anchors
            else:
                fea = self.manifold.to_poincare(embeddings)
                Z_np = fea.detach().cpu().numpy()
                M = data.num_classes

                kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
                kmeans.fit(Z_np)
                predicts = kmeans.labels_

            trues = data.labels
            nmis, amis, aris = [], [], []
            if self.args.pre_anchor:
                predicts = getNewPredict(predicts, data.anchor_ass)  # anchor_ass is from kmeans
            for step in range(self.args.n_cluster_trials):
                metrics = cluster_metrics(trues, predicts.astype(int))
                nmi, ami, ari = metrics.evaluateFromLabel()
                nmis.append(nmi)
                amis.append(ami)
                aris.append(ari)

            nmi, ami, ari = np.mean(nmis), np.mean(amis), np.mean(aris)
            nmi, ami, ari = round(nmi, 2), round(ami, 2), round(ari, 2)
            
            if nmi >= best_cluster['nmi'] and ami >= best_cluster['ami'] and ari >= best_cluster['ari']:
                self.patience = self.args.patience

                model_path = f'{self.save_model_path}/{block_id}'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(self.state_dict(), f"{model_path}/model.pt")
                
                best_cluster['nmi'] = nmi
                best_cluster['ami'] = ami
                best_cluster['ari'] = ari

            elif nmi < best_cluster['nmi'] and ami < best_cluster['ami'] and ari < best_cluster['ari']:
                self.patience -= 1

            else:
                if nmi > best_cluster['nmi']:
                    print(f'nmi: {nmi}, ami: {ami}, ari: {ari}')
                elif ami > best_cluster['ami']:
                    print(f'ami: {ami}, nmi: {nmi}, ari: {ari}')
                elif ari > best_cluster['ari']:
                    print(f'ari: {ari}, nmi: {nmi}, ami: {ami}')
                else:
                    print()

        return best_cluster

    # 离线模式学习
    def offline_train(self):
        pass

    # 开始测试
    def test(self, data, block_i):
        model_path = f'{self.save_model_path}/{block_i}'
        self.load_state_dict(torch.load(model_path + "/model.pt"))
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(data, "test")
            manifold = self.hyperSE.manifold.cpu()

            if self.args.dsi:
                tree = construct_tree(torch.tensor([i for i in range(embeddings.shape[0])]).long(),
                                    manifold,
                                    self.hyperSE.embeddings, self.hyperSE.ass_mat, height=self.args.height,
                                    num_nodes=embeddings.shape[0])
                tree_graph = to_networkx_tree(tree, manifold, height=self.args.height)
                if self.args.plot:
                    labels = data.anchor_labels if self.args.pre_anchor else data.labels
                    _, color_dict = plot_leaves(tree_graph, manifold, embeddings, labels, height=self.args.height,
                                                save_path='./' + f"{self.args.height}_{1}_true.pdf")

                predicts = decoding_cluster_from_tree(manifold, tree_graph,
                                                    data.num_classes, embeddings.shape[0],
                                                    height=self.args.height)
            
            else:
                fea = self.manifold.to_poincare(embeddings)
                Z_np = fea.detach().cpu().numpy()
                M = data.num_classes

                kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
                kmeans.fit(Z_np)
                predicts = kmeans.labels_
            

            trues = data.labels

            nmis, amis, aris = [], [], []
            if self.args.pre_anchor:
                predicts = getNewPredict(predicts, data.anchor_ass)
            for step in range(self.args.n_cluster_trials):
                metrics = cluster_metrics(trues, predicts)
                nmi, ami, ari = metrics.evaluateFromLabel()
                nmis.append(nmi)
                amis.append(ami)
                aris.append(ari)

            nmi, ami, ari = np.mean(nmis), np.mean(amis), np.mean(aris)
            metrics.get_new_predicts()
            new_pred = metrics.new_predicts
            plot_leaves(tree_graph, manifold, embeddings, new_pred, height=self.args.height,
                        save_path='./' + f"{self.args.height}_{1}_pred.pdf",
                        colors_dict=color_dict)

        return nmi, ami, ari


    # 获取训练集以及验证集数据
    def getNTBlockData(self, datas, block_id):
        # views = list(Views._fields)
        data = datas[block_id]
        features, labels, num_classes, num_features, num_nodes = data.feature, data.labels, data.num_classes, data.num_features, data.num_nodes

        datasets = {}
        # for view in views:
        edge_index_type = data.edge_index_types
        # 划分训练集、测试集以及验证集
        pos_edges, neg_edges = mask_edges(edge_index_type.edge_index, edge_index_type.neg_edge_index, 0.1, 0.2 if block_id == 0 else 0)

        train_adj, train_degrees, train_weight = getOtherByedge(pos_edges[0], num_nodes)
        val_adj, val_train_degrees, val_train_weight = getOtherByedge(pos_edges[1], num_nodes)
        train_edge = EdgeIndexTypes(adj=train_adj, degrees=train_degrees, weight=train_weight, edge_index=pos_edges[0], neg_edge_index=neg_edges[0])
        val_edge = EdgeIndexTypes(adj=val_adj, degrees=val_train_degrees, weight=val_train_weight, edge_index=pos_edges[1], neg_edge_index=neg_edges[1])
        if block_id == 0:
            test_adj, test_train_degrees, test_train_weight = getOtherByedge(pos_edges[1], num_nodes)
            test_edge = EdgeIndexTypes(adj=test_adj, degrees=test_train_degrees, weight=test_train_weight, edge_index=pos_edges[2], neg_edge_index=neg_edges[2])
            datasets = {"train": train_edge, "val":  val_edge, "test":  test_edge}
        else:
            test_edge_index_type = EdgeIndexTypes(adj=edge_index_type.adj, degrees=edge_index_type.degrees, weight=edge_index_type.weight, edge_index=edge_index_type.edge_index, neg_edge_index=edge_index_type.neg_edge_index)
            datasets = {"train": train_edge, "val":  val_edge, "test": test_edge_index_type}

        dealed_data = DataPartition(features=features, labels=labels, num_nodes=num_nodes, num_classes=num_classes, num_features=num_features, views=datasets)
        return dealed_data
