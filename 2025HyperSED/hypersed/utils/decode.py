import torch
import networkx as nx
from queue import Queue


class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None,
                 tree_index=None, is_leaf=False, height: int = None):
        self.index = index  # T_alpha, (44,)
        self.embeddings = embeddings  # coordinates of nodes in T_alpha, (44,3)
        self.children = []
        self.coords = coords  # node coordinates, (1,3)
        self.tree_index = tree_index  # 44
        self.is_leaf = is_leaf
        self.height = height  # 0

# for cluster evaluation.
def construct_tree(nodes_list: torch.LongTensor, manifold, coords_list: dict,  # coords_list=embeddings={2:(44,3), 1:(300,3), 0:(1,3)}
                   ass_list: dict, height, num_nodes):  # ass_list=ass_mat={2:(44,44), 1:(44,300)}; height=2; num_nodes=44
    nodes_count = num_nodes  # node cluster num count.
    que = Queue()  # 创建一个队列对象，先进先出
    root = Node(nodes_list, coords_list[height][nodes_list].cpu(),  # root.index=nodes_list, root embedding contained leaf nodes=(44,3)
                coords=coords_list[0].cpu(), tree_index=nodes_count, height=0)  # Tλ, root, cluster embedding=(1,3); tree_index=44, each leaf node as a sub-tree; tree.height for root=0
    que.put(root)

    while not que.empty():  # from top to down constructing partition tree according to height order=0,1,2.
        node = que.get()   # queue，先入先出，root node
        L_nodes = node.index  # leaf nodes, (44,)
        k = node.height + 1  # the tree.height=1 planning to construct
        if k == height:  # constructing height layer of tree for height -> height-1.
            for i in L_nodes:
                node.children.append(Node(i.reshape(-1), coords_list[height][i].cpu(), coords=coords_list[k][i].cpu(),  # 2-layer cluster corresponding leaf nodes at leaf layer
                                          tree_index=i.item(), is_leaf=True, height=k))
        else:  # middle layer k of tree for height-1 -> root.
            temp_ass = ass_list[k][L_nodes].cpu()  # 从leaf node maps to k layer of tree, (44,300).
            for j in range(temp_ass.shape[-1]):
                temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]  # leaf_node_ass maps to j-th cluster at tree k layer (,44)->nonzero as child node of root.
                if len(temp_child) > 0:  # j-th cluster of 1-layer of tree contains nodes, else the j-th cluster is null, e.g j=2,3 !!
                    nodes_count += 1
                    child_node = Node(temp_child, coords_list[height][temp_child].cpu(),  # the leaf node embeddings contributed to formulate j-th cluster at k-th layer: (44,3) extract (1,3).
                                      coords=coords_list[k][j].cpu(),  # Tj cluster embeddings at tree k-layer.
                                      tree_index=nodes_count, height=k)  # tree_index; tree.height at layer=1.
                    node.children.append(child_node)
                    que.put(child_node)
    return root


def to_networkx_tree(root: Node, manifold, height):
    edges_list = []
    nodes_list = []
    que = Queue()
    que.put(root)  # put, 入队；get 出队
    nodes_list.append(
        (
            root.tree_index,  # 44
            {'coords': root.coords.reshape(-1),
             'is_leaf': root.is_leaf,
             'children': root.index,  # root 直接对应的leaf nodes, (44,) corresponding to ass_mat for direct SE computation.
             'height': root.height}
        )
    )

    while not que.empty():
        cur_node = que.get()
        if cur_node.height == height:
            break
        for node in cur_node.children:
            nodes_list.append(
                (
                    node.tree_index,
                    {'coords': node.coords.reshape(-1),
                     'is_leaf': node.is_leaf,
                     'children': node.index,
                     'height': node.height}
                )
            )
            edges_list.append(
                (
                    cur_node.tree_index,  # parent node
                    node.tree_index,
                    {'weight': torch.sigmoid(1. - manifold.dist(cur_node.coords, node.coords)).item()}
                )
            )
            que.put(node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)
    return graph
