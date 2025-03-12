from collections import namedtuple
import numpy as np

Dataset = namedtuple('Dataset', ['data', 'embedding', "path"])

EdgeIndexTypes = namedtuple('EdgeIndexTypes', ['edge_index', 'weight', 'degrees', 
                                               'neg_edge_index', 'adj'])
SingleBlockData = namedtuple('SingleBlockData', ['feature', 'num_features', 'labels', 
                                                 'num_nodes', 'num_classes', 
                                                 'edge_index_types', 'anchor_feature', 'num_anchors', 
                                                 'anchor_edge_index_types', 'anchor_ass', 'anchor_labels'])
DSIData = namedtuple('DSIData', ['edge_index', 'device', 'pretrain', 'weight', 'adj', 
                                   'feature', 'degrees', 'neg_edge_index'])
DataPartition = namedtuple('DataPartition', ['features', 'labels', 'num_classes', 
                                             'num_features', 'views', 'num_nodes'])
Views = namedtuple('Views', ['userid', 'word', 'entity', 'all'])

SingleBlockData.__annotations__ = {
    'feature': np.ndarray,  # 假设feature是一个NumPy数组
    'num_features': int,
    'labels': np.ndarray,   # 假设labels也是一个NumPy数组
    'num_nodes': int,
    'num_classes': int,
    'edge_index_types': EdgeIndexTypes,  # 假设edge_index_types是一个字符串列表
    'adj': None
}