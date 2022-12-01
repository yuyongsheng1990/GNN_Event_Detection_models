# -*- coding: utf-8 -*-
# @Time : 2022/11/29 16:21
# @Author : yysgz
# @File : S1_construct_message_graph_functions.py
# @Project : P3_FinEvent_Model Models
# @Description :

# construct incremental message graphs
'''
This file splits the Twitter dataset into 21 message blocks (please see Section 4.3 of the paper for more details),
use the message blocks to construct heterogeneous social graphs (please see Figure 1(a) and Section 3.2 of the paper for more details)
and maps them into homogeneous message graphs (Figure 1(c)).
Note that:
# 1) We adopt the Latest Message Strategy (which is the most efficient and gives the strongest performance. See Section 4.4 of the paper for more details) here,
# as a consequence, each message graph only contains the messages of the date and all previous messages are removed from the graph;
# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)' inside construct_incremental_dataset_0922().
# 2) For test purpose, when calling construct_incremental_dataset_0922(), set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
'''

import numpy as np
import pandas as pd
import datetime
import time

import networkx as nx
from scipy import sparse

import torch

import networkx as nx
from scipy import sparse
from time import time
import dgl

from P3_FinEvent_Model.S1_gen_dataset import sparse_trans

import os
project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # # 获取上上级路径

# construct a heterogeneous graph using tweet ids, user_ids, entities and rare(sampled) words(4 modalities模态)
# if G is not None then insert new nodes to G
# 创建heterogeneous graph
def construct_graph_from_df(df, G=None):
    if G is None:
        G = nx.Graph()  # 创建无向图
    for _, row in df.iterrows():  # 返回可迭代元组(index,row)
        # 1st modality: tweet_id
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)  # 一次添加一个节点，字符串作为节点id
        G.nodes[tid]['tweet_id'] = True  # 设置节点属性；right-hand side value is irrelevant for the lookup

        # 2nd modality: user_id
        user_ids = row['user_mentions']  # list.apend(str)
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)  # 添加多个节点
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        # 3rd modality: entities
        entities = row['entities']  # 命名实体识别的实体
        #         words = ['e_' + each for each in entities]
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entities'] = True

        # 4th modality:sampled_words
        words = row['sampled_words']
        words = ['w_' + each for each in words]
        G.add_nodes_from(words)
        for each in words:
            G.nodes[each]['word'] = True

        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        edges += [(tid, each) for each in words]
        G.add_edges_from(edges)  # 同时添加多条边
    return G


# convert a heterogeneous social graph G to a homogeneous message graph following eq. 1 of the paper,
# and store the sparse binary adjacency matrix of the homogeneous message graph.
# DGL(Deep Graph Library)构建更高效的图神经网络

def dgl_hetegraph_to_homograph(G, save_path=None):
    message = ''
    print('Start converting heterogeneous networks graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networks graph to homogeneous dgl graph.\n'
    all_start = time()

    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes)
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\tGetting adjacency matrix ...')
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    A = nx.to_numpy_matrix(G)  # Returns the graph adjacency matrix as a Numpy matrix
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    tid_nodes = list(
        nx.get_node_attributes(G, 'tweet_id').keys())  # get_node_attributes return node and its attributes;获得tweet_id列表
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())  # 同理，获得user_id列表
    word_nodes = list(nx.get_node_attributes(G, 'word').keys())
    entity_nodes = list(nx.get_node_attributes(G, 'entities').keys())
    del G  # 删除original 无向图
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # 将节点转换成all_nodes中的索引index
    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    # fine细化 the index of target nodes in the list of all nodes
    indices_tid = [all_nodes.index(x) for x in tid_nodes]
    indices_userid = [all_nodes.index(x) for x in userid_nodes]
    indices_word = [all_nodes.index(x) for x in word_nodes]
    indices_entity = [all_nodes.index(x) for x in entity_nodes]
    del tid_nodes
    del userid_nodes
    del word_nodes
    del entity_nodes
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------------tweet-user-tweet------------------
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
    start = time()
    # 笛卡尔积实际上是生成一个二维坐标矩阵，其作用是从A中抽取出x和y这两类节点的一个子邻接矩阵
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)]  # np.ix_(list1, list2)生成一个笛卡尔积的映射关系；
    # return a N(indiced_tid)*N(indices_userid) matrix, representing the weight of edges between tid and userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)  # 其实就是将邻接矩阵转换成稀疏矩阵。matrix compression
    del w_tid_userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_userid_tid = s_w_tid_userid.transpose()  # 转置
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    '''
    将meta-path: tweet-user-tweet转换成tweet-tweet矩阵，这样才能得到tweet_id0的直接邻居节点tweet_id1,2,3...，不用再隔着user关系。
    '''
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid  # 根据user_id生成tweet_id homogeneous message graph
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print('sparse binary userid commuting matrix saved.')
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # ----------tweet-ent-tweet-----------------
    print('\tStart constructing tweet-ent-tweet conmuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
    start = time()
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]  # 抽取tweet_id和entity的邻接矩阵
    mins = (time() - start) / 60
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConver ting to sparse matrix ...\n'
    start = time()
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)  # 邻接矩阵转换成csr稀疏矩阵
    del w_tid_entity
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed : ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_entity_tid = s_w_tid_entity.transpose()  # 转置
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid  # 根据entity生成tweet_id homogeneous message graph
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print('Sparse binary entity commuting matrix saved.')
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # --------tweet-word-tweet------------------
    print('\tStart constructing tweet-word-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-word matrix ...')
    message += '\tStart constructing tweet-wrod-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word matrix ...'
    start = time()
    w_tid_word = A[np.ix_(indices_tid, indices_word)]
    del A
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to Sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_word = sparse.csr_matrix(w_tid_word)  # tweet_id和word稀疏矩阵
    del w_tid_word
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_word_tid = s_w_tid_word.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-word * word-tweet ...')
    message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
    start = time()
    s_m_tid_word_tid = s_w_tid_word * s_w_word_tid  # 根据word生成的tweet_id homogeneous message graph
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
        print("Sparse binary word commuting matrix saved.")
        del s_m_tid_word_tid
    del s_w_tid_word
    del s_w_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # -----------compute tweet-tweet adjacency matrix --------
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + 's_m_tid_userid_tid.npz')
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_word_tid.npz")
        print("Sparse binary word commuting matrix loaded.")

    # 合并三个user_id, entity, word生成的tweet_id homogeneous graph
    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')  # confirm the connect between tweets
    del s_m_tid_word_tid
    del s_A_tid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start) / 60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'

    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")

    # create correspoinding dgl graph
    G = dgl.DGLGraph(s_bool_A_tid_tid)  # 传入稀疏矩阵，转换成图神经网络
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges' % G.number_of_edges())
    print()
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    return all_mins, message


# To switch to the All Message Strategy or the Relevant Message Strategy, replace 'G = construct_graph_from_df(incr_df)' with 'G = construct_graph_from_df(incr_df, G)'.
# 2) For test purpose, set test=True, and the message blocks, as well as the resulted message graphs each will contain 100 messages.
# To use all the messages, set test=False, and the number of messages in the message blocks will follow Table. 4 of the paper.
def construct_offline_dataset(df, save_path, combined_features, test=True):
    # If test equals true, construct the initial graph using test_ini_size_tweets
    # and increment the graph by test_incr_size tweets each day
    test_ini_size = 500
    test_incr_size = 100

    # save data splits for training/validate/test mask generation
    data_split = []
    # save time spent for the heterogeneous -> homogeneous conversion of each graph
    all_graph_mins = []
    message = ''
    # extract distingct dates
    distinct_dates = df.date.unique()  # 所有unique的date
    print('Number of distinct dates: ', len(distinct_dates))
    message += 'Number of distinct dates: '
    message += str(len(distinct_dates))
    message += '\n'

    # split data by dates and construct graphs
    # first week -> initial graph (20254 tweets)
    print('Start constructing initial graph ...')
    message += '\nStart constructing initial graph ...\n'
    #     ini_df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 days
    #     if test:
    #         ini_df = ini_df[: test_ini_size]  # top test_ini_size dates
    G = construct_graph_from_df(df)
    path = save_path
    if path is None:
        os.mkdir(path)  # 创建目录
    graph_mins, graph_message = dgl_hetegraph_to_homograph(G,
                                                           save_path=path)  # convert a heterogeneous social graph to a homogeneous message graph
    message += graph_message
    print('Initial graph saved')
    message += 'Initial graph saved\n'
    # record the totoal number of tweets
    all_graph_mins.append(graph_mins)
    # extract and save the labels of corresponding tweets
    labels = [int(each) for each in df['event_id'].values]
    np.save(path + 'labels.npy', np.asarray(labels))  # ndarray数组，实际只创建一个指针
    print('Labels saved.')
    message += 'Labels saved.\n'
    # extract and save the features of corresponding tweets
    indices = df['index'].values.tolist()
    x = combined_features[indices, :]  # features是指combined_features: document_embeddings + time_features
    np.save(path + 'features_embeddings.npy', x)
    print('Features saved.')
    message += 'Features saved. \n\n'

    #     # subsequent days -> insert tweets day by day(skip the last day because it only contains on tweet)
    #     for i in range(7, len(distinct_dates) -1):
    #         print('Start constructing graph', str(i - 6), '...')
    #         message += '\nStart constructing graph'
    #         message += str(i-6)
    #         message += '...\n'
    #         incr_df = df.loc[df['date']==distinct_dates[i]]
    #         if test:

    return message, all_graph_mins
