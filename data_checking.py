# -*- coding: utf-8 -*-
# @Time : 2022/10/21 11:09
# @Author : yysgz
# @File : data_checking.py
# @Project : data_checking.py
# @Description :

import numpy as np
import pandas as pd

import os
project_path = os.getcwd()

# ------------------incremental_data----------------------------------
incremental_data = '/data/FinEvent_datasets/incremental'
data_features = np.load(project_path + incremental_data + '/0/features.npy')
data_labels = np.load(project_path + incremental_data + '/0/labels.npy')
data_relation_config = np.load(project_path + incremental_data + '/0/relation_config.npy', allow_pickle=True)

print(data_features.shape)
print(data_labels.shape)
print(data_relation_config)