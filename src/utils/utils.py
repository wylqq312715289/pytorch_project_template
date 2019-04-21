#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 7:58 PM
# @Author  : 王不二
# @Site    : 
# @File    : utils
# @Software: PyCharm
# @describe:

import numpy as np
import sys, random, torch
from torch import cuda
import re

# reload(sys)
# sys.setdefaultencoding('utf8')

from datetime import datetime
from datetime import timedelta
import pandas as pd
import os, logging, copy, math, time, h5py, json, shutil
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file

# from svmutil import svm_read_problem

def init_logging(filename, level=logging.DEBUG):
    logging.basicConfig(
        filename=filename, filemode="w", datefmt="%d-%M-%Y %H:%M:%S",
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", level=level)


def init_randseed(RANDOM_SEED=20190421):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if cuda.is_available():
        cuda.manual_seed(RANDOM_SEED)


# 一般矩阵归一化
def my_normalization(data_ary, axis=0):
    # axis = 0 按列归一化; 1时按行归一化
    if axis == 1:
        data_ary = np.matrix(data_ary).T
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
        ans = min_max_scaler.fit_transform(ans)
        ans = np.matrix(ans).T
        ans = np.array(ans)
    else:
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
        ans = min_max_scaler.fit_transform(ans)
    return ans


def one_hot(data_ary, one_hot_len):
    # data_ary = array([1,2,3,5,6,7,9])
    # one_hot_len: one_hot最长列
    max_num = np.max(data_ary);
    ans = np.zeros((len(data_ary), one_hot_len), dtype=np.float64)
    for i in range(len(data_ary)):
        ans[i, int(data_ary[i])] = 1.0
    return ans


def re_onehot(data_ary):
    # data_ary = array([[0,0,0,1.0],[1.0,0,0,0],...])
    ans = np.zeros((len(data_ary),), dtype=np.float64)
    for i in range(len(data_ary)):
        for j in range(len(data_ary[i, :])):
            if data_ary[i, j] == 1.0:
                ans[i] = 1.0 * j;
                break;
    return ans


# 将数据写入h5文件
def write2H5(h5DumpFile, data):
    # if not os.path.exists(h5DumpFile): os.makedir(h5DumpFile)
    with h5py.File(h5DumpFile, "w") as f:
        f.create_dataset("data", data=data, dtype=np.float64)


# 将数据从h5文件导出
def readH5(h5DumpFile):
    feat = [];
    with h5py.File(h5DumpFile, "r") as f:
        feat.append(f['data'][:])
    feat = np.concatenate(feat, 1)
    print('readH5 Feature.shape=', feat.shape)
    return feat.astype(np.float64)


# 将dict数据保存到json
def store_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


# 将json文件中的数据读取到dict
def load_json(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
        return data


# 将一个文件copy到指定目录
def moveFileto(sourceDir, targetDir): shutil.copy(sourceDir, targetDir)


# 删除目录下的所有文件
def removeDir(dirPath):
    if not os.path.isdir(dirPath): return
    files = os.listdir(dirPath)
    try:
        for file in files:
            filePath = os.path.join(dirPath, file)
            if os.path.isfile(filePath):
                os.remove(filePath)
            elif os.path.isdir(filePath):
                removeDir(filePath)
        os.rmdir(dirPath)
    except Exception():
        print("removeDir exception")


# 尽量等长分割一维数组(最后一组可能会更短)
def list_cut(x, batch_num):
    ans = [[] for i in range(batch_num)]
    for i, item in enumerate(x, 0):
        ans[i % batch_num].append(item)
    return ans


def kaldi_scp_passer(file_path):
    pass

#
# def load_conf(conf_file):
#     dict_data = dict()
#     with open(conf_file,"r") as f:
#         lines = f.readlines()
#         for line in lines:
#             if "=" not in line: continue
#             key = re.sub(r"([\s\S]*)=([\s\S]*)", r"\1", line).strip()
#             value = re.sub(r"([\s\S]*)=([\s\S]*)(#[\s\S]*)", r"\2", line).strip()
#             dict_data[key] = value
#     return dict_data
