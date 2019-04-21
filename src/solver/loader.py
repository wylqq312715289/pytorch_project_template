#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 3:57 PM
# @Author  : 王不二
# @Site    : 
# @File    : loader
# @Software: PyCharm
# @describe: 
import numpy as np
import sys, random, torch

#reload(sys)
#sys.setdefaultencoding('utf8')

RANDOM_SEED = 20190314
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

import pandas as pd
# import matplotlib.pyplot as plt
import time, os, json, math, re
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

import torch
import torch.utils.data as data

import kaldi_io
from utils import load_json, store_json


class AudioDataset(data.Dataset):
    def __init__(self, conf, data):
        # self.json_data_path = conf.get("json_data_path")
        # assert self.json_data_path
        # self.json_data = load_json(self.json_data_path) # type:dict
        # self.data = list(self.json_data.values())
        self.data = data
        self.init_data()

    def init_data(self):
        pass

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class AudioDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        # DIY self collate_fn
        # self.collate_fn = self._collate_fn


    def _collate_fn(self, batch):
        assert len(batch) == 1
        pass


