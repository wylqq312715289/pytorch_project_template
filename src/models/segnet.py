#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 4:24 PM
# @Author  : 王不二
# @Site    : 
# @File    : segnet
# @Software: PyCharm
# @describe: 
import numpy as np
import sys, random, torch

#reload(sys)
#sys.setdefaultencoding('utf8')



import pandas as pd
# import matplotlib.pyplot as plt
import time, os, json, math, re
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class SegNet(nn.Module):
    def __init__(self, conf):
        super(SegNet, self).__init__()
        input_size = conf.get("input_size",128)
        hidden_size = conf.get("hidden_size",512)
        num_layers = conf.get("num_layers",2)
        class_num = conf.get("class_num",2)
        dropout = conf.get("dropout",0.5)
        bidirectional = conf.get("bidirectional",False)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, class_num),
        )
        # self.loss_fun = nn.CrossEntropyLoss()


    def forward(self, padded_input, input_lengths, target):
        """
        Args:
            padded_input: size = N(batch) x Ti(帧数) x D(mfcc+cmvn特征维度)
                ( batch, 帧数, mfcc+cmvn的特征维度 ) # 不同的batch帧数(padding后)是不同的。
            input_lengths:n(batch, ) input在填充(padding)之前的长度。
            target: loss
        """
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        # print(output.size(),len(hidden),hidden[0].size())
        output = output[:, -1, :].float()
        out = self.fc(output)  # batch * class_num
        # print(out.size(), target.size())
        # ce_loss = self.loss_fun(out, target)
        ce_loss = F.cross_entropy(out, target)
        return ce_loss


    @staticmethod  # 将模型数据(checkpoint)序列化到文件
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # encoder
            'input_size': model.rnn.input_size,
            'hidden_size': model.rnn.hidden_size,
            'num_layer': model.rnn.num_layers,
            'dropout': model.rnn.dropout,
            'bidirectional': model.rnn.bidirectional,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package