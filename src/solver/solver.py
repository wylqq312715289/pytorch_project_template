#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 8:04 PM
# @Author  : 王不二
# @Site    : 
# @File    : solver
# @Software: PyCharm
# @describe: 
import numpy as np
import sys, random, torch

# reload(sys)
# sys.setdefaultencoding('utf8')

import pandas as pd
# import matplotlib.pyplot as plt
import time, os, json, math, re
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

import torch
from torch import cuda
from torch import nn
import torch.utils.data as data
from torch.autograd import Variable
from sklearn import metrics
from visdom import Visdom

import kaldi_io
from utils import load_json, store_json


class Mult_Classify_Metric(object):
    ## 混淆矩阵分batch计算
    def __init__(self, labels=[0, 1]):
        self.labels = labels
        self.init()


    def init(self):
        self.confusion_matrix = np.zeros((len(self.labels), len(self.labels)))


    def get_metric(self, confusion_matrix):
        """
        :param confusion_matrix:  type:numpy
        :return: type:dict
        """
        assert confusion_matrix.shape[0]==confusion_matrix.shape[1]
        acc = np.diag(confusion_matrix) / (1e-6 + confusion_matrix.sum(axis=0))
        recall = np.diag(confusion_matrix) / (1e-6 + confusion_matrix.sum(axis=1))
        return {"acc": acc, "recall": recall}

    def add_batch(self, y_pred, y_true):
        """
        :param pred: like [0,1,2,3,0,1,2,0] : # type:numpy
        :param labels: like [0,1,2,3,0,1,2,0] # type:numpy
        :return: type:numpy
        """
        sub_matrix = metrics.confusion_matrix(y_true, y_pred, self.labels)  # type:np
        self.confusion_matrix += np.array(sub_matrix)
        return self.get_metric(sub_matrix)

    def evaluate(self):
        return self.get_metric(self.confusion_matrix)



class BaseSolver(object):
    default_gpu_id = 0

    def __init__(self, conf, loader_dict, model, optimizer):
        assert loader_dict.get('tr_loader')
        assert loader_dict.get('cv_loader')
        self.tr_loader = loader_dict.get('tr_loader')
        self.cv_loader = loader_dict.get('cv_loader')
        self.model = model
        self.optimizer = optimizer
        self.epochs = conf.get("epochs",200) # type:int
        self.early_stop = conf.get("early_stop",1e10)
        self.class_num = conf.get("class_num", 2)
        self.use_gpu = conf.get("use_gpu",False) & cuda.is_available()  # 判断是否有GPU加速
        self.gpu_id = conf.get("gpu_id", self.default_gpu_id)

    def tensor2variable(self, x):
        """Convert tensor to variable."""
        if self.use_gpu: x = x.cuda()
        return Variable(x)

    def variable2numpy(self, x):
        """Convert variable to tensor."""
        if self.use_gpu: x = x.cpu()
        return x.data.numpy()

    def check_and_adjust_lr(self, epoch):
        #########################  一定轮数后调整学习率  #############################
        # [0,25,50,75]
        # if self.half_lr and val_loss >= self.prev_val_loss:
        #     if self.early_stop and self.halving:
        #         print("Already start halving learing rate, it still gets "
        #               "too small imporvement, stop training early.")
        #         break
        #     self.halving = True
        # if self.halving:
        #     optim_state = self.optimizer.state_dict()
        #     optim_state['param_groups'][0]['lr'] = \
        #         optim_state['param_groups'][0]['lr'] / 2.0
        #     self.optimizer.load_state_dict(optim_state)
        #     print('Learning rate adjusted to: {lr:.6f}'.format( lr=optim_state['param_groups'][0]['lr']))
        # self.prev_val_loss = val_loss
        pass



class Solver(BaseSolver):
    def __init__(self, conf, loader_dict, model, optimizer):
        super(Solver, self).__init__(conf, loader_dict, model, optimizer)
        # Training config

        self.metricer = Mult_Classify_Metric([ i for i in range(self.class_num)])
        # save and load model
        self.save_dir = conf.get("save_dir","")
        self.checkpoint = conf.get("checkpoint",False)
        self.continue_from_file = conf.get("continue_from_file")
        self.model_file_name = conf.get("model_file_name")

        # logging
        self.print_freq = conf.get("print_freq")
        self.logging_folder = conf.get("logging_folder")

        # visdom
        self.visdom = conf.get("visdom",False)
        self.visdom_id = conf.get("visdom_id","deubg visdom")

        # visualizing loss using visdom
        self.tr_loss = torch.zeros((self.epochs,))
        self.cv_loss = torch.zeros((self.epochs,))

        if self.visdom:
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id, ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'eval loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from_file:
            print('Loading checkpoint model %s' % self.continue_from_file)
            package = torch.load(self.continue_from_file)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_dir, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    # 每个epoch都需要可视化
    def epoch_visdom(self):
        pass

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            ########################## Train one epoch  ##########################
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            ########################### Save model each epoch #################################
            if self.checkpoint:
                file_path = os.path.join(self.save_dir, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(
                    self.model.serialize(
                        self.model, self.optimizer, epoch + 1, tr_loss=self.tr_loss, cv_loss=self.cv_loss
                    ),
                    file_path
                )
                print('Saving checkpoint model to %s' % file_path)
                logging.info('Saving checkpoint model to %s' % file_path)

            ########################### Cross validation #######################################
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Valid Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            ########################### Adjust learning rate (halving) ##########################
            # self.check_and_adjust_lr(epoch)

            ########################### Save the best model #####################################
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_dir, self.model_file_name)
                torch.save(
                    self.model.serialize(
                        self.model, self.optimizer, epoch + 1,
                        tr_loss=self.tr_loss, cv_loss=self.cv_loss
                    ),
                    file_path
                )
                print("Find better validated model, saving to %s" % file_path)

            ########################### visualizing loss using visdom  ##########################
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch), ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            ############################ loader数据格式 ###########################
            # print(padded_input.size(),input_lengths.size(),padded_target.size())
            # 例: torch.Size([16, 1451, 240]) torch.Size([16]) torch.Size([16, 29])
            # padded_input: ( batch, 帧数, mfcc+cmvn的特征维度 ) # 不同的batch帧数(padding后)是不同的。
            # input_lengths: (batch, ) input在填充(padding)之前的长度。
            # padded_target: (batch, 语句长度) 不同的batch语句长度是不同的。
            # 每个batch中的数据是对齐的，但是不同的batch的帧数和转译后的语句长度不同。
            ######################################################################
            padded_input = self.tensor2variable(padded_input).float()
            input_lengths = self.tensor2variable(input_lengths).long()
            padded_target = self.tensor2variable(padded_target).long()
            loss = self.model(padded_input, input_lengths, padded_target)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                # grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            if i % self.print_freq == 0:
                out_format = 'Epoch {0} | Iter {1} | Average loss {2:.3f} | Current batch loss {3:.6f} | {4:.1f} ms/batch'
                print(out_format.format(
                    epoch + 1, i + 1, total_loss / (i + 1), loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)

            # visualizing loss using visdom
            if self.visdom and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i + 1]
                    y_axis = vis_iters_loss[:i + 1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis, opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch, update='replace')

        return total_loss / (i + 1)
