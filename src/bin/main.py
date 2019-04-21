#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 8:44 PM
# @Author  : 王不二
# @Site    :
# @File    : main
# @Software: PyCharm
# @describe:

import argparse
import torch
from torch.optim import SGD, Adam
import logging
import sys, random, os
import yaml
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

from loader import AudioDataLoader, AudioDataset
from segnet import SegNet
from solver import Solver
from utils import init_logging, init_randseed


def get_args():
    parser = argparse.ArgumentParser("Speech Segmentation")
    parser.add_argument('--conf_file', default="../../conf/template.conf", type=str)
    args = parser.parse_args()
    return args

def get_optim(model, conf):
    # TODO: 从conf中解析优化器
    if conf.get("optimizer", "sgd") == 'sgd':
        lr = conf.get("lr", 0.0002)
        momt = conf.get("sgd_momentum", 0)
        # w_dcy: 权重衰减(如L2惩罚)(默认: 0)
        w_dcy = conf.get("opt_l2_penalty", 0)
        optimizier = SGD(model.parameters(), lr=lr,  momentum=momt, weight_decay=w_dcy)
    elif conf.get("optimizer", "sgd") == 'adam':
        lr = conf.get("lr", 0.0002)
        # w_dcy: 权重衰减(如L2惩罚)(默认: 0)
        w_dcy = conf.get("opt_l2_penalty", 0)
        optimizier = Adam(model.parameters(), lr=lr, weight_decay=w_dcy)
    else:
        print("Not support optimizer")
        logging.ERROR("Not support optimizer")
        return None
    return optimizier



data = list(zip(
    np.random.randn(100, 50, 13),  # batch * seqlen * feat_dim
    np.array([50 for i in range(100)]),  # seqlen
    np.random.randint(0, 2, size=(100,)))  # target
)


def main(args, conf):
    ##  DIY dataset
    tr_dataset = AudioDataset(conf, data)
    cv_dataset = AudioDataset(conf, data)

    ## DIY dataloader
    batch_size=conf.get("batch_size", 33)
    nun_workers = conf.get("num_workers", 5)
    tr_loader = AudioDataLoader( tr_dataset, batch_size=batch_size,num_workers=nun_workers )
    cv_loader = AudioDataLoader( cv_dataset, batch_size=batch_size,num_workers=nun_workers)
    loader_dict = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    ## DIY model
    model = SegNet(conf)
    print(model)
    model.cuda()
    optimizier = get_optim(model, conf)

    # solver
    solver = Solver(conf, loader_dict, model, optimizier)
    solver.train()


if __name__ == '__main__':
    args = get_args()
    conf = yaml.load(open(args.conf_file), Loader=yaml.FullLoader)
    init_randseed(conf.get("random_seed",20190421))
    init_logging(filename=os.path.join(conf.get("logging_dir"), "test.log"))
    print("conf: ", conf)
    main(args, conf)
