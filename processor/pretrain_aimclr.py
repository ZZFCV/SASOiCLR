import sys
import argparse
import yaml
import math
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class AimCLR_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()  # 训练模式
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2], labels in loader:  # data1.shape=data2.shape=(128,3,25,25,2)
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            labels = labels.long().to(self.dev, non_blocking=True)  # 自监督用不上label

            if self.arg.stream == 'joint':  # 走三流时stream仍然为joint
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)  # [128,3,50,25,2]
                motion2 = torch.zeros_like(data2)
                motion3 = torch.zeros_like(data3)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]  # 仍是[128,3,50,25],但是这是沿着时间维度后一帧减前一帧，表示了骨骼的移动
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
                data3 = motion3
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),  # 所谓Bone,就是代表了edge，也就是邻接矩阵
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(data1)  # [128,3,50,25,2]
                bone2 = torch.zeros_like(data2)
                bone3 = torch.zeros_like(data3)

                for v1, v2 in Bone:  # 取出一个edge相连的两个vertex
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]  # motion是记录一个骨骼点移动的数据，那么bone就是连接两个骨骼点的边相连的移动的数据了。
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                    bone3[:, :, :, v1 - 1, :] = data3[:, :, :, v1 - 1, :] - data3[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
                data3 = bone3
            else:
                raise ValueError

            # forward
            if epoch <= 2:  # 第一轮训练时队列为空，所以第一轮先把负样本及其类别存进去，然后从第二轮开始走supervised contrastive
                outputs, _ = self.model(data1, data2, label=labels)
                if isinstance(outputs, list):  # 如果outputs是列表，就说明此时model是3s
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(outputs[0].size(0))
                    else:
                        self.model.update_ptr(outputs[0].size(0))  # 索引ptr往前进一个batch的数量，作为下次更新队列时的开始索引
                    logits = outputs[0]  # [batch,32769]
                    targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                    loss = self.CEloss(logits,targets)
                    self.iter_info['loss'] = loss.data.item()
                else:
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(outputs.size(0))
                    else:
                        self.model.update_ptr(outputs.size(0))  # 索引ptr往前进一个batch的数量，作为下次更新队列时的开始索引
                    targets = torch.zeros(outputs.shape[0], dtype=torch.long).cuda()
                    loss = self.CEloss(outputs,targets)
                    self.iter_info['loss'] = loss.data.item()
            else:
                outputs, neg_labels = self.model(data1, data2, label=labels)  # output.shape=[batch,32769],neg_labels.shape=[32768]注意虽然是对比学习，但我要传label了，因为要把label记在队列里,并且返回队列里所有负样本的类别信息。
                if isinstance(outputs, list):  # 如果outputs是列表，就说明此时model是3s
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(outputs[0].size(0))
                    else:
                        self.model.update_ptr(outputs[0].size(0))
                    logits = outputs[0]  # [batch,32769]
                    loss = self.loss(logits, labels, neg_labels)
                    self.iter_info['loss'] = loss.data.item()
                else:
                    if hasattr(self.model, 'module'):
                        self.model.module.update_ptr(outputs.size(0))
                    else:
                        self.model.update_ptr(outputs.size(0))
                    loss = self.loss(outputs, labels, neg_labels)
                    self.iter_info['loss'] = loss.data.item()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        # self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')  # 居然用GCN

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser
