import random
import numpy as np
import pickle, torch
from . import tools


class Feeder_single(torch.utils.data.Dataset):  # single是用在线性探针阶段，此时只要做一次数据增强
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        '''
        Semi-supervised with 10%
        '''
        # if len(self.label) != 16487:
        #     total = len(self.sample_name)
        #     sample_total = int(total * 0.1)
        #     self.sample_name = self.sample_name[:sample_total]
        #     self.label = self.label[:sample_total]
        #     self.data = self.data[:sample_total]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_triple(torch.utils.data.Dataset):  # Feeder_triple是用在预训练阶段，triple就是数据增强了三个新数据
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude  # 0.5
        self.temperal_padding_ratio = temperal_padding_ratio  # 6

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')  # self.data.shape=(40091,3,50,25,2),可以看到与st-gcn用同一个数据集，但帧数被改了，帧数只有50帧了。好事
        else:
            self.data = np.load(self.data_path)



    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])  # data_numpy.shape=(3,50,25,2)
        label = self.label[index]  # class

        # processing  # 数据增强
        data1 = self._aug(data_numpy)
        data2 = self._aug(data_numpy)
        C, T, V, M = data1.shape

        # 加mask
        mask = tools.temporal_mask(C, T, V, M, mask_ratio=0.5)
        idx = torch.from_numpy(mask).nonzero().squeeze(-1)
        data1 = data1[:, idx, :]  # 剔除不要的那部分了，data1.shape=[3,25,25,2]

        mask = 1 - mask
        idx = torch.from_numpy(mask).nonzero().squeeze(-1)
        data2 = data2[:, idx, :]  # 剔除不要的那部分了,data2.shape=[3,25,25,2]
        return [data1, data2], label

    def _aug(self, data_numpy):
        data_numpy = tools.random_rotate(data_numpy)  # x,y,z沿某个轴旋转一点点

        if self.shear_amplitude > 0:  # shear_amplitude=0.5  稍微改变一下x,y,z的位置
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:  # self.temperal_padding_ratio=6
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)  # 拼接两段被倒放的帧，然后再取50帧出来
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)  # shear_amplitude=0.5  稍微改变一下x,y,z的位置
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)  # 随机翻转，如果执行的话，就会把左手的坐标放到右手的索引上，把右手的坐标放到左手的索引上，也就是让人转了个身
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)  # x,y,z沿某个轴旋转一点点
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)  # 也是改变一下x,y,z的位置
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)  # 不好意思这步在干什么我看不懂
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)  # 直接将某一个轴的信息变为零，太狠了
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy


class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
