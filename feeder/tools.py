import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos


transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]  # 从-0.5到0.5均匀采样三个值,作为弧度
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()  # ？？？迷惑操作
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)  # 好像就是拿(x,y,z)矩阵乘R，从而微改x,y,z的值。
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape  # 3,50,25,2  3表示x,y,z T表示帧数，V表示骨骼点，M表示一帧里的人数
    padding_len = T // temperal_padding_ratio  # padding_len = 8
    frame_start = np.random.randint(0, padding_len * 2 + 1)  # 从0到16随机选一个值
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],  # 前8帧取出来，然后全部取反，也就是从第8帧开始播放，播放到第一帧
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),  # 把最后八帧取出来，然后再全部取反，也就是从最后一帧播放，播放到倒数第八帧
                                axis=1)  # data_numpy.shape=(3,66,25,2),可以看到多了16帧
    data_numpy = data_numpy[:, frame_start:frame_start + T]  # 在新添加帧后再拿出50帧来，所以data_numpy.shape仍然为[3,50,25,2]
    return data_numpy


def random_spatial_flip(seq, p=0.5):
    if random.random() < p:  # 随机翻转，如果执行的话，就会把左手的坐标放到右手的索引上，把右手的坐标放到左手的索引上，也就是让人转了个身
        # Do the left-right transform C,T,V,M
        index = transform_order['ntu']
        trans_seq = seq[:, :, index, :]
        return trans_seq
    else:
        return seq


def random_time_flip(seq, p=0.5):
    T = seq.shape[1]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return seq[:, time_range_reverse, :, :]
    else:
        return seq


def random_rotate(seq):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],  # 旋转矩阵
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)  # 仍然是拿x,y,z三个坐标对旋转矩阵做乘法，从而实现旋转的效果
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 3, 0))
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)  # 随机选择一个主轴,0或1，也就是x轴或y轴，选中的轴旋转大角度
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)  # 旋转10°
            rotate_angle = math.radians(rotate_angle)  # 角度转成弧度，比如180°变成3.14
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (3, 0, 1, 2))

    return new_seq


def gaus_noise(data_numpy, mean= 0, std=0.01, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))  # 就是很普通地加个噪声，从而稍稍改变一点骨骼点的位置
        return temp + noise
    else:
        return data_numpy


def gaus_filter(data_numpy):
    g = GaussianBlurConv(3)
    return g(data_numpy)


class GaussianBlurConv(nn.Module):  # 所谓做高斯模糊，其实就是做平均吗，图像上的高斯模糊就是用高斯核在图像上滑动。
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels  # 3
        self.kernel = kernel  # 15
        self.min_max_sigma = sigma
        radius = int(kernel / 2)  # 7
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (1,1,1,15) -> (3,1,1,15)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(3,0,2,1) # M,C,V,T  可以看到将时间维度放在最后了，也就是说高斯模糊是对时间序列做的，这也合理，毕竟无法直接对spatial做卷积
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            x = x.permute(1,-1,-2, 0) #C,T,V,M

        return x.numpy()

class Zero_out_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis  # None


    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)  # x,y轴随机选一个轴

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new  # 直接将某个轴的坐标全部掩去了，这也太狠了吧，有点MAE的感觉
        return temp

def axis_mask(data_numpy, p=0.5):
    am = Zero_out_axis()
    if random.random() < p:
        return am(data_numpy)
    else:
        return data_numpy

def temporal_mask(C,T,V,M, mask_ratio=0.5):
    mask_count = int(np.ceil(T * mask_ratio))  # 25
    mask_idx = np.random.permutation(T)[:mask_count]
    mask = np.zeros(T, dtype=int)
    mask[mask_idx] = 1  # 为1的就是被掩盖的部分
    mask = 1 - mask  # 此时为1的是保留的那部分
    return mask

if __name__ == '__main__':
    data_seq = np.ones((3, 50, 25, 2))
    data_seq = axis_mask(data_seq)
    print(data_seq.shape)
