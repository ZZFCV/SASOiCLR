import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.att_drop import Simam_Drop


class Model(nn.Module):  # 这个类就是从st-gcn拿出来的
    r"""Spatial temporal graph convolutional networks."""
    # in_channels=3,hidden_channels=256,hidden_dim
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)  # 搭建图，比如邻接矩阵，然后将邻接矩阵转成spatial的分区策略(3,25,25)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # A.shape=[3,25,25]

        # build networks
        spatial_kernel_size = A.size(0)  # 3
        temporal_kernel_size = 9  # 沿着temporal那条线一次汇聚9个骨骼点信息
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # (9,3)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # nn.BatchNorm1d(75)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((  # in_channels=3,hidden_channels=16,而在st-gcn中in_channels=3,hidden_channels=64
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),  # 每个骨骼点的维度从16到32
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),  # 再从32到64
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),  # 最后从64到256
        ))
        self.fc = nn.Linear(hidden_dim, num_class)  # 全连接层，将所有的骨骼点的维度都从256维变到num_class维

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:  # st-gcn中提到的方法，也就是为partition得到的三个矩阵赋予权重
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x, drop=False):  # x.shape=(128,3,25,25,2)  当输入的数据是经过stronger augmentation时，drop=True,也就是做EADM

        # data normalization
        N, C, T, V, M = x.size()  # 128,3,25,25,2
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)  # [256,75,25],256表示共有batch=128,而每个batch两个演员，所以共256个人
        x = self.data_bn(x)  # bn
        x = x.view(N, M, V, C, T)  # [128,2,25,3,25]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)  # [256,3,25,25]

        # forward  st-gcn的forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)  # A*importance，也就是做attention

        # global pooling  输入x.shape=(256,256,7,25)  发现沿着时间维度，被压缩了,从25压缩到7
        x = F.avg_pool2d(x, x.size()[2:])  # [256,256,13,25] -> [256,256,1,1]
        x = x.view(N, M, -1).mean(dim=1)  # [128,256]  将M=2做平均，所以此时每一个batch(128个batch)都只剩一个256维向量了,与st-gcn源码仍然相似

        # prediction
        x = self.fc(x)  # [128,128]
        x = x.view(x.size(0), -1)  # [128,128]

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # kernel_size=(9,3)
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # (9,1)，作用不用我多说了，沿着temporal维度做聚合
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),  # 第一层st_gcn没有dropout,后面就有了
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A