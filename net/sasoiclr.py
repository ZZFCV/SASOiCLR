import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SaSOiCLR(nn.Module):  # torchlight\io里面实例化了这个类
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64, sa=True,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        base_encoder='net.st_gcn.Model', pretrain=True, feature=128, queue_size=32768, hidden_channel=16,
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)  # base_encoder=net.st_gcn.Model,应该是导入了st_gcn.py里的Model这个类，但还没实例化
        self.pretrain = pretrain  # True,声明此时是pretrain阶段

        if not self.pretrain:  # 如果不是预训练阶段就很简单，只要搭建一个网络而不是孪生网络
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size  # 32768
            self.m = momentum  # 0.999
            self.T = Temperature  # 0.07
            # base_encoder就是st_gcn,现在要实例化st_gcn
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          # 实例化backbone，注意backbone就是st-gcn，但创新了一个EADM
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          # graph_args={'layout':'ntu-rgb+d','strategy':'spatial'}
                                          edge_importance_weighting=edge_importance_weighting,  # True
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            if sa:
                dim_ca = feature_dim  # 获得mlp的输出维度
                self.projector_query_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_query_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            for param_q, param_k in zip(self.projector_query_q.parameters(), self.projector_query_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            for param_q, param_k in zip(self.projector_key_q.parameters(), self.projector_key_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            for param_q, param_k in zip(self.projector_value_q.parameters(), self.projector_value_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            self.a = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.b = nn.Parameter(torch.tensor(1).float(),requires_grad=True)

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim,
                                                      queue_size))  # queue.shape=(128,32768),也就是存储32768个输出向量，每一个向量的维度为128
            self.queue = F.normalize(self.queue, dim=0)
            label = torch.zeros((1, queue_size))  # label.shape=[1,32768] 为队列加上表示类别的维度
            self.queue = torch.cat((self.queue, label), dim=0)  # self.queue.shape=[129,32768]
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):  # keys.shape=(128,128)
        temp_labels = self.queue[-1,
                      :].clone().detach()  # 先把此时的队列里的负样本的类别取出来用于计算当前iter的supervised contrastive，然后再入队更新队列。
        batch_size = keys.shape[0]  # 128
        ptr = int(self.queue_ptr)  # 0
        gpu_index = keys.device.index  # 0
        keys = keys.T  # [batch,128] -> [128,batch]
        keys = torch.cat((keys, labels.unsqueeze(0)), dim=0)  # [129,batch]
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys  # 入队
        return temp_labels  # temp_labels.shape=[32768]

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None,label=None):  # im_q,im_k都是数据增前后的数据，shape都为[128,3,25,25,2]，线性探针阶段im_q_extreme=None
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if not self.pretrain:  # 线性探针走这里
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC  走一遍st-gcn q.shape=[128,128]
        query_q = self.projector_query_q(q)  # [batch,128]
        key_q = self.projector_key_q(q)  # [batch,128]
        value_q = self.projector_value_q(q)

        attention_map = (torch.einsum('ni,nj->nij',[query_q,key_q]))  # [batch,128,128]
        attention_map = F.softmax(attention_map,dim=2)  # [batch,128,128]
        q2 = torch.einsum('nij,nj->ni',[attention_map,value_q])  # [batch,128]

        # Normalize the feature
        q = self.a * q + self.b * q2
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC  [128,128]
            query_k = self.projector_query_k(k)  # [batch,128]
            key_k = self.projector_key_k(k)  # [batch,128]
            value_k = self.projector_value_k(k)

            attention_map = (torch.einsum('ni,nj->nij', [query_k, key_k]))  # [batch,128,128]
            attention_map = F.softmax(attention_map, dim=2)  # [batch,128,128]
            k2 = torch.einsum('nij,nj->ni', [attention_map, value_k])  # [batch,128]

            k = self.a * k + self.b * k2
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
        # positive logits: Nx1  下面这一段基本借鉴于moco
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # l_pos.shape=[128,1] 也就是得到了正样本
        # negative logits: NxK    self.queue.clone = [128,32768] 也就是有32768个向量   可是总共也才40000多个样本啊？？？
        queue_neg = self.queue[:-1, :].clone().detach()  # 把label先拿开，只用feature计算内积
        l_neg = torch.einsum('nc,ck->nk', [q, queue_neg])  # [128,32768]  也就是为每个batch都计算了32768个负样本
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # [128,32769]
        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        labels = self._dequeue_and_enqueue(k,label)

        return logits, labels