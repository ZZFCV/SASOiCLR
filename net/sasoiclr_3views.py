import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SaSOiCLR_3s(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64, sa=True,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
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
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
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

                self.projector_query_motion_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_motion_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_motion_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_query_motion_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_motion_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_motion_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_query_bone_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_bone_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_bone_q = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_query_bone_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_key_bone_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

                self.projector_value_bone_k = nn.Sequential(nn.ReLU(),
                                                nn.Linear(dim_ca, dim_ca))

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
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
            for param_q, param_k in zip(self.projector_query_motion_q.parameters(), self.projector_query_motion_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.projector_key_motion_q.parameters(), self.projector_key_motion_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.projector_value_motion_q.parameters(), self.projector_value_motion_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.projector_query_bone_q.parameters(), self.projector_query_bone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.projector_key_bone_q.parameters(), self.projector_key_bone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.projector_value_bone_q.parameters(), self.projector_value_bone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False


            self.a = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.b = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.a_motion = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.b_motion = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.a_bone = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            self.b_bone = nn.Parameter(torch.tensor(1).float(),requires_grad=True)
            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            label = torch.zeros((1, queue_size))
            self.queue = torch.cat((self.queue, label), dim=0)  # self.queue.shape=[129,32768]
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(feature_dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(feature_dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        temp_labels = self.queue[-1,:].clone().detach()  # 先把此时的队列里的负样本的类别取出来用于计算当前iter的supervised contrastive，然后再入队更新队列。
        batch_size = keys.shape[0]  # 128
        ptr = int(self.queue_ptr)  # 0
        gpu_index = keys.device.index  # 0
        keys = keys.T  # [batch,128] -> [128,batch]
        keys = torch.cat((keys, labels.unsqueeze(0)),dim=0)  # [129,batch]
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys  # 入队
        return temp_labels  # temp_labels.shape=[32768]

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        gpu_index = keys.device.index
        self.queue_motion[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        gpu_index = keys.device.index
        self.queue_bone[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, view='all', label=None):  # im_q.shape=(batch,3,50,25,2)
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """


        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]  # 计算motion

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:  # 计算bone
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        if not self.pretrain:  # 如果是用三流做训练，三流做线性探针的话，就走下面这个view='all',也就是把三个类型的数据分别送到三个网络，然后得到的向量除以3
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone)) / 3.
            else:
                raise ValueError

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        query_q = self.projector_query_q(q)  # [batch,128]
        key_q = self.projector_key_q(q)  # [batch,128]
        value_q = self.projector_value_q(q)

        attention_map = (torch.einsum('ni,nj->nij',[query_q,key_q]))  # [batch,128,128]
        attention_map = F.softmax(attention_map,dim=2)  # [batch,128,128]
        q2 = torch.einsum('nij,nj->ni',[attention_map,value_q])  # [batch,128]

        # Normalize the feature
        q = self.a * q + self.b * q2
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        query_q_motion = self.projector_query_motion_q(q_motion)  # [batch,128]
        key_q_motion = self.projector_key_motion_q(q_motion)  # [batch,128]
        value_q_motion = self.projector_value_motion_q(q_motion)

        attention_map_motion = (torch.einsum('ni,nj->nij',[query_q_motion,key_q_motion]))  # [batch,128,128]
        attention_map_motion = F.softmax(attention_map_motion,dim=2)  # [batch,128,128]
        q2_motion = torch.einsum('nij,nj->ni',[attention_map_motion,value_q_motion])  # [batch,128]

        # Normalize the feature
        q_motion = self.a_motion * q_motion + self.b_motion * q2_motion
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        query_q_bone = self.projector_query_bone_q(q_bone)  # [batch,128]
        key_q_bone = self.projector_key_bone_q(q_bone)  # [batch,128]
        value_q_bone = self.projector_value_bone_q(q_bone)

        attention_map_bone = (torch.einsum('ni,nj->nij',[query_q_bone,key_q_bone]))  # [batch,128,128]
        attention_map_bone = F.softmax(attention_map_bone,dim=2)  # [batch,128,128]
        q2_bone = torch.einsum('nij,nj->ni',[attention_map_bone,value_q_bone])  # [batch,128]

        # Normalize the feature
        q_bone = self.a_bone * q_bone + self.b_bone * q2_bone
        q_bone = F.normalize(q_bone, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            query_k = self.projector_query_k(k)  # [batch,128]
            key_k = self.projector_key_k(k)  # [batch,128]
            value_k = self.projector_value_k(k)

            attention_map = (torch.einsum('ni,nj->nij', [query_k, key_k]))  # [batch,128,128]
            attention_map = F.softmax(attention_map, dim=2)  # [batch,128,128]
            k2 = torch.einsum('nij,nj->ni', [attention_map, value_k])  # [batch,128]

            k = self.a * k + self.b * k2
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            query_k_motion = self.projector_query_k(k_motion)  # [batch,128]
            key_k_motion = self.projector_key_k(k_motion)  # [batch,128]
            value_k_motion = self.projector_value_k(k_motion)

            attention_map_motion = (torch.einsum('ni,nj->nij', [query_k_motion, key_k_motion]))  # [batch,128,128]
            attention_map_motion = F.softmax(attention_map_motion, dim=2)  # [batch,128,128]
            k2_motion = torch.einsum('nij,nj->ni', [attention_map_motion, value_k_motion])  # [batch,128]

            k_motion = self.a_motion * k_motion + self.b_motion * k2_motion
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            query_k_bone = self.projector_query_k(k_bone)  # [batch,128]
            key_k_bone = self.projector_key_k(k_bone)  # [batch,128]
            value_k_bone = self.projector_value_k(k_bone)

            attention_map_bone = (torch.einsum('ni,nj->nij', [query_k_bone, key_k_bone]))  # [batch,128,128]
            attention_map_bone = F.softmax(attention_map_bone, dim=2)  # [batch,128,128]
            k2_bone = torch.einsum('nij,nj->ni', [attention_map_bone, value_k_bone])  # [batch,128]

            k_bone = self.a_bone * k_bone + self.b_bone * k2_bone
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        queue_neg = self.queue[:-1, :].clone().detach()  # 把label先拿开，只用feature计算内积
        l_neg = torch.einsum('nc,ck->nk', [q, queue_neg])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch,32769]
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)  # [batch,32769]
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)  # [batch,32769]

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # dequeue and enqueue
        labels = self._dequeue_and_enqueue(k, label)  # 送进去的labels.shape=[batch],输出出来的label.shape=[32768],也就是获得了当前队列里的每个特征向量的类别
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return [logits, logits_motion, logits_bone], labels