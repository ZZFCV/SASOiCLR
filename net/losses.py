import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, logits, labels=None, neg_labels=None):  # dot_results.shape=[128,32769],labels.shape=[128]
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if logits.is_cuda
                  else torch.device('cpu'))

        batch_size = logits.shape[0]  # 128

        labels = labels.contiguous().view(-1, 1)  # [128,1]
        neg_labels = neg_labels.contiguous().view(-1,1)  # [32768,1]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, neg_labels.T).float()  # [128,32768],此时mask中每一行表示:当前特征向量与队列里的负样本属于同一个类别的等于1，否则为0

        mask = torch.cat((torch.ones(mask.shape[0],1).to(device),mask),dim=1)  # [128,32769]

        # compute log_prob
        # logits = F.log_softmax(logits, dim=-1)  # [128,32769]
        # loss = - mask * logits  # [128,32769]
        # loss = torch.sum(loss,dim=-1) / mask.sum(dim=-1)  # [128]
        # loss = loss.mean()
        logits = F.softmax(logits,dim=-1)
        logits = (logits * mask).sum(dim=-1) / mask.sum(dim=-1)
        loss = -1 * torch.log(logits)
        loss = loss.mean()

        return loss
