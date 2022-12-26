import torch
from torch import nn

# Spatial Drop
class DropBlock_Ske(nn.Module):
    def __init__(self, num_point=25, keep_prob=0.9):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
    # 经过下面这步的forward后，只会drop掉了所有important的骨骼点的所有帧上的特征。只剩下相对没那么重要的了
    def forward(self, input, mask):  # n,c,t,v  input.shape=[256,256,13,25], mask.shape=[256,25]
        n, c, t, v = input.size()
        mask[mask >= self.keep_prob] = 2.0
        mask[mask < self.keep_prob] = 1.0
        mask[mask == 2.0] = 0.0  # 这步赋为0.0就体现了drop的思想
        mask = mask.view(n, 1, 1, self.num_point)   # [256,25] -> [256,1,1,25]
        return input * mask * mask.numel() / mask.sum()  # mask.numel()/mask.sum()是因为input*mask会导致important feature变0，但我们只希望相对大小变，而不是整体都变小了，所以要做一个mask.numel()/mask.sum()

# Temporal Drop
class DropBlockT_1d(nn.Module):
    def __init__(self, keep_prob=0.9):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = keep_prob
    # 经过下面这一步，我们关注剩余那些没那么重要的各个骨骼点的帧上，是否有某个没那么重要的骨骼点，但他在某个帧上又比较重要，如果有的话，drop掉
    def forward(self, input, mask):  # input.shape=(256,256,13,25), mask.shape=(256,6400,13)
        n, c, t, v = input.size()
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)  # [256,256,13,25] -> [256,6400,13]
        mask[mask >= self.keep_prob] = 2.0
        mask[mask < self.keep_prob] = 1.0
        mask[mask == 2.0] = 0.0  # 这步赋为0.0就体现了drop的思想

        return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)


class Simam_Drop(nn.Module):
    def __init__(self, num_point=25, e_lambda=1e-4, keep_prob=0.9):  # keep_prob=0.9
        super(Simam_Drop, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.dropSke = DropBlock_Ske(num_point=num_point, keep_prob=keep_prob)
        self.dropT_skip = DropBlockT_1d(keep_prob=keep_prob)

    def forward(self, x):  # x.shape=(256,256,13,25)
        NM, C, T, V = x.size()  # 256,256,13,25
        num = V * T - 1  # T表示这个动作共有几帧，V表示每一帧骨骼点，所以num表示整个视频总共有多少个骨骼点
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # x_minus_mu_square.shape = x.shape,这一行就是在求论文中的(t-μ)²
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / num + self.e_lambda)) + 0.5  # y.shape=x.shape 这一步就是求出来了论文中的公式(3)
        att_map = self.activaton(y)  # sigmoid函数
        att_map_s = att_map.mean(dim=[1, 2])  # shape=(256,25)  生成空间注意力mask
        att_map_t = att_map.permute(0, 1, 3, 2).contiguous().view(NM, C * V, T)  # (256,6400,13)  # 生成时空注意力mask
        output = self.dropT_skip(self.dropSke(x, att_map_s), att_map_t)  # output.shape=(256,256,13,25),也就是输入输出维度一样，但drop掉了重要骨骼点
        return output


if __name__ == '__main__':
    NM, C, T, V = 256, 16, 13, 25
    x = torch.randn((NM, C, T, V))
    drop_sk = Simam_Drop(num_point=25, keep_prob=0.9)
    w = drop_sk(x)
    print(w.shape)
