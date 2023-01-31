from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class GraphAttentionLayer111111(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        # print(np.shape(inp))
        # print(np.shape(adj))
        # print(np.shape(self.W))
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        # print(np.shape(h))
        N = h.size()[1]  # N 图的节点数
        # print()
        # print(np.shape(torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1)))
       
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,N,N,2 * self.out_features)
        # print(np.shape(a_input))
        # [B, N, N, 2*out_features]
        # print(np.shape(self.a))
        # print(np.shape(torch.matmul(a_input, self.a)))
        # print(torch.matmul(a_input, self.a))
        # print(torch.matmul(a_input, self.a).squeeze(3))
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # print(np.shape(e))
        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        # print(zero_vec)

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # print(attention)
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        print("1111")
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(np.shape(Wh))
        e = self._prepare_attentional_mechanism_input(Wh)
        # print(np.shape(e))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        # pdb.set_trace()
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1,2)
        return self.leakyrelu(e)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        # self.attentions = [GraphAttentionLayer(490, 490, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(1)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(12544, 12544, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # print(np.shape(x))
        # print(np.shape(adj))
        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        # print(x.shape)#因为有2个头所以concat起来 特征维度有490*2
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        # x = F.elu(x)
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        # return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
        return x


# x = torch.randn(2, 5, 32, 12)
# adj = torch.tensor([
#     [0, 1, 0, 1, 1],
#     [1, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 1, 0]
# ], dtype=torch.float32)
# print(np.shape(x))
# # w = torch.randn(32 * 12, 32 * 12)
# # x = x.permute(0, 2, 1, 3)
# x = torch.reshape(x, (2, 5, -1))

# net = GAT(32 * 12, 32 * 12, 64, 0.1, 1, 2)
# out = net(x, adj)

# print(out.shape)
