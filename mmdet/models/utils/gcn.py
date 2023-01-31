import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pdb

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        # x = torch.bmm(A, features)
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b,n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        # out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Module):
    def __init__(self, input, output):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(input, affine=False)
        self.conv1 = GraphConv(input, output, MeanAggregator)
        self.conv2 = GraphConv(input, output, MeanAggregator)

        # self.classifier = nn.Sequential(
        #     nn.Linear(64, output),
        #     nn.PReLU(output),
        #     nn.Linear(output, 2))

    def forward(self, x, A):
        # data normalization l2 -> bn
        B, N, D = x.shape  # B batch,子图个数  ，N 子图中节点个数

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        x = self.conv1(x, A)
        x = self.conv2(x, A)

        return x