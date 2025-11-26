import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, dropout=0.1):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, node_n=seq_len)
        self.gc2 = GraphConvolution(hidden_dim, output_dim, node_n=seq_len)
        self.dropout = dropout
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = F.relu(self.gc1(x))  # (batch_size, seq_len, hidden_dim)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)  # (batch_size, seq_len, output_dim)
        return x
    
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py#L9
    """

    def __init__(self, in_features, out_features, bias=True, node_n=13):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

