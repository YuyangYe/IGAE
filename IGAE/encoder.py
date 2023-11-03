import torch
import torch.nn as nn
import torch.nn.functional as F
from hetgcn_layer import HetGCN
from Transfomer_layer import Dual_SelfAttention

'''
Encoder is a stack of ST_Block, each ST_Block contains a sequence of HetGCN layer and a self-attention layer
The output of the ST_Block is the node embeddings of U and C, which are the input of the next ST_Block
The output of the last ST_Block is the node embeddings of U and C, which are the input of the decoder
'''
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, n_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.st_blocks = nn.ModuleList([ST_Block(input_dim, hidden_dim, output_dim, seq_len) for _ in range(n_layers)])
        self.mlp = nn.Linear(seq_len*output_dim, output_dim, hidden_dim=256)

    def forward(self, w_uu_seq, w_uc_seq, w_cc_seq, X_U, X_C):
        node_embed_u, node_embed_c = X_U, X_C
        for i in range(self.n_layers):
            node_embed_u, node_embed_c = self.st_blocks[i](w_uu_seq, w_uc_seq, w_cc_seq, node_embed_u, node_embed_c)
        node_embed_u, node_embed_c = self.dense(node_embed_u), self.dense(node_embed_c)
        node_embed_u, node_embed_c = node_embed_u.view(node_embed_u.shape[0], -1), node_embed_c.view(node_embed_c.shape[0], -1)
        next_node_embed_u, next_node_embed_c = self.mlp(node_embed_u), self.mlp(node_embed_c)

        return next_node_embed_u, next_node_embed_c


class ST_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(ST_Block, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.hetgcns = nn.ModuleList([HetGCN(input_dim, hidden_dim, output_dim) for _ in range(seq_len)])
        self.attention = Dual_SelfAttention(hidden_dim, seq_len)

    def forward(self, w_uu_seq, w_uc_seq, w_cc_seq, X_U, X_C):
        node_embed_u_seq = []
        node_embed_c_seq = []
        y_u_seq = []
        y_c_seq = []
        for i in range(0, self.seq_len):
            node_embed_u, node_embed_c, y_u, y_c = self.hetgcns[i](w_uu_seq[i], w_uc_seq[i], w_cc_seq[i], X_U, X_C)
            node_embed_u_seq.append(node_embed_u)
            node_embed_c_seq.append(node_embed_c)
            y_u_seq.append(y_u)
            y_c_seq.append(y_c)
        node_embed_u_seq = torch.stack(node_embed_u_seq, dim=1)
        node_embed_c_seq = torch.stack(node_embed_c_seq, dim=1)
        y_u_seq = torch.stack(y_u_seq, dim=1)
        node_embed_u = self.attention(node_embed_c_seq, y_u_seq)
        node_embed_c = self.attention(node_embed_u_seq, y_c_seq)
        return node_embed_u, node_embed_c

class MLP_layer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP_layer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

