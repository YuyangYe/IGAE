import torch
import torch.nn as nn
import torch.nn.functional as F
from hetgcn_layer import HetGCN
from transfomer_layer import SelfAttention

'''
Encoder is a stack of ST_Block, each ST_Block contains a sequence of HetGCN layer and a self-attention layer
The output of the ST_Block is the node embeddings of U and C, which are the input of the next ST_Block
The output of the last ST_Block is the node embeddings of U and C, which are the input of the decoder
'''
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.st_blocks = nn.ModuleList([ST_Block(input_dim, hidden_dim, output_dim, seq_len) for _ in range(seq_len)])
        self.mlp = nn.Linear(seq_len*output_dim, output_dim, hidden_dim=256)

    def forward(self, adj_uu_seq, adj_uc_seq, adj_cc, X_U, X_C):
        node_embed_u, node_embed_c = X_U, X_C
        for i in range(self.seq_len):
            node_embed_u, node_embed_c = self.st_blocks[i](adj_uu_seq, adj_uc_seq, adj_cc, node_embed_u, node_embed_c)
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
        self.hetgcns = nn.ModuleList([HetGCN(input_dim, hidden_dim, output_dim, seq_len) for _ in range(seq_len)])
        self.attention = SelfAttention(hidden_dim, seq_len)

    def forward(self, adj_uu_seq, adj_uc_seq, adj_cc, X_U, X_C):
        node_embed_u_seq = []
        node_embed_c_seq = []
        for i in range(1, self.seq_len):
            node_embed_u, node_embed_c, _, _ = self.hetgcns[i](adj_uu_seq[i], adj_uc_seq[i], adj_cc, X_U, X_C)
            node_embed_u_seq.append(node_embed_u)
            node_embed_c_seq.append(node_embed_c)
        node_embed_u_seq = torch.stack(node_embed_u_seq, dim=1)
        node_embed_c_seq = torch.stack(node_embed_c_seq, dim=1)
        node_embed_u = self.attention(node_embed_c_seq)
        node_embed_c = self.attention(node_embed_u_seq)
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

