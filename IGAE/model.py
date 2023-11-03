import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

#the class of IGAE, which is composed of encoder and decoder
class IGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, n_layers):
        super(IGAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, output_dim, seq_len, n_layers)
        self.decoder = Decoder(weight_norm=True, link_norm=True, lambda_weight=1, lambda_link=1)

    def forward(self, W_uu_seq, W_uc_seq, W_cc_seq, X_U_seq, X_C_seq):
        node_embed_u, node_embed_c = self.encoder(W_uu_seq, W_uc_seq, W_cc_seq, X_U_seq, X_C_seq)
        l_pre, w_pre = self.decoder(node_embed_u, node_embed_c)
        return l_pre, w_pre

