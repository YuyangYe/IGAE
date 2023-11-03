import numpy as np
from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, weight_norm=True, link_norm=True, lambda_weight=1, lambda_link=1):

        super(Decoder, self).__init__()
        self.weight_norm = weight_norm
        self.link_norm = link_norm
        self.lambda_weight = lambda_weight
        self.lambda_link = lambda_link
        self.mlp = MLPLayer(256, 32, 1)

    def forward(self, node_embed, U_indices, C_indices):
        l_pre = self.link_pred(node_embed)
        U_embed = node_embed[U_indices]
        C_embed = node_embed[C_indices]
        w_pre = self.weight_pred(U_embed, C_embed)
        return l_pre, w_pre

    def weight_pred(self, U_embed, C_embed):
        dim = U_embed.shape[1]
        num_u = U_embed.shape[0]
        num_c = C_embed.shape[0]
        if self.weight_norm:
            inputs_zu = self.l2_normalize(U_embed[:, :(dim-2)], axis=1)
            inputs_zc = self.l2_normalize(C_embed[:, :(dim-2)], axis=1)
        else:
            inputs_zu = U_embed[:, :(dim-2)]
            inputs_zc = C_embed[:, :(dim-2)]

        dist = self.pairwise_distance2(inputs_zu, inputs_zc)
        hub = U_embed[:, (dim - 1):dim].t()
        auth = C_embed[:, (dim - 2):(dim - 1)].t()
        hub = hub.unsqueeze(0).expand(-1, num_u)
        auth = auth.unsqueeze(0).expand(1, num_c)
        # outputs = hub_i + auth_j - tf.scalar_mul(FLAGS.lamb, tf.log(dist))
        outputs = torch.log(hub) + torch.log(auth.t()) - self.weight_lamb * torch.log(dist)
        return outputs

    def link_pred(self, node_embed):
        num = node_embed.shape[0]
        dim = node_embed.shape[1]
        if self.link_norm:
            inputs_z = self.l2_normalize(node_embed[:, :(dim-2)], axis=1)
        else:
            inputs_z = node_embed[:, :(dim-2)]

        dist = self.pairwise_distance(inputs_z)
        hub = node_embed[:, (dim - 1):dim].t()
        auth = node_embed[:, (dim - 2):(dim - 1)].t()
        hub = hub.unsqueeze(0).expand(-1, num)
        auth = auth.unsqueeze(0).expand(num, -1)
        # outputs = hub_i + auth_j - tf.scalar_mul(FLAGS.lamb, tf.log(dist))
        outputs = torch.log(hub) + torch.log(auth.t()) - self.link_lamb * torch.log(dist)
        outputs = torch.sigmoid(outputs)
        return outputs

    def pairwise_distance(self, D, epsilon=0.01):
        D1 = (D * D).sum(1).unsqueeze(-1)
        D2 = torch.matmul(D, D.t())

        return D1 - 2 * D2 + D1.t() + epsilon

    def pairwise_distance2(self, A, B, epsilon=0.01):
        A1 = (A * A).sum(1).unsqueeze(-1)
        B1 = (B * B).sum(1).unsqueeze(-1)
        AB = torch.matmul(A, B.t())

        return A1 - 2 * AB + B1.t() + epsilon

    def l2_normalize(self, D, axis=1):
        row_sqr_sum = (D * D).sum(axis).unsqueeze(-1)
        return torch.div(D, row_sqr_sum)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range

        return nn.Parameter(initial)

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x