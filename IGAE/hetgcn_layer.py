import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class HetGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HetGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gcn_same = GCN_S(input_dim, hidden_dim)
        self.gcn_cross = GCN_C(input_dim, hidden_dim)

        self.W = torch.nn.Linear(input_dim + output_dim, output_dim, bias=False)
        self.a = torch.nn.Parameter(torch.FloatTensor(output_dim, 1))

    def forward(self, X_U, X_C, adj_uu, adj_uc, adj_cc):
        node_embed_u = self.gcn_same(adj_uu, X_U)
        node_embed_c1 = self.gcn_same(adj_cc, X_C)
        node_embed_c2 = self.gcn_cross(adj_uc, X_U, X_C)

        c_concat_1 = torch.cat((node_embed_c1, X_C), dim=-1)
        a_input_1 = self.W(c_concat_1)
        a_tau_1 = torch.matmul(F.relu(a_input_1), self.a)

        c_concat_2 = torch.cat((node_embed_c2, X_C), dim=-1)
        a_input_2 = self.W(c_concat_2)
        a_tau_2 = torch.matmul(F.relu(a_input_2), self.a)

        a_tau = torch.cat((a_tau_1, a_tau_2), dim=1)
        e_tau = F.softmax(a_tau, dim=1)
        Y_c = e_tau[:, 0].unsqueeze(1) * node_embed_c1
        Y_u = e_tau[:, 1].unsqueeze(1) * node_embed_c2

        node_embed_c = Y_c + Y_u

        return node_embed_u, node_embed_c, Y_c, Y_u

#GCN for aggregating the node information from the neighbors of the same type
class GCN_S(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN_S, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.glorot_init(in_features, out_features)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range

        return nn.Parameter(initial)

    def forward(self, weighted_adjacency_matrix, feature_matrix):
        # Calculate the normalization factor
        degree_matrix_inv_sqrt = torch.diag_embed(torch.pow(1 + torch.sum(weighted_adjacency_matrix, dim=1), -0.5))

        # Applying the normalization factor to the edge weights
        norm_adjacency_matrix = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, weighted_adjacency_matrix), degree_matrix_inv_sqrt)

        # Perform the GCN operation
        support = torch.matmul(feature_matrix, self.weight)
        output = torch.matmul(norm_adjacency_matrix, support)

        return output

#GCN for aggregating the node information from the neighbors of the different type
class GCN_C(torch.nn.Module):
    def __init__(self, in_features_u, in_features_c, out_features):
        super(GCN_C, self).__init__()
        self.in_features_u = in_features_u
        self.in_features_c = in_features_c
        self.out_features = out_features

        self.projection_u = self.glorot_init(in_features_u, out_features)
        self.projection_c = self.glorot_init(in_features_c, out_features)
        self.weight = self.glorot_init(out_features, out_features)


    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range

        return nn.Parameter(initial)

    def forward(self, adjacency_matrix, feature_matrix_u, feature_matrix_c):
        # Project the features of U and C into the same latent space
        projected_u = torch.matmul(feature_matrix_u, self.projection_u)
        projected_c = torch.matmul(feature_matrix_c, self.projection_c)

        # Calculate the normalization factor
        degree_matrix_u_inv_sqrt = torch.diag_embed(torch.pow(torch.sum(adjacency_matrix, dim=1), -0.5))
        degree_matrix_c_inv_sqrt = torch.diag_embed(torch.pow(torch.sum(adjacency_matrix, dim=0), -0.5))

        # Applying the normalization factor to the adjacency matrix
        norm_adjacency_matrix = torch.matmul(torch.matmul(degree_matrix_u_inv_sqrt, adjacency_matrix),degree_matrix_c_inv_sqrt)

        # Perform the GCN operation
        support = torch.matmul(projected_u, self.weight)
        output_c = torch.matmul(norm_adjacency_matrix, support)

        return output_c

