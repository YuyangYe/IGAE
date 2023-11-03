import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import random
import pickle as pkl

#We have an adjency matrix A, and a feature matrix X for a heterogeneous graph with two types of nodes {U, C}
#This script is to get A_UU, A_UC, A_CC, X_U, X_C with index of U and C
def get_adjacency_matrix(W, idx_U, idx_C):
    W_UU = W[idx_U][:, idx_U]
    W_UC = W[idx_U][:, idx_C]
    W_CC = W[idx_C][:, idx_C]

    return W_UU, W_UC, W_CC

#return the seperated feature matrix into the orginal format
def get_adjacency_matrix_org(X_shape, X_U, X_C, idx_U, idx_C):
    X = torch.zeros(X_shape)
    X[idx_U] = X_U
    X[idx_C] = X_C
    return X

#Convert a scipy sparse matrix to dense tensor
def sparse_to_torch(matrix):
    return torch.from_numpy(matrix.toarray()).float()


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

#symmetric adjacency matrix preprocessing
def preprocess_graph_sym(adj, power=-1/2, self_loop=True):
    adj = sp.coo_matrix(adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, power).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

#Asymmetric adjacency matrix preprocessing
def preprocess_graph_asym(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # Out-degree normalization of adj
    degree_mat_inv_sqrt = sp.diags(
        np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return sparse_to_tuple(adj_normalized)


def normalize_graph(sparse_mx):
    # Find the row scalars as a Matrix_(n,1)
    sparse_mx = sparse_mx.transpose()
    rowSum = sp.csr_matrix(sparse_mx.sum(axis=1))
    rowSum.data = 1 / rowSum.data

    # Find the diagonal matrix to scale the rows
    rowSum = rowSum.transpose()
    scaling_matrix = sp.diags(rowSum.toarray()[0]).dot(sparse_mx)

    return sparse_to_tuple(scaling_matrix)

def get_unique_nodes(snapshots):
    all_nodes = set()
    for snapshot in snapshots:
        for edge in snapshot:
            all_nodes.update(edge[:2])  #(node1, node2, weight)
    return all_nodes

def read_snapshot(filename):
    with open(filename, 'r') as file:
        snapshot = []
        for line in file:
            if line.strip():  # If the line is not empty
                edge = tuple(map(float, line.split()))  # Convert the line to an edge tuple
                snapshot.append(edge)

    return snapshot

def Load_Graph(data_path, total, node_types):                 #Load all snapshots of the graph
    snapshots = [None] * total
    for i in range(total):
        snapshot = read_snapshot(data_path + str(i))
        snapshots[i] = snapshot
    unique_nodes = get_unique_nodes(snapshots)
    idx_U = [node for node in unique_nodes if node_types[node] == 'U']
    idx_C = [node for node in unique_nodes if node_types[node] == 'C']
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

    W_seq = []
    for snapshot in snapshots:
        weight_matrix = np.zeros((len(unique_nodes), len(unique_nodes)))
        for edge in snapshot:
            i, j, w = edge
            weight_matrix[node_to_index[i]][node_to_index[j]] = w
            weight_matrix[node_to_index[j]][node_to_index[i]] = w
        W_seq.append(weight_matrix)

    return idx_U, idx_C, W_seq

def feature_z_normalization(martix):
    mean = np.mean(martix, axis=0)
    std = np.std(martix, axis=0)
    return (martix - mean) / std

def Load_Feature(data_path, total, idx_U, idx_C, feature_len_U, feature_len_C):
    X_U = []
    X_C = []
    for i in range(total):
        file = open(data_path + str(i), 'rb')
        # Initialize the feature matrix
        feature_matrix_u = np.zeros((len(idx_U), feature_len_U))
        feature_matrix_c = np.zeros((len(idx_C), feature_len_C))
        node_present_u = np.zeros(len(idx_U), dtype=bool)  #mark if the node is present in the snapshot
        node_present_c = np.zeros(len(idx_C), dtype=bool)

        for line in file:
            parts = line.strip().split(',')
            node = parts[0]
            features = [float(x) for x in parts[1:]]
            if node in idx_U:
                feature_matrix_u[idx_U[node]] = features
                node_present_u[idx_U[node]] = True
            elif node in idx_C:
                feature_matrix_c[idx_C[node]] = features
                node_present_c[idx_C[node]] = True

        #average pooling for the nodes that are not present in the snapshot
        avg_features_u = np.sum(feature_matrix_u, axis=0) / np.sum(node_present_u)
        avg_features_c = np.sum(feature_matrix_c, axis=0) / np.sum(node_present_c)
        feature_matrix_u[~node_present_u] = avg_features_u
        feature_matrix_c[~node_present_c] = avg_features_c
        feature_matrix_u = feature_z_normalization(feature_matrix_u)
        feature_matrix_c = feature_z_normalization(feature_matrix_c)

        X_U.append(feature_matrix_u)
        X_C.append(feature_matrix_c)

    return X_U, X_C

def Dynamic_Graph_loader(data_path, size, lookback, total, train, indim_u, indim_c):            #Load the dynamic graph data where size is the length of snapshots and lookback is the length of historical snapshots
    node_types = pkl.load(open(data_path + 'node_types.pkl', 'rb'))
    idx_U, idx_C, W_seq = Load_Graph(data_path, total, node_types)
    X_U, X_C = Load_Feature(data_path, total, idx_U, idx_C, indim_u, indim_c)
    if train:
        DG_set = [None] * size
        for i in range(size):
            n = total/2 + i + 1
            DG_set[i] = (W_seq[n - lookback - 1: n], X_U[n - lookback - 1: n], X_C[n - lookback - 1: n])
    else:
        DG_set = [None] * size
        for i in range(size):
            n = total - i
            DG_set[i] = (W_seq[n - lookback - 1: n], X_U[n - lookback - 1: n], X_C[n - lookback - 1: n])

    return DG_set


def ismember(a, b, tol=5):                              #Check whether a row in a 2D numpy array
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)


#Negative Sampling for Link Prediction Valuation
def negative_sampling(adj_matrix):
    """Generate negative samples from an adjacency matrix based on average out-degree."""

    # Calculate average out-degree
    avg_degree = np.sum(adj_matrix) / adj_matrix.shape[0]

    # Find where the matrix is zero (i.e., no edge/link)
    no_edges = np.where(adj_matrix == 0)

    # Create a list of node pairs (tuples) that don't have a link
    no_edge_list = list(zip(no_edges[0], no_edges[1]))

    # Sample from the list based on average out-degree
    sampled_negative_edges = np.random.choice(len(no_edge_list), int(avg_degree), replace=False)

    return [no_edge_list[i] for i in sampled_negative_edges]

def link_prediction_loss(true_adj, false_adj, predicted_adj):
    # Loss for true edges
    true_loss = -np.sum(true_adj * np.log(predicted_adj + 1e-10))  # Small value added to prevent log(0)

    # Loss for false edges
    false_loss = -np.sum(false_adj * np.log(1 - predicted_adj + 1e-10))

    total_loss = true_loss + false_loss

    return total_loss


class StandScaler(object):
    def __init__(self, miu, std):
        self.miu = miu
        self.std = std

    def transform(self, data):
        return (data - self.miu) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.miu

# Convert adjacency matrix to edge list
def adj2edge(adj_matrix):
    rows, cols = np.where(adj_matrix >= 0)  # adjust the condition if needed
    values = adj_matrix[rows, cols]
    edge_list = np.column_stack([rows, cols, values])

    return edge_list


def weight2adj(weight_matrix, threshold=0.5):
    # Create an adjacency matrix with the same shape as weight_matrix
    adjacency_matrix = np.zeros(weight_matrix.shape, dtype=int)

    # Set entries to 1 where weight_matrix is above the threshold
    adjacency_matrix[weight_matrix > threshold] = 1

    return adjacency_matrix