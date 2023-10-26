import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#We have an adjency matrix A, and a feature matrix X for a heterogeneous graph with two types of nodes {U, C}
#This script is to get A_UU, A_UC, A_CC, X_U, X_C with index of U and C
def get_adjacency_matrix(A, X, idx_U, idx_C):
    A_UU = A[idx_U][:, idx_U]
    A_UC = A[idx_U][:, idx_C]
    A_CC = A[idx_C][:, idx_C]
    X_U = X[idx_U]
    X_C = X[idx_C]
    return A_UU, A_UC, A_CC, X_U, X_C

#return the seperated feature matrix into the orginal format
def get_adjacency_matrix_org(X_shape, X_U, X_C, idx_U, idx_C):
    X = torch.zeros(X_shape)
    X[idx_U] = X_U
    X[idx_C] = X_C
    return X