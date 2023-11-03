import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


def link_eval(link_A, test_edges_pos, test_edges_false):                #link prediction
    link_pred = []
    link_true = np.hstack([np.ones(len(test_edges_pos)),
                           np.zeros(len(test_edges_false))])
    for idx, idy in test_edges_pos:
        link_pred.append(link_A[idx][idy].detach().item())

    for idx, idy in test_edges_false:
        link_pred.append(link_A[idx][idy].detach().item())
    link_pred = np.array(link_pred)

    roc_score = roc_auc_score(link_true, link_pred)
    ap_score = average_precision_score(link_true, link_pred)

    return roc_score, ap_score


def weight_eval(Pred_A_UC, A_UC, scaler):               #Edge Weight Prediction
    Pred_A_UC = scaler.inverse_transform(Pred_A_UC.detach().numpy())

    rmse = np.sqrt(mean_squared_error(Pred_A_UC, A_UC))
    mae = mean_absolute_error(Pred_A_UC, A_UC)
    return rmse, mae
