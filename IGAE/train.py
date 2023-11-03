import torch
import torch.nn as nn
import torch.nn.functional as F
from model import IGAE
from evaluate import link_eval, weight_eval
from sklearn.metrics import mean_absolute_error
from utils import *


#the script of traning the dynamic graph autoencoder

def train(model, training_dynamic_graph, device, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)
    for n in range(args.tr_size):
        W_seq, X_seq = training_dynamic_graph[n]
        A_seq = [None] * len(W_seq)
        for i in range(len(W_seq)):
            A_seq[i] = weight2adj(W_seq[i])

        A_seq = torch.Tensor(A_seq).to(device)
        W_seq = torch.Tensor(W_seq).to(device)
        X_seq = torch.Tensor(X_seq).to(device)

        w_uu_seq, w_uc_seq, w_cc_seq = W_seq[:-1]
        X_U, X_C = X_seq[:-1]
    
        W_true = W_seq[-1]
        A_true = A_seq[-1]
        A_false = negative_sampling(A_true)
    
        train_mean = np.mean(W_true.tocoo().data)
        train_std = np.std(W_true.tocoo().data)
        scaler = StandScaler(train_mean, train_std)
    
        norm = A_true.shape[0] * A_true.shape[0] / \
            float((A_true.shape[0] * A_true.shape[0] - A_true.sum()) * 2)
    
        for epoch in range(1, args.epoch+1):
            link_pred, weight_pred = model(w_uu_seq, w_uc_seq, w_cc_seq, X_U, X_C)
            optimizer.zero_grad()
            weight_loss = F.mse_loss(scaler.inverse_transform(weight_pred), W_true.coalesce().values())
            link_loss = norm * link_prediction_loss(link_pred, A_true, A_false)
    
            loss = args.beta * weight_loss + link_loss
            loss.backward()
            optimizer.step()
            print("Dynamic Graph:{}".format(n), "Epoch: {}".format(epoch), "Weight_Loss={:.4f}".format(weight_loss.item()), "Link_loss={:.4f}".format(link_loss.item()))

def test(model, test_dynamic_graph, te_size, device):
    model.eval()
    for n in range(te_size):
        W_seq, X_U_seq, X_C_seq = test_dynamic_graph[n]
        A_seq = [None] * len(W_seq)
        for i in range(len(W_seq)):
            A_seq[i] = weight2adj(W_seq[i])

        A_seq = torch.Tensor(A_seq).to(device)
        W_seq = torch.Tensor(W_seq).to(device)
        X_seq = torch.Tensor(X_seq).to(device)

        w_uu_seq, w_uc_seq, w_cc_seq = W_seq[:-1]
        X_U, X_C = X_seq[:-1]

        W_true = W_seq[-1]
        A_true = A_seq[-1]
        A_false = negative_sampling(A_true)

        train_mean = np.mean(W_true.tocoo().data)
        train_std = np.std(W_true.tocoo().data)
        scaler = StandScaler(train_mean, train_std)

        link_pred, weight_pred = model(w_uu_seq, w_uc_seq, w_cc_seq, X_U, X_C)

        with torch.no_grad():
            test_rmse, test_mae = weight_eval(weight_pred, W_true, scaler)
            print("Weight Prediction Task: ", "Test_RMSE={:.4f}".format(test_rmse), "Test_MAE={:.4f}".format(test_mae))

            test_auc, test_ap = link_eval(link_pred, adj2edge(A_true), adj2edge(A_false))
            print("Link Prediction Task: ", "Test_AUC={:.4f}".format(test_auc), "Test_AP={:.4f}".format(test_ap))