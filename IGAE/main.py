import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import IGAE
from utils import Dynamic_Graph_loader
from train import train, test
import numpy as np
import random

parser = argparse.ArgumentParser(description='IGAE')

parser.add_argument('-lr', '--learn_rate', default=0.001, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
parser.add_argument('-dp', '--data_path', default='LinkedIn', type=str)
parser.add_argument('-ep', '--epoch', default=300, type=int)
parser.add_argument('-l', '--lookback', default=4, type=int)
parser.add_argument('-tr_size', '--train_size', default=5, type=int)
parser.add_argument('-te_size', '--test_size', default=1, type=int)
parser.add_argument('-b', '--beta', default=0.2, type=float, help='weight loss weight')
parser.add_argument('-indim_u', '--input_dim', default=698, type=int)
parser.add_argument('-indim_c', '--input_dim_c', default=738, type=int)
parser.add_argument('-hidim', '--hidden_dim', default=128, type=int)
parser.add_argument('-nlayer', '--n_layers', default=2, type=int)
parser.add_argument('-outdim', '--output_dim', default=128, type=int)
parser.add_argument('-lw', '--lamb_w', default=0.1, type=float)
parser.add_argument('-ll', '--lamb_e', default=5.0, type=float)
parser.add_argument('-de', '--device', default='cpu', type=str)
parser.add_argument('-ns', '--number_snapshots', default=12, type=int, help='total number of snapshots to be used for training')

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print(args)
    device = torch.device(args.de)
    training_dynamic_graph = Dynamic_Graph_loader(args.data_path, size=args.tr_size, lookback=args.lookback, total_num=args.ns, train=True, indim_u=args.indim_u, indim_c=args.indim_c).to(device)
    test_dynamic_graph = Dynamic_Graph_loader(args.data_path, size=args.te_size, lookback=args.lookback, total_num=args.ns, train=False, indim_u=args.indim_u, indim_c=args.indim_c).to(device)
    igae = IGAE(args.input_dim, args.hidden_dim, args.output_dim, args.lookback, args.n_layers).to(device)
    setup_seed(2023)
    train(igae, training_dynamic_graph, device, args)
    test(igae, test_dynamic_graph, args.te_size, device)