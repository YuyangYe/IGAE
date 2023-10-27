import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='IGAE')

parser.add_argument('-lr', '--learn_rate', default=0.001, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
parser.add_argument('-dp', '--data_path', default='LinkedIn', type=str)
parser.add_argument('-ep', '--epoch', default=300, type=int)
parser.add_argument('-l', '--lookback', default=4, type=int)
parser.add_argument('-indim', '--input_dim', default=768, type=int)
parser.add_argument('-hidim', '--hidden_dim', default=128, type=int)
parser.add_argument('-outdim', '--output_dim', default=128, type=int)
parser.add_argument('-lw', '--lamb_w', default=0.1, type=float)
parser.add_argument('-ll', '--lamb_e', default=5.0, type=float)

args = parser.parse_args()
