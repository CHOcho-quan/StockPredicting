#-*- coding:utf-8 -*-
import argparse, random, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import gc

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(install_path)  # append root dir to sys.path

#import maodel

from utils.data_reader import DataLoader
from utils.optim import set_optimizer
from utils.loss_function import select_loss_function
from models.lstm_model import LSTMModel
from solvers.lstm_solver import LSTMSolver

ROOT = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'StockPredicting')
DATAROOT = os.path.join(ROOT, 'data', 'data.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=10, help='n-th future')
parser.add_argument('--seq_len', type=int, default=10, help='n-history')
parser.add_argument('--sample_gap', type=int, default=10, help='sample gap')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')


# model paras
parser.add_argument('--input_size', type=int, default=108, help='feature vector size')
parser.add_argument('--encode_size', type=int, default=100, help='encoder hidden size')
parser.add_argument('--hidden_size', type=int, default=100, help='lstm hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')

# training paras
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay of learning rate')
parser.add_argument('--l2', type=float, default=0, help='weight decay (L2 penalty)')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--loss_function', type=str, default='MSELoss', help='loss function')
parser.add_argument('--optim', default='adam', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')

parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')
parser.add_argument('--device', type=str, default='cpu', help='set device')

opt = parser.parse_args()

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)

N = opt.N
seq_len = opt.seq_len
sample_gap = opt.sample_gap
batch_size = opt.batch_size

input_size = opt.input_size
encode_size = opt.encode_size
hidden_size = opt.hidden_size
num_layers = opt.num_layers

(train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader(DATAROOT, N, seq_len, sample_gap, batch_size)

train_model = LSTMModel(input_size, encode_size, hidden_size, num_layers, batch_size).to(opt.device)

# set loss function and optimizer
loss_function = select_loss_function(opt.loss_function)
optimizer = set_optimizer(train_model, opt)

solver = LSTMSolver(train_model, loss_function, optimizer)

print("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
solver.train_and_decode(train_input, train_label, dev_input, dev_label,
                            test_input, test_label, opt, max_epoch=opt.max_epoch)
