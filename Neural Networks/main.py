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

import utils.data_reader as data_reader
from utils.optim import set_optimizer
from utils.loss_function import select_loss_function
from utils.solver import XXXSolver
import utils.util as util

ROOT = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'CodeFramework')
DATAROOT = os.path.join(ROOT, 'data', 'dataset')
TRAINROOT = os.path.join(DATAROOT, 'train.json')
DEVROOT = os.path.join(DATAROOT, 'dev.json')
TESTROOT = os.path.join(DATAROOT, 'test.json')

#################################################################################################
############################### Arguments parsing and Preparations ##############################
#################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, help='')
parser.add_argument('--dataroot', required=False, default=DATAROOT, help='path to dataset')
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')
parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--out_path', required=False, help='Only test: out_path')

# model paras
parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--bidirectional', action='store_true',
                    help='Whether to use bidirectional RNN (default is unidirectional)')

# training paras
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay of learning rate')
parser.add_argument('--l2', type=float, default=0, help='weight decay (L2 penalty)')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
parser.add_argument('--init_weight', type=float, default=0.2,
                    help='all weights will be set to [-init_weight, init_weight] during initialization')
parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--optim', default='sgd', choices=['adadelta', 'sgd', 'adam', 'rmsprop'],
                    help='choose an optimizer')


# special paras

parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')

opt = parser.parse_args()
# Some Arguments Check
if opt.test_batchSize == 0:
    opt.test_batchSize = opt.batchSize
opt.dataset = opt.dataset.lower()


##################################################################################################
################### Output path, logger, device and random seed configuration ####################
##################################################################################################

if not opt.testing:
    exp_path = util.hyperparam_string(opt)#TODO:一些和模型有关的特殊参数需要在此定义以便储存试验结果
    exp_path = os.path.join(opt.experiment, 'semantic_parser', exp_path)
else:
    exp_path = opt.out_path
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logFormatter = logging.Formatter('%(message)s')  # ('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
if opt.testing:
    fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
else:
    fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')  # override written
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
if not opt.noStdout:
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
logger.info("Parameters:" + str(opt))
logger.info("Experiment path: %s" % (exp_path))
logger.info(time.asctime(time.localtime(time.time())))

if opt.deviceId >= 0:
    import utils.gpu_selection as gpu_selection

    if opt.deviceId > 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
    elif opt.deviceId == 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
    logger.info("Valid GPU list: %s ; GPU %d (%s) is auto selected." % (valid_gpus, opt.deviceId, gpu_name))
    torch.cuda.set_device(opt.deviceId)
    opt.device = torch.device(
        "cuda")  # is equivalent to torch.device('cuda:X') where X is the result of torch.cuda.current_device()
else:
    logger.info("CPU is used.")
    opt.device = torch.device("cpu")

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
if torch.cuda.is_available():
    if opt.device.type != 'cuda':
        logger.info("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
    else:
        torch.cuda.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)

####################################################################################################
####################################### Data Reader ################################################
####################################################################################################



train_inputs, train_outputs = data_reader.DataLoader(TRAINROOT)
dev_inputs, dev_outputs = data_reader.DataLoader(DEVROOT)
test_inputs, test_outputs = data_reader.DataLoader(TESTROOT)

####################################################################################################
##################################### Model Construction ###########################################
####################################################################################################

#TODO:选择模型
train_model = ''
train_model = train_model.to(opt.device)

####################################################################################################
##################################### Model Initialization #########################################
####################################################################################################

# set loss function and optimizer
loss_function = select_loss_function('')#TODO:loss_function里可能会有其他参量，也需要加到输入列表中
optimizer = set_optimizer(train_model, opt)

####################################################################################################
###################################### Training and Decoding #######################################
####################################################################################################


solver = XXXSolver(train_model, loss_function, optimizer, exp_path, logger, opt.device)
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(train_inputs, train_outputs, dev_inputs, dev_outputs,
                            test_inputs, test_outputs, opt, max_epoch=opt.max_epoch, later=10)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    accuracy = solver.decode(test_inputs, test_outputs, os.path.join(exp_path, 'test.eval'), opt)
    logger.info('Evaluation cost: %.4fs\tAcc : %.4f' % (time.time() - start_time, accuracy))