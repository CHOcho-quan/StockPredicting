#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

def set_optimizer(train_model, opt):
    '''
        train_model can be nn.Module or list/tuple of nn.Modules
        opt must have fields:
            optim, lr, l2
    '''
    params = []
    if type(train_model) not in [list, tuple]:
        train_model = [ train_model ]
    for model in train_model:
        params += list(model.parameters())
    if opt.optim.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.l2, momentum=0.5)
    elif opt.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.l2) # (beta1, beta2)
    elif opt.optim.lower() == 'adadelta':
        optimizer = optim.Adadelta(params, rho=0.95, lr=1.0, weight_decay=opt.l2)
    elif opt.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=opt.lr, weight_decay=opt.l2)
    else:
        optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.l2, momentum=0.5)

    return optimizer