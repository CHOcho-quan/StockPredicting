#-*- coding:utf-8 -*-
import torch
import torch.nn as nn

def select_loss_function(loss_function):#根据选择的loss_function选择返回值
    if loss_function == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_function == '':
        return ''
    else:
        return ''