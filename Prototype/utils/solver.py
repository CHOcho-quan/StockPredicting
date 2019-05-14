#-*- coding:utf-8 -*-

class Solver():

    def __init__(self, model, loss_function, optimizer, exp_path='', logger='', device='cpu'):
        super(Solver, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device

    def train_and_validate(self, *args, **kargs):
        raise NotImplementedError

    def test(self, *args, **kargs):
        raise NotImplementedError


