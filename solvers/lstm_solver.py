#-*- coding:utf-8 -*-
from utils.loss_function import *
from utils.optim import set_optimizer
import numpy as np
import time, os, gc, csv

from utils.solver import Solver


class LSTMSolver(Solver):

    def __init__(self, model, loss_function, optimizer, exp_path='', logger='', device=None):
        super(LSTMSolver, self).__init__(model, loss_function, optimizer, exp_path, logger, device)
        self.best_result = {"losses": [], "iter": 0, "v_acc": 0., "t_acc": 0., "v_loss": float('inf')}
        self.batch_size = self.model.batch_size

    def write_csv(self, results, file_name):
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'midPrice'])
            writer.writerows(results)

    def decode(self, data_inputs, data_outputs, opt):
        # output_path = opt.out_path
        # model = opt.model().eval()
        # if opt.load_model_path:
        #     model.load(opt.load_model_path)
        # model.to(opt.device)
        self.model.batch_size = 1
        results = []
        for idx, data in enumerate(data_inputs):
            output = self.model(data)
            label = data_outputs[idx]
            loss = self.loss_function(output.view_as(label), label)
            results.append(loss.detach().numpy())

        mse_loss = np.mean(results)

        #self.write_csv(results, output_path)

        return mse_loss

    def train_and_decode(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                         test_inputs, test_outputs, opt, max_epoch=100, later=0):

        for i in range(max_epoch):

            ########################### Training Phase ############################
            start_time = time.time()
            self.model.batch_size = self.batch_size
            losses = []

            for idx, data in enumerate(train_inputs):
                label = train_outputs[idx]
                self.optimizer.zero_grad()
                output = self.model(data)
                batch_loss = self.loss_function(output.view_as(label), label)
                losses.append(batch_loss.detach().numpy())
                batch_loss.backward()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100),
                  'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.mean(losses)
            self.best_result['losses'].append(epoch_loss)

            print('Train:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, epoch_loss))

            # whether evaluate later after training for some epochs
            if i < later:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_mse_loss = self.decode(valid_inputs, valid_outputs, opt)
            print('Evaluation:\tEpoch : {}\tTime : {}s\tMSELoss : {}'.format(i, time.time() - start_time, dev_mse_loss))
            start_time = time.time()
            test_mse_loss = self.decode(test_inputs, test_outputs, opt)
            print('Test:\tEpoch : {}\tTime : {}s\tMSELoss : {}\n'.format(i, time.time() - start_time, test_mse_loss))


