#-*- coding:utf-8 -*-

from utils.loss_function import *
from utils.optim import set_optimizer
import numpy as np
import time, os, gc

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


class XXXSolver(Solver):#类名要改

    def __init__(self, model, loss_function, optimizer, exp_path, logger, device=None):
        super(XXXSolver, self).__init__(model, loss_function, optimizer, exp_path, logger, device)
        self.best_result = {"losses": [], "iter": 0, "v_acc": 0., "t_acc": 0., "v_loss": float('inf')}

    def decode(self, data_inputs, data_outputs, output_path, opt):
        pass
        ########################### Evaluation Phase ############################

        ###################### Obtain minibatch data ######################

        ############################# Writing Result to File ###########################

        ########################### Calculate accuracy ###########################

        # return accuracy

    def train_and_decode(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                         test_inputs, test_outputs, opt, max_epoch=100, later=10):
        train_data_index = np.arange(len(train_inputs['data']))

        for i in range(max_epoch):

            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            losses, nsentences = [], len(train_data_index)

            for j in range(0, nsentences, opt.batchSize):
                ###################### Obtain minibatch data ######################

                ############################ Forward Model ############################
                self.optimizer.zero_grad()

                ############################ Loss Calculation #########################
                batch_loss = self.loss_function('')
                losses.append(batch_loss.item())

                ########################### Backward and Optimize ######################
                batch_loss.backward()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100),
                  'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.sum(losses, axis=0)
            self.best_result['losses'].append(epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss of tgt : %.5f' \
                             % (i, time.time() - start_time, epoch_loss))
            gc.collect()

            # whether evaluate later after training for some epochs
            if i < later:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            accuracy_v, loss_val = self.decode()
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tAcc : %.4f' \
                             % (i, time.time() - start_time, loss_val, accuracy_v))
            start_time = time.time()
            accuracy_t = self.decode()
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tAcc : %.4f' \
                             % (i, time.time() - start_time, accuracy_t))

            ######################## Pick best result and save #####################
            if accuracy_v > self.best_result['v_acc'] or \
                    (accuracy_v == self.best_result['v_acc'] and loss_val < self.best_result['v_loss']):
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['v_acc'], self.best_result['v_loss'] = accuracy_v, loss_val
                self.best_result['t_acc'] = accuracy_t
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Acc : %.4f;\tBest Test Acc : %.4f' \
                                 % (i, accuracy_v, accuracy_t))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (Loss: %.5f Acc : %.4f)\tBest Test (Acc : %.4f)'
                         % (self.best_result['iter'], self.best_result['v_loss'], self.best_result['v_acc'],
                            self.best_result['t_acc']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))





