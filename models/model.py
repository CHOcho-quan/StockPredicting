#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Model(nn.Module):

    def __init__(self, input_size):
        """ Initialize model. """
        super(Model, self).__init__()
        self.model_name = "FirstEditionModel"
        self.features = nn.Sequential(
            nn.LSTM(input_size, hidden_size = 100, num_layers = 128),
            nn.BatchNorm2d(num_features=100),
            # nn.Dropout(p=0.3),
            nn.LSTM(input_size = 100, hidden_size = 50, num_layers = 50),
            nn.BatchNorm2d(num_features=50),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features = 50, out_features = 1, bias=True)
        )

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for name, params in self.named_parameters():
            if name.find('LSTM') != -1:
                nn.init.xavier_normal(params[0])
            elif name.find('Linear') != -1:
                nn.init.xavier_normal(params[0])

    def forward(self, input):
        x = self.features(x)
        return x

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

if __name__ == '__main__':
    model = Model(10)
