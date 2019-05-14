#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd.variable import Variable as V

class LSTMModel(nn.Module):

    def __init__(self, input_size, encode_size, hidden_size, num_layers, batch_size):
        """ Initialize model. """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.encode_size = encode_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "LSTMModel"
        self.encoder = nn.Sequential(nn.Linear(input_size, encode_size), nn.ReLU())
        self.rnn = nn.LSTM(encode_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.batch_size = batch_size


    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        for name, params in self.named_parameters():
            if name.find('LSTM') != -1:
                nn.init.xavier_normal(params[0])
            elif name.find('Linear') != -1:
                nn.init.xavier_normal(params[0])

    def init_hidden(self):
        self.hidden = (V(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                       V(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    def forward(self, input):
        fv = self.encoder(input)
        self.init_hidden()
        op, self.hidden = self.rnn(fv, self.hidden)
        op = self.decoder(op[-1])
        return op

