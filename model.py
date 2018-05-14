import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_num, batch_size):
        super(Model, self).__init__()
        self.layers_num = layers_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(input_size, hidden_size, 1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, layers_num)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    # def forward(self, input_seq):
    #     output_seq, _ = self.lstm(input_seq)
    #     last_output = output_seq[-1]
    #     return self.linear(last_output)

    def forward(self, input_seq):
        outputs = []

        for i, input_b in enumerate(input_seq.chunk(input_seq.size(1), dim=1)):
            h_t = Variable(torch.zeros(1, 1, self.hidden_size), requires_grad=False)
            c_t = Variable(torch.zeros(1, 1, self.hidden_size), requires_grad=False)
            h_t2 = Variable(torch.zeros(self.layers_num, 1, self.hidden_size), requires_grad=False)
            c_t2 = Variable(torch.zeros(self.layers_num, 1, self.hidden_size), requires_grad=False)
            for i, input_t in enumerate(input_b.chunk(input_b.size(0), dim=0)):
                input_t, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))
                input_t, (h_t2, c_t2) = self.lstm2(input_t, (h_t2, c_t2))
                input_t = F.relu(self.linear(input_t))
                output = self.linear_out(input_t)
            outputs += [output]

        outputs = torch.cat(outputs).view(-1, 1)
        return outputs
