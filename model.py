import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_num):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers_num)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        output_seq, _ = self.lstm(input_seq)
        last_output = output_seq[-1]
        class_predictions = self.linear(last_output)
        return class_predictions
