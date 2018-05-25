import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(Model, self).__init__()
        self._gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self._out_module = nn.Linear(in_features=hidden_size, out_features=output_size)
        if device.type == 'cuda':
            self._gru.cuda()
            self._out_module.cuda()

    def forward(self, input_seq):
        gru_output, hidden_state = self._gru(input_seq)
        batch_outputs = gru_output[:, -1, :]

        output = self._out_module(batch_outputs)

        return output
