import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_loader import DataLoader
from model import Model
from preprocessor import Preprocessor
from utils import load_config


class Executor:

    def __init__(self):
        config = load_config()['model']
        self.model = Model(config['input_size'], config['hidden_size'], config['output_size'])
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.X, self.y = Executor._read_data()

    @staticmethod
    def _read_data():
        data = DataLoader().load()
        preprocessor = Preprocessor(data)
        return preprocessor.prepare_dataset()

    def train(self):
        for idx, x in enumerate(self.X):
            x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
            out = Variable(torch.LongTensor(self.y[idx]))
            output, loss = self._run_step(x, out)
            print(loss)

    def _run_step(self, input_seq, target):
        self.optimizer.zero_grad()

        output = self.model(input_seq)
        err = self.loss(output, torch.nonzero(target).squeeze(1))
        err.backward()
        self.optimizer.step()

        return output, err.data[0]


if __name__ == '__main__':
    executor = Executor()
    executor.train()
    print('All done')
