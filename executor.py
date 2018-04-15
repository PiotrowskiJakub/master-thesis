import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from data_loader import DataLoader
from model import Model
from preprocessor import Preprocessor
from utils import load_config


class Executor:

    def __init__(self):
        self.config = load_config()['model']
        self.model = Model(self.config['input_size'], self.config['hidden_size'], self.config['output_size'])
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.X_train, self.X_test, self.y_train, self.y_test = Executor._read_data()
        self.running_loss = 0.0

    @staticmethod
    def _read_data():
        data = DataLoader().load()
        preprocessor = Preprocessor(data)
        X, y = preprocessor.prepare_dataset()
        return train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

    def train(self):
        for epoch in range(self.config['epochs']):
            for i, x in enumerate(self.X_train):
                x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
                out = Variable(torch.LongTensor(self.y_train[i]))
                output, loss = self._run_step(x, out)
                self.running_loss += loss
                if i % 100 == 99:
                    print('[%d, %5d/%5d] loss: %.3f' %
                          (epoch + 1, i + 1, len(self.X_train), self.running_loss / 100))
                    self.running_loss = 0.0
            self.test()

    def _run_step(self, input_seq, target):
        self.optimizer.zero_grad()

        output = self.model(input_seq)
        err = self.loss(output, torch.nonzero(target).squeeze(1))
        err.backward()
        self.optimizer.step()

        return output, err.data[0]

    def test(self):
        correct = 0
        for i, x in enumerate(self.X_test):
            x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
            target = Variable(torch.LongTensor(self.y_test[i]))
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.nonzero(target.data).squeeze(1)).sum()

        print('Accuracy of the network: %d %%' % (100 * correct / len(self.y_test)))


if __name__ == '__main__':
    executor = Executor()
    executor.train()
    print('Finished Training')
