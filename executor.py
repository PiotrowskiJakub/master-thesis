from random import sample

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from data_loader import DataLoader
from model import Model
from preprocessor import Preprocessor
from utils import load_config


class Executor:

    def __init__(self):
        config = load_config()
        self._init_comet_experiment(config)
        self.config = config['model']
        self.model = Model(self.config['input_size'], self.config['hidden_size'], self.config['output_size'],
                           self.config['layers_num'])
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
        self.X_train, self.X_test, self.y_train, self.y_test = Executor._read_data()

    def _init_comet_experiment(self, config):
        self.experiment = Experiment(api_key=config['comet_key'])
        self.experiment.log_multiple_params(config['model'])

    @staticmethod
    def _read_data(test_size=0.1):
        data = DataLoader().load()
        preprocessor = Preprocessor(data)
        X, y = preprocessor.prepare_dataset()
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    def train(self):
        with self.experiment.train():
            for epoch in range(self.config['epochs']):
                correct = 0
                for i, x in enumerate(self.X_train):
                    x = np.array(x)
                    x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(len(x), 1, self.config['input_size'])
                    out = Variable(torch.LongTensor(self.y_train[i]))
                    output, loss = self._run_step(x, out)
                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == torch.nonzero(out.data).squeeze(1)).sum()
                    print('Training loss: %.3f. Accuracy: %.3f' % (loss, correct / (i + 1)))
                    self.experiment.log_metric('loss', loss)
                    self.experiment.log_metric('accuracy', correct / (i + 1))
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
        with self.experiment.test():
            for i, x in enumerate(self.X_test):
                x = np.array(x)
                x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(len(x), 1, self.config['input_size'])
                target = Variable(torch.LongTensor(self.y_test[i]))
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == torch.nonzero(target.data).squeeze(1)).sum()

            accuracy = (100 * correct / len(self.y_test))
            print('Accuracy of the network: %.3f %%' % accuracy)
            self.experiment.log_metric('accuracy', accuracy)

    def visualize_data(self):
        fig, ax = plt.subplots()
        data = sample(self.y_train, 50)
        for y in data:
            sns.distplot(y, ax=ax, kde=False)
        plt.show()


if __name__ == '__main__':
    executor = Executor()
    executor.train()
    # executor.visualize_data()
    print('Finished Training')
