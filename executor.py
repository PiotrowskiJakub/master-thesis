from random import sample

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.batch_size = self.config['batch_size']
        self.model = Model(self.config['input_size'], self.config['hidden_size'], self.config['output_size'],
                           self.config['layers_num'], self.batch_size)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        self.X_train, self.X_test, self.y_train, self.y_test = Executor._read_data(0)

    def _init_comet_experiment(self, config):
        self.experiment = Experiment(api_key=config['comet_key'])
        self.experiment.log_multiple_params(config['model'])

    @staticmethod
    def _read_data(test_size=0.2):
        data = DataLoader().load()
        preprocessor = Preprocessor(data)
        X, y = preprocessor.prepare_dataset()
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    def train(self):
        with self.experiment.train():
            for epoch in range(self.config['epochs']):
                self.experiment.log_current_epoch(epoch)
                i = 0
                while i < len(self.X_train) - self.batch_size:
                    x = np.concatenate(self.X_train[i:i + self.batch_size])
                    y = np.concatenate(self.y_train[i:i + self.batch_size])
                    x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(self.X_train[i].size,
                                                                                   self.batch_size, 1)
                    y = Variable(torch.from_numpy(y).type(torch.FloatTensor)).view(self.batch_size, -1)
                    output, loss = self._run_step(x, y)
                    i += self.batch_size
                    self.experiment.log_metric('loss', loss)
                    # similarity = F.cosine_similarity(output, y)
                    # self.experiment.log_metric('similarity', similarity.data[0])
                # self.test()

    def _run_step(self, input_seq, target):
        self.optimizer.zero_grad()

        output = self.model(input_seq)
        err = self.loss(output, target)
        err.backward()
        self.optimizer.step()

        return output, err.data[0]

    def test(self):
        similarity_sum = 0.0
        with self.experiment.test():
            for i, x in enumerate(self.X_test):
                x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
                y = Variable(torch.from_numpy(self.y_test[i]).type(torch.FloatTensor)).view(1, -1)
                output = self.model(x)
                similarity = F.cosine_similarity(output, y)
                similarity_sum += similarity.data[0]

            similarity_sum /= len(self.y_test)
            print('Accuracy of the network: %d %%' % similarity)
            self.experiment.log_metric('accuracy', similarity)

    def visualize_data(self):
        fig, ax = plt.subplots()
        data = sample(self.X_train, 50)
        for x in data:
            sns.distplot(x, ax=ax, kde=False)
        plt.show()

    def visualize_labels(self):
        # data = [y.index(1) for y in self.y_train]
        # data2 = [y.index(1) for y in self.y_test]
        # data = data + data2
        data = self.y_train + self.y_test
        sns.distplot(data, kde=False)
        plt.show()


if __name__ == '__main__':
    executor = Executor()
    executor.train()
    # executor.visualize_labels()
    print('Finished Training')
