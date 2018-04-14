import torch
import torch.nn as nn
from torch.autograd import Variable

from data_loader import DataLoader
from model import Model
from preprocessor import Preprocessor


def train(model, loss, input_seq, target):
    rnn.zero_grad()

    output = model(input_seq)
    err = loss(output, torch.nonzero(target).squeeze(1))
    err.backward()

    return rnn, criterion, output, err.data[0]


if __name__ == '__main__':
    rnn = Model(1, 128, 5)
    criterion = nn.CrossEntropyLoss()
    data = DataLoader().load()
    preprocessor = Preprocessor(data)
    X, y = preprocessor.prepare_dataset()
    for idx, x in enumerate(X):
        x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
        out = Variable(torch.LongTensor(y[idx]))
        rnn, criterion, output, loss = train(rnn, criterion, x, out)
        print(loss)

    print('All done')
