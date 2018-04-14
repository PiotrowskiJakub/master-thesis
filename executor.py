import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_loader import DataLoader
from model import Model
from preprocessor import Preprocessor


def train(optimizer, model, loss, input_seq, target):
    optimizer.zero_grad()

    output = model(input_seq)
    err = loss(output, torch.nonzero(target).squeeze(1))
    err.backward()
    optimizer.step()

    return optimizer, model, criterion, output, err.data[0]


if __name__ == '__main__':
    model = Model(1, 128, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data = DataLoader().load()
    preprocessor = Preprocessor(data)
    X, y = preprocessor.prepare_dataset()
    for idx, x in enumerate(X):
        x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(x.size, 1, 1)
        out = Variable(torch.LongTensor(y[idx]))
        optimizer, model, criterion, output, loss = train(optimizer, model, criterion, x, out)
        print(loss)

    print('All done')
