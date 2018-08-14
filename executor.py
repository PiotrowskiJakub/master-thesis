import random
import unittest.mock

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from comet_ml import Experiment
from torch.utils.data import DataLoader

from model import Model, train
from preprocessing.padding import pad_batch
from stock_dataset import StockDataset
from utils import load_config


def main():
    config = load_config()
    if config['random_seed']:
        random.seed(config['random_seed'])
        torch.manual_seed(config['random_seed'])
    if config['use_cometml']:
        experiment = Experiment(api_key=config['comet_key'], project_name=config['project_name'])
    else:
        experiment = unittest.mock.create_autospec(Experiment)
    experiment.log_multiple_params(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StockDataset(config, device=device)
    data_loader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=False, collate_fn=collate)
    # visualize_data(dataset)

    model_config = config['model']
    model = Model(input_size=model_config['input_size'], hidden_size=model_config['hidden_size'],
                  output_size=model_config['output_size'], num_layers=model_config['num_layers'], device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    train(model=model, criterion=criterion, optimizer=optimizer, train_data_loader=data_loader,
          epochs_count=model_config['epochs_count'], experiment=experiment)


def collate(samples):
    return pad_batch(samples)


def visualize_data(dataset):
    prices = []
    volumes = []
    fig = plt.figure()
    for data in dataset:
        prices += data['input'][:, 0].tolist()
        volumes += data['input'][:, 1].tolist()
    prices_low = [p for p in prices if p < 300]
    prices_high = [p for p in prices if p >= 300]
    volumes_low = [v for v in volumes if v < 15000000]
    volumes_high = [v for v in volumes if v >= 15000000]
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.set_xlabel('price')
    ax2.set_xlabel('price')
    ax3.set_xlabel('volume')
    ax4.set_xlabel('volume')
    ax1.set_ylabel('quantity')
    ax2.set_ylabel('quantity')
    ax3.set_ylabel('quantity')
    ax4.set_ylabel('quantity')
    sns.distplot(prices_low, kde=False, ax=ax1)
    sns.distplot(prices_high, kde=False, ax=ax2)
    sns.distplot(volumes_low, kde=False, ax=ax3)
    sns.distplot(volumes_high, kde=False, ax=ax4)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
