import random
import unittest.mock

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

    model_config = config['model']
    experiment.log_multiple_params(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StockDataset(config, device=device)
    data_loader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=True)

    model = Model(input_size=model_config['input_size'], hidden_size=model_config['hidden_size'],
                  output_size=model_config['output_size'], num_layers=model_config['num_layers'], device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    train(model=model, criterion=criterion, optimizer=optimizer, train_data_loader=data_loader,
          epochs_count=model_config['epochs_count'], experiment=experiment)


def collate(samples):
    return pad_batch(samples)


if __name__ == '__main__':
    main()
