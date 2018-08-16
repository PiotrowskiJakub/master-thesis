import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import quandl
import seaborn as sns
import torch
from torch.utils.data import Dataset

from utils import load_pickle, save_pickle, load_config


class StockDataset(Dataset):

    def __init__(self, config, device, mode, normalize=True):
        data_config = config['data']
        raw_data = StockDataset._load_raw_data(data_config, mode)
        self._device = device
        self.input_label_pairs = StockDataset._prepare_raw_data(raw_data, data_config=data_config, normalize=normalize)

    @staticmethod
    def _load_raw_data(data_config, mode):
        data_path = os.path.join('data', mode, data_config['data_filename'])
        data = load_pickle(data_path)
        if data is None:
            quandl.ApiConfig.api_key = data_config['quandl_key']
            tickers_path = os.path.join('data', mode, data_config['tickers_filename'])
            tickers = ['WIKI/' + ticker for ticker in load_tickers(tickers_path)]
            start_date = datetime.date(1990, 1, 1)
            end_date = datetime.datetime.now().date()
            data = quandl.get(tickers, start_date=start_date, end_date=end_date)
            save_pickle(data, data_path)

        return data

    @staticmethod
    def _prepare_raw_data(raw_data, data_config, normalize):
        input_label_pairs = []

        past_days = data_config['past_days']
        forecast_days = data_config['forecast_days']

        close = select_column(raw_data, 'Adj. Close').values
        volume = select_column(raw_data, 'Adj. Volume').values

        for company_num in range(close.shape[1]):
            i = 0
            while i < close.shape[0] - past_days:
                close_input = close[i:i + past_days, company_num]
                volume_input = volume[i:i + past_days, company_num]
                prices = close[i + past_days:i + past_days + forecast_days, company_num]
                i = i + past_days
                if np.isnan(np.concatenate((close_input, volume_input, prices))).any():
                    continue
                max_price = np.max(prices)
                last_mean = np.mean(close_input[-forecast_days:])
                change_percentage = (max_price - last_mean) / last_mean
                y = StockDataset._generate_labels(change_percentage, data_config=data_config)
                # derivatives = np.diff(close_input)
                # derivatives = np.append(derivatives, derivatives[-1])
                if normalize:
                    close_input_normalized = (close_input - np.mean(close_input)) / np.std(close_input)
                    volume_input_normalized = (volume_input - np.mean(volume_input)) / np.std(volume_input)
                    inputs = list(zip(close_input_normalized, volume_input_normalized))
                else:
                    inputs = list(zip(close_input, volume_input))
                input_label_pairs.append((inputs, y))

        return input_label_pairs

    @staticmethod
    def _generate_labels(change_percentage, data_config):
        """Creates a vector that shows price changes.
        [1 0 0 0 0] - the price fell by more than 5%
        [0 1 0 0 0] - the price fell by more than 3% but less than 5%
        [0 0 1 0 0] - the price has not changed significantly
        [0 0 0 1 0] - the price rose by more than 3% but less than 5%
        [0 0 0 0 1] - the price rose by more than 5%
        """
        change_threshold_boundaries = data_config['change_threshold_boundaries']
        vector_length = len(change_threshold_boundaries) * 2 + 1
        change_vector = ([0] * vector_length)

        for idx, threshold in enumerate(change_threshold_boundaries):
            if change_percentage < -threshold:
                change_vector[idx] = 1
                return change_vector
            elif change_percentage > threshold:
                change_vector[vector_length - 1 - idx] = 1
                return change_vector

        change_vector[len(change_threshold_boundaries)] = 1
        return change_vector

    def get_subset(self, step):
        return self.input_label_pairs[::step]

    def __len__(self):
        return len(self.input_label_pairs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_label_pairs[idx][0], dtype=torch.float, device=self._device)
        label_tensor = torch.tensor(self.input_label_pairs[idx][1], dtype=torch.float, device=self._device)
        sample = {
            'input': input_tensor,
            'label': label_tensor
        }

        return sample


def load_tickers(tickers_path):
    with open(tickers_path, 'r') as f:
        tickers = list(csv.reader(f))[0]

    return tickers


def select_column(data, col_name):
    return data.select(lambda col: col.endswith(col_name), axis=1)


def plot_labels_distribution(config, mode):
    dataset = StockDataset(config, None, mode=mode, normalize=False)
    label_classes = np.argmax([p[1] for p in dataset.input_label_pairs], axis=1) + 1
    ax = sns.distplot(label_classes, kde=False, label='Dupa')
    ax.set_xlabel('Category')
    ax.set_ylabel('Quantity')
    plt.show()


def plot_input_distribution_unnormalized(config, mode):
    dataset = StockDataset(config, 'cpu', mode=mode, normalize=False)
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
    ax1.set_xlabel('Price')
    ax2.set_xlabel('Price')
    ax3.set_xlabel('Volume')
    ax4.set_xlabel('Volume')
    ax1.set_ylabel('Quantity')
    ax2.set_ylabel('Quantity')
    ax3.set_ylabel('Quantity')
    ax4.set_ylabel('Quantity')
    sns.distplot(prices_low, kde=False, ax=ax1)
    sns.distplot(prices_high, kde=False, ax=ax2)
    sns.distplot(volumes_low, kde=False, ax=ax3)
    sns.distplot(volumes_high, kde=False, ax=ax4)
    plt.tight_layout()
    plt.show()


def plot_input_distribution_normalized(config, mode):
    dataset = StockDataset(config, 'cpu', mode=mode, normalize=True)
    prices = []
    volumes = []
    fig = plt.figure()
    for data in dataset:
        prices += data['input'][:, 0].tolist()
        volumes += data['input'][:, 1].tolist()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Normalized price')
    ax2.set_xlabel('Normalized volume')
    ax1.set_ylabel('Quantity')
    ax2.set_ylabel('Quantity')
    sns.distplot(prices, kde=False, ax=ax1)
    sns.distplot(volumes, kde=False, ax=ax2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = load_config()
    mode = 'train'
    plot_labels_distribution(config, mode)
    plot_input_distribution_unnormalized(config, mode)
    plot_input_distribution_normalized(config, mode)
