import csv
import datetime

import numpy as np
import quandl
import torch
from torch.utils.data import Dataset

from utils import load_pickle, save_pickle


class StockDataset(Dataset):

    def __init__(self, config, device):
        data_config = config['data']
        raw_data = StockDataset._load_raw_data(data_config)
        self._device = device
        self._input_label_pairs = StockDataset._prepare_raw_data(raw_data, data_config=data_config)

    @staticmethod
    def _load_raw_data(data_config):
        data_path = data_config['data_path']
        data = load_pickle(data_path)
        if data is None:
            quandl.ApiConfig.api_key = data_config['quandl_key']
            tickers = ['WIKI/' + ticker for ticker in load_tickers(data_config['tickers_path'])]
            start_date = datetime.date(1990, 1, 1)
            end_date = datetime.datetime.now().date()
            data = quandl.get(tickers, start_date=start_date, end_date=end_date)
            save_pickle(data, data_path)

        return data

    @staticmethod
    def _prepare_raw_data(raw_data, data_config):
        input_label_pairs = []

        past_days = data_config['past_days']
        forecast_days = data_config['forecast_days']

        close = select_column(raw_data, 'Adj. Close').as_matrix()
        volume = select_column(raw_data, 'Adj. Volume').as_matrix()
        max_close = np.nanmax(close)
        max_volume = np.nanmax(volume)

        for company_num in range(close.shape[1]):
            i = 0
            while i < close.shape[0]:
                close_input = remove_nan(close[i:i + past_days, company_num])
                volume_input = remove_nan(volume[i:i + past_days, company_num])
                prices = remove_nan(close[i + past_days:i + past_days + forecast_days, company_num])
                i = i + past_days
                if prices.size == 0 or close_input.size == 0 or volume_input.size == 0:
                    continue
                max_price = np.max(prices)
                last_mean = np.mean(close_input[-forecast_days:])
                change_percentage = (max_price - last_mean) / last_mean
                y = StockDataset._generate_labels(change_percentage, data_config=data_config)
                derivatives = np.diff(close_input)
                derivatives = np.append(derivatives, derivatives[-1])
                input = list(zip(close_input / max_close, volume_input / max_volume))
                input_label_pairs.append((input, y))

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

    def __len__(self):
        return len(self._input_label_pairs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self._input_label_pairs[idx][0], dtype=torch.float, device=self._device)
        label_tensor = torch.tensor(self._input_label_pairs[idx][1], dtype=torch.float, device=self._device)
        sample = {
            'input': input_tensor,
            'label': label_tensor
        }

        return sample


def load_tickers(tickers_path):
    with open(tickers_path, 'r') as f:
        tickers = list(csv.reader(f))[0]

    return tickers


def remove_nan(array):
    return array[~np.isnan(array)]


def select_column(data, col_name):
    return data.select(lambda col: col.endswith(col_name), axis=1)
