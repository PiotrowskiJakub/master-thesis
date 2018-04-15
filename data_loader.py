import csv
import datetime
import pickle
from os import path

import quandl

from utils import load_config


class DataLoader:

    def __init__(self):
        config = load_config()['data']
        quandl.ApiConfig.api_key = config['quandl_key']
        tickers_path = config['tickers_path']
        self.data_path = config['data_path']
        self.tickers = ['WIKI/' + ticker for ticker in DataLoader._load_tickers(tickers_path)]
        self.start_date = datetime.date(1990, 1, 1)
        # self.start_date = datetime.date(2018, 3, 27)
        self.end_date = datetime.datetime.now().date()

    @staticmethod
    def _load_tickers(tickers_path):
        with open(tickers_path, 'r') as f:
            tickers = list(csv.reader(f))[0]

        return tickers

    @staticmethod
    def _load_pickle(filename):
        if path.exists(filename):
            print('Loading %s.' % filename)
            return pickle.load(open(filename, 'rb'))
        else:
            return None

    @staticmethod
    def _save_pickle(data, filename):
        print('Pickling %s.' % filename)
        pickle.dump(data, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)
        return data

    def load(self):
        data = DataLoader._load_pickle(self.data_path)
        if data is None:
            data = quandl.get(self.tickers, start_date=self.start_date, end_date=self.end_date)
            DataLoader._save_pickle(data, self.data_path)

        return data
