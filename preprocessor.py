import numpy as np
from sklearn import preprocessing

PAST_DAYS = 100
FORECAST_DAYS = 10
CHANGE_THRESHOLD_BOUNDARIES = [0.1, 0.05, 0.03]  # Price change boundaries


class Preprocessor:

    def __init__(self, data):
        self.data = data

    def prepare_dataset(self):
        inputs, labels = [], []
        close = self._select('Adj. Close').as_matrix()
        volume = self._select('Adj. Volume').as_matrix()
        for company_num in range(close.shape[1]):
            i = 0
            while i < close.shape[0]:
                close_X = self._remove_nan(close[i:i + PAST_DAYS, company_num])
                volume_X = self._remove_nan(volume[i:i + PAST_DAYS, company_num])
                prices = self._remove_nan(close[i + PAST_DAYS:i + PAST_DAYS + FORECAST_DAYS, company_num])
                i = i + PAST_DAYS
                if prices.size == 0 or close_X.size == 0:
                    continue
                max_price = np.max(prices)
                last_mean = np.mean(close_X[-FORECAST_DAYS:])
                change_percentage = (max_price - last_mean) / last_mean
                y = self._generate_labels(change_percentage)
                derivatives = np.diff(close_X)
                derivatives = np.append(derivatives, derivatives[-1])
                inputs.append(list(zip(preprocessing.scale(close_X), preprocessing.scale(volume_X))))
                labels.append(y)

        return inputs, labels

    def _remove_nan(self, array):
        return array[~np.isnan(array)]

    def _generate_labels(self, change_percentage):
        """Creates a vector that shows price changes.
        [1 0 0 0 0] - the price fell by more than 5%
        [0 1 0 0 0] - the price fell by more than 3% but less than 5%
        [0 0 1 0 0] - the price has not changed significantly
        [0 0 0 1 0] - the price rose by more than 3% but less than 5%
        [0 0 0 0 1] - the price rose by more than 5%
        """
        vector_length = len(CHANGE_THRESHOLD_BOUNDARIES) * 2 + 1
        change_vector = ([0] * vector_length)

        for idx, threshold in enumerate(CHANGE_THRESHOLD_BOUNDARIES):
            if change_percentage < -threshold:
                change_vector[idx] = 1
                return change_vector
            elif change_percentage > threshold:
                change_vector[vector_length - 1 - idx] = 1
                return change_vector

        change_vector[len(CHANGE_THRESHOLD_BOUNDARIES)] = 1
        return change_vector

    def _select(self, col_name):
        return self.data.select(lambda col: col.endswith(col_name), axis=1)
