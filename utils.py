import json
import pickle
from os import path


def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


def load_pickle(filename):
    if path.exists(filename):
        print('Loading %s.' % filename)
        return pickle.load(open(filename, 'rb'))
    else:
        return None


def save_pickle(data, filename):
    print('Pickling %s.' % filename)
    pickle.dump(data, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)
    return data
