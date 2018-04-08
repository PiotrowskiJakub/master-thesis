from data_loader import DataLoader
from preprocessor import Preprocessor

if __name__ == '__main__':
    data = DataLoader().load()
    preprocessor = Preprocessor(data)
    X, y = preprocessor.prepare_dataset()
    print('All done')
