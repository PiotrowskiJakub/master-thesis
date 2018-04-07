from data_loader import DataLoader
from preprocessor import Preprocessor

if __name__ == '__main__':
    data = DataLoader().load()
    preprocessor = Preprocessor(data)
    close = preprocessor.select('Adj. Close')
    volume = preprocessor.select('Adj. Volume')
    print('All done')
