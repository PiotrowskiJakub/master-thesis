class Preprocessor:

    def __init__(self, data):
        self.data = data

    def select(self, col_name):
        return self.data.select(lambda col: col.endswith(col_name), axis=1)
