import pandas as pd
from sklearn.utils import shuffle


class DataSet:
    def __init__(self, filepath, data_name, sep=","):
        self.name = data_name
        self.filepath = filepath
        self.sep = sep
        self.df = None
        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.filepath, self.sep, header=None)
        if self.name == 'iris':
            del df[0]
            del df[3]
            df.columns = ['x', 'y', 'label']
            df[['x', 'y']] = df[['x', 'y']].astype(float)
            df = df[df['label'] != "Iris-setosa"]
            df['label'] = df['label'].map({'Iris-versicolor': 1, 'Iris-virginica': -1})
            df[['label']] = df[['label']].astype(int)
        else:
            df.columns = ['x', 'label', 'y']
            df[['x', 'y']] = df[['x', 'y']].astype(float)
            df[['label']] = df[['label']].astype(int)
            df['label'] = df['label'].map({2: -1, 1: 1})
        df['weight'] = 0
        self.df = df
        # df.sample(frac=1)

    def print_data(self):
        print(self.df)
