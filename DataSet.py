import pandas as pd


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
            df['weight'] = 0
        else:
            df.columns = ['x', 'label', 'y']
            df[['x', 'y']] = df[['x', 'y']].astype(float)
            df[['label']] = df[['label']].astype(int)
            df['label'] = df['label'].map({2: -1, 1: 1})
            df['weight'] = 0
            colnames = df.columns.tolist()
            colnames = [colnames[0]] + [colnames[2]] + [colnames[1]] + [colnames[3]]
            df = df[colnames]
        lst = df.values
        self.df = lst

    def print_data(self):
        print(self.df)
