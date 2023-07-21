import pandas as pd
import numpy as np

class DMGT:
    def __init__(self, path) -> None:
        self.path = path
        self.current_resolution = '1min'
        self.data = pd.read_csv(path, parse_dates=['time'], index_col=['time'])
        self.data.drop(['Unnamed: 0'], axis=1, inplace=True)


        self.df = self.data.copy()

    def change_resolution(self, new_resolution:str):
        df = self.df
        resample_dict = {
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last',
            'volume':'sum',
        }

        df = df.resample(new_resolution).agg(resample_dict)
        self.df = df
        self.current_resolution = new_resolution