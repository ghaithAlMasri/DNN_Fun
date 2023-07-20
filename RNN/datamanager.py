import pandas as pd
import numpy as np
import pandas_ta as ta


class DGMT:
    def __init__(self, csv_path, date_col):

        self.data = pd.read_csv(csv_path, parse_dates=[date_col],index_col=date_col)
        self.data.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.data.dropna(inplace=True)
        self.data = self.data[self.data.volume != 0]
        self.df = self.data.copy()
        self.timeframe = '1min'

    def change_resolution(self, new_resolution = '1min'):
        df = self.df

        resample_dict = {'volume': 'sum', 'open': 'first',
                         'low': 'min', 'high': 'max',
                         'close': 'last'}
        
        df = self.data.resample(new_resolution).agg(resample_dict)
        # df['rsi'] = ta.rsi(self.df.close, 14)
        # df['ma_50'] = ta.ema(self.df.close, 50)
        # df['ma_100'] = ta.ema(self.df.close, 100)
        # df['ma_200'] = ta.ema(self.df.close, 200)
        df.dropna(inplace=True)
        self.df = df
        self.timeframe = new_resolution

