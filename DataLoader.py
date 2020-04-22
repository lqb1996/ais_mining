import os, sys, glob
import numpy as np
import pandas as pd

import time
import datetime


class CSVDataLoader:
    def __init__(self, csv_path):
        self.csv_files = []
        self.mmsi = []
        self.sort_by_mmsi = {}
        self.df_total = None
        self.np_total = None
        for root, dirs, files in os.walk(csv_path):
            for f in files:
                if f.endswith('.csv'):
                    self.csv_files.append(os.path.join(root, f))
        for csv_file in self.csv_files:
            self.read_feat(csv_file)
        for mmsi, group in self.df_total.groupby('mmsi'):
            print(mmsi)
            print(group)

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.iloc[::-1]
        self.df_total = pd.concat([self.df_total, df])
        return df