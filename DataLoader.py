import os, sys, glob
import numpy as np
import pandas as pd

import time
import datetime


class CSVDataLoader:
    def __init__(self, csv_path):
        self.csv_files = []
        self.mmsi_list = []
        self.sort_by_mmsi = {}
        self.df_total = None
        self.np_total = None
        for root, dirs, files in os.walk(csv_path):
            for f in files:
                if f.endswith('.csv'):
                    self.csv_files.append(os.path.join(root, f))
        for csv_file in self.csv_files:
            self.read_feat(csv_file)
        mmsi_group = self.df_total.groupby('mmsi')
        count = mmsi_group.count()['time']
        max_mmsi = count.max()
        for mmsi, group in mmsi_group:
            if count[mmsi] > 10:
                self.mmsi_list.append(mmsi)
                # self.sort_by_mmsi[mmsi] = group
                # print(group)
        print(self.sort_by_mmsi)
        print(count)

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.iloc[::-1]
        self.df_total = pd.concat([self.df_total, df])
        return df
