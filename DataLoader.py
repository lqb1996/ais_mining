import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime


class CSVDataSet(Dataset):
    def __init__(self, csv_path):
        self.csv_files = []
        self.mmsi_list = []
        self.df_total = None
        self.np_total_x = []
        self.np_total_y = []
        self.mask = []
        for root, dirs, files in os.walk(csv_path):
            for f in files:
                if f.endswith('.csv'):
                    self.csv_files.append(os.path.join(root, f))
        for csv_file in self.csv_files:
            self.read_feat(csv_file)

        mmsi_group = self.df_total.groupby('mmsi')
        count = mmsi_group.count()['time']
        max_mmsi = count.max()  # max_mmsi=18375
        for mmsi, group in mmsi_group:
            if count[mmsi] > 10:    # 去除ais数据中,数据不超过10条的船,超过10条数据才能称为轨迹数据
                self.mmsi_list.append(mmsi)
                mmsi_sorted = np.array(group.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8]])    # 去除time(String)
                mmsi_sorted[np.isnan(mmsi_sorted)] = 0  # 去除数据中的nan值
                self.np_total_x.append(torch.from_numpy(mmsi_sorted))  # shape:(12920, 18374, 8)

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df_total = pd.concat([self.df_total, df])
        return df

    def __getitem__(self, index):
        return self.np_total_x[index]

    def __len__(self):
        return len(self.np_total_x)
