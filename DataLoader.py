import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from tqdm import tqdm


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
        base_time = int(time.mktime(time.strptime('2019-10-01 00:00:00', "%Y-%m-%d %H:%M:%S")))
        # max_gap_time = 21600
        max_gap_time = int(time.mktime(time.strptime('2019-10-01 06:00:00', "%Y-%m-%d %H:%M:%S"))) - base_time
        print("Loading csv files:")
        for mmsi, group in tqdm(mmsi_group):
            self.mmsi_list.append(mmsi)
            mmsi_sorted = group
            mmsi_sorted['time'] = mmsi_sorted['time'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))-base_time)
            mmsi_sorted = mmsi_sorted.sort_values(by="time", ascending=True, inplace=False)
            mmsi_sorted['gap_time'] = mmsi_sorted['time'].diff()
            splited_idx = 0
            mmsi_sorted = mmsi_sorted.reset_index(drop=True)
            for i, gap in mmsi_sorted.iterrows():
                if gap['gap_time'] > max_gap_time or gap['gap_time'] is 'NaN':
                    if i-splited_idx > 5:   # 去除ais数据中,数据不超过5条的船,超过5条数据才能称为轨迹数据
                        mmsi_sorted.iloc[splited_idx, 9] = 0
                        mmsi_splited = np.array(mmsi_sorted.iloc[splited_idx: i])
                        splited_idx = i
                        mmsi_splited[np.isnan(mmsi_splited)] = 0  # 去除数据中的nan值
                        self.np_total_x.append(torch.from_numpy(mmsi_splited))  # shape:(12920, 18374, 10)
                    else:
                        splited_idx = i
        print("Finished loading!")

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df_total = pd.concat([self.df_total, df])
        return df

    def __getitem__(self, index):
        return self.np_total_x[index]

    def __len__(self):
        return len(self.np_total_x)
