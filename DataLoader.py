import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils


class CSVDataSet(Dataset):
    def __init__(self, csv_path=None, csv_file=None, np_file=None, tensor_file=None):
        self.csv_files = []
        self.csv_path = csv_path
        self.mmsi_mapping = []
        self.df_total = None
        self.np_total_x = []
        self.np_total_y = []
        self.mask = []
        if np_file is not None:
            self.total_tensor = torch.from_numpy(np.load(np_file))
        elif tensor_file is not None:
            self.total_tensor = torch.load(tensor_file)
        else:
            if csv_file is not None:
                self.read_feat(csv_file)
            else:
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
                self.mmsi_mapping.append(mmsi)
                mmsi_sorted = group
                mmsi_sorted['mmsi'] = mmsi_sorted['mmsi'].apply(lambda x: self.mmsi_mapping.index(x))
                mmsi_sorted['time'] = mmsi_sorted['time'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))-base_time)
                mmsi_sorted = mmsi_sorted.sort_values(by="time", ascending=True, inplace=False)
                mmsi_sorted['gap_time'] = mmsi_sorted['time'].diff()
                mmsi_sorted['offset_lon'] = mmsi_sorted['longitude'].diff()
                mmsi_sorted['offset_lat'] = mmsi_sorted['latitude'].diff()
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
            self.np_total_x.sort(key=lambda x: len(x), reverse=True)
            data_x = [sq[0:-1] for sq in self.np_total_x]
            data_y = [torch.cat((sq[1:, 1:3], sq[1:, -2:]), 1) for sq in self.np_total_x]
            self.datax_length = torch.tensor([len(sq) for sq in data_x]).float()
            self.datay_length = torch.tensor([len(sq) for sq in data_x]).float()
            self.data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0).float()
            self.data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0).float()
            self.total_tensor = torch.cat((self.data_x, self.data_y), dim=2)
            self.total_length = torch.cat((self.datax_length, self.datay_length), dim=1)
        print("Finished loading!")

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df_total = pd.concat([self.df_total, df])
        return df

    def save_as_csv(self, file='data/total.csv'):
        self.df_total.tocsv(file)

    def save_as_np(self, file=None):
        t_data = self.total_tensor.numpy()
        if file is not None:
            np.save(file, t_data)
        else:
            np.save('total_np', t_data)

    def save_as_tensor(self, path=None):
        if path is None:
            path = ''
        torch.save(self.total_tensor, os.path.join(path, 'total_tensor'))
        torch.save(self.total_length, 'total_length')

    def __getitem__(self, index):
        return self.data_x[index], self.datax_length[index], self.data_y[index], self.datay_length[index]

    def __len__(self):
        return self.total_tensor.shape[0]
