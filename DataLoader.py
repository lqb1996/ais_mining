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
        if np_file is not None:
            self.total_tensor = torch.from_numpy(np.load(np_file[0]))
            self.total_length = torch.from_numpy(np.load(np_file[1]))
            self.data_x = self.total_tensor[:, :, :12]
            self.data_y = self.total_tensor[:, :, -4:]
            self.datax_length = self.total_length[0]
            self.datay_length = self.total_length[1]
        elif tensor_file is not None:
            self.total_tensor = torch.load(tensor_file[0])
            self.total_length = torch.load(tensor_file[1])
            self.data_x = self.total_tensor[:, :, :12]
            self.data_y = self.total_tensor[:, :, -4:]
            self.datax_length = self.total_length[0]
            self.datay_length = self.total_length[1]
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
            # count = mmsi_group.count()['time']
            # max_mmsi=18375
            # max_mmsi = count.max()
            # max_gap_time = 21600
            base_time = int(time.mktime(time.strptime('2019-10-01 00:00:00', "%Y-%m-%d %H:%M:%S")))
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

                sorted_np = np.array(mmsi_sorted)
                biger_than_max = np.zeros(sorted_np[:, -3].shape)
                biger_than_max[sorted_np[:, -3] > max_gap_time] = 1
                isNan = np.isnan(sorted_np[:, -3])
                idx = biger_than_max + isNan
                idx = np.where(idx > 0)[0]
                for e, i in enumerate(idx):
                    if (e + 1) < len(idx) and idx[e+1] - i > 5:
                        splited = sorted_np[i: idx[e+1]]
                        splited[np.isnan(splited)] = 0
                        self.np_total_x.append(torch.from_numpy(splited))
                if biger_than_max.shape[0] - 1 - idx[-1] > 5:
                    splited = sorted_np[idx[-1]:]
                    splited[np.isnan(splited)] = 0
                    self.np_total_x.append(torch.from_numpy(splited))

                # for i, gap in mmsi_sorted.iterrows():
                #     if gap['gap_time'] > max_gap_time or gap['gap_time'] is 'NaN':
                #         if i-splited_idx > 5:   # 去除ais数据中,数据不超过5条的船,超过5条数据才能称为轨迹数据
                #             mmsi_sorted.iloc[splited_idx, 9] = 0
                #             mmsi_splited = np.array(mmsi_sorted.iloc[splited_idx: i])
                #             splited_idx = i
                #             mmsi_splited[np.isnan(mmsi_splited)] = 0  # 去除数据中的nan值
                #             print(mmsi_splited)
                #             self.np_total_x.append(torch.from_numpy(mmsi_splited))  # shape:(12920, 18374, 10)
                #         else:
                #             splited_idx = i
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
        return self.np_total_x[index]

    def __len__(self):
        return len(self.np_total_x)
