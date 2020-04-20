import os, sys, glob
import numpy as np
import pandas as pd

import time
import datetime


class CSVDataLoader:
    def __init__(self, csv_path):
        self.csv_files = []
        for root, dirs, files in os.walk(csv_path):
            for f in files:
                if f.endswith('.csv'):
                    self.csv_files.append(os.path.join(root, f))
        self.read_feat(self.csv_files[0])

    def read_feat(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.iloc[::-1]

        # if test_mode:
        #     df_feat = [df['渔船ID'].iloc[0], df['type'].iloc[0]]
        #     df = df.drop(['type'], axis=1)
        # else:
        #     df_feat = [df['渔船ID'].iloc[0]]

        df_feat = [df['mmsi'].iloc[0]]

        df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        df_diff = df.diff(1).iloc[1:]
        df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
        df_diff['dis'] = np.sqrt(df_diff['longitude'] ** 2 + df_diff['latitude'] ** 2)

        df_feat.append(df['time'].dt.day.nunique())
        df_feat.append(df['time'].dt.hour.min())
        df_feat.append(df['time'].dt.hour.max())
        df_feat.append(df['time'].dt.hour.value_counts().index[0])

        df_feat.append(df['sog'].min())
        df_feat.append(df['sog'].max())
        df_feat.append(df['sog'].mean())

        # df_feat.append(df_diff['time'].min())
        # df_feat.append(df_diff['time'].max())
        # df_feat.append(df_diff['time'].mean())

        df_feat.append(df_diff['sog'].min())
        df_feat.append(df_diff['sog'].max())
        df_feat.append(df_diff['sog'].mean())
        df_feat.append((df_diff['sog'] > 0).mean())
        df_feat.append((df_diff['sog'] == 0).mean())

        df_feat.append(df_diff['cog'].min())
        df_feat.append(df_diff['cog'].max())
        df_feat.append(df_diff['cog'].mean())
        df_feat.append((df_diff['cog'] > 0).mean())
        df_feat.append((df_diff['cog'] == 0).mean())

        df_feat.append((df_diff['longitude'].abs() / df_diff['time_seconds']).min())
        df_feat.append((df_diff['longitude'].abs() / df_diff['time_seconds']).max())
        df_feat.append((df_diff['longitude'].abs() / df_diff['time_seconds']).mean())
        df_feat.append((df_diff['longitude'] > 0).mean())
        df_feat.append((df_diff['longitude'] == 0).mean())

        df_feat.append((df_diff['latitude'].abs() / df_diff['time_seconds']).min())
        df_feat.append((df_diff['latitude'].abs() / df_diff['time_seconds']).max())
        df_feat.append((df_diff['latitude'].abs() / df_diff['time_seconds']).mean())
        df_feat.append((df_diff['latitude'] > 0).mean())
        df_feat.append((df_diff['latitude'] == 0).mean())

        df_feat.append(df_diff['dis'].min())
        df_feat.append(df_diff['dis'].max())
        df_feat.append(df_diff['dis'].mean())

        df_feat.append((df_diff['dis'] / df_diff['time_seconds']).min())
        df_feat.append((df_diff['dis'] / df_diff['time_seconds']).max())
        df_feat.append((df_diff['dis'] / df_diff['time_seconds']).mean())
        print(df_diff)
        print(df_feat)

        return df_feat
