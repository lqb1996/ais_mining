import os, sys, glob
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import configparser
from module.lstm_base import *
from sklearn.cluster import KMeans
from DataLoader import *
from plot import *

proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "setting.cfg")
cf = configparser.ConfigParser()
cf.read(configPath)
csv_file = os.path.join(proDir, cf.get("path", "test_file"))
csv_loader = CSVDataSet(csv_file=csv_file)

# 处理一个batchsize
def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_x = [torch.cat((sq[0:-1, 1:7], sq[0:-1, 9:]), 1) for sq in data]
    data_y = [torch.cat((sq[1:, 1:3], sq[1:, -2:]), 1) for sq in data]
    datax_length = [len(sq) for sq in data_x]
    datay_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    return data_x, datax_length, data_y, datay_length


test_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)

# rnn = LSTM4PRE()
# rnn = LSTMlight4PRE()
# rnn = ResLSTM_Attention4PRE()
rnn = TransLSTM4PRE()
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()
rnn = torch.load(cf.get("path", "model_file"))
png_save_path = cf.get("path", "test_png_path")
feature_total = None
x_total = []

for tx, tx_len, ty, ty_len in test_loader:
    if torch.cuda.is_available():
        tx = tx.float().cuda()
        ty = ty.float().cuda()
    with torch.no_grad():
        rnn.eval()
        pre_y, feature = rnn(tx, tx_len)
        feature = torch.mean(feature, dim=1)
        t_source = tx.cpu().detach().numpy()

        for i, n in enumerate(t_source):
            x = np.concatenate((x, n_y[i][:ty_len[i]][:, 2]))
            y = np.concatenate((y, n_y[i][:ty_len[i]][:, 3]))
        if feature_total is None:
            feature_total = feature.cpu().detach().numpy()
        else:
            feature_total = np.concatenate((feature_total, feature.cpu().detach().numpy()))

estimator = KMeans(n_clusters=3)
y_pred = estimator.fit_predict(feature_total)

print(y_pred)
