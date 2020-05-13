import os, sys, glob
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import configparser
from module.lstm_base import *
import torch.nn.utils.rnn as rnn_utils

from DataLoader import *
from plot import *

proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "setting.cfg")
cf = configparser.ConfigParser()
cf.read(configPath)
csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
csv_loader = CSVDataSet(csv_path)

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

rnn = TransLSTM4PRE()
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()
rnn = torch.load(cf.get("path", "model_file"))
png_save_path = cf.get("path", "test_png_path")

x_truth = np.array([])
y_truth = np.array([])
x_pre = np.array([])
y_pre = np.array([])
lon = np.array([])
lat = np.array([])
lon_pre = np.array([])
lat_pre = np.array([])

for tx, tx_len, ty, ty_len in test_loader:
    if torch.cuda.is_available():
        tx = tx.float().cuda()
        ty = ty.float().cuda()
    with torch.no_grad():
        rnn.eval()
        pre_y, _ = rnn(tx, tx_len)

        for i, n in enumerate(pre_y.cpu().detach().numpy()):
            lon = np.concatenate((lon, ty[i][:tx_len[i]][:, 0].cpu().detach().numpy()))
            lat = np.concatenate((lat, ty[i][:tx_len[i]][:, 1].cpu().detach().numpy()))
            x_truth = np.concatenate((x_truth, ty[i][:tx_len[i]][:, 2].cpu().detach().numpy()))
            y_truth = np.concatenate((y_truth, ty[i][:tx_len[i]][:, 3].cpu().detach().numpy()))
            x_pre = np.concatenate((x_pre, n[:ty_len[i]][:, 0]))
            y_pre = np.concatenate((y_pre, n[:ty_len[i]][:, 1]))
            lon_pre = np.concatenate((lon_pre, n[:ty_len[i]][:, 0]+tx[i][:tx_len[i]][:, 0].cpu().detach().numpy()))
            lat_pre = np.concatenate((lat_pre, n[:ty_len[i]][:, 1]+tx[i][:tx_len[i]][:, 1].cpu().detach().numpy()))

save_path = os.path.join(proDir, png_save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# s = "let route = '"
# truth = []
# predict = []
#     for j, p in enumerate(n_y[i][:ty_len[i]][:, 0]):
#         truth.append({"lon": str(p), "lat": str(n_y[i][:ty_len[i]][j, 1])})
#         predict.append({"lon": str(n[:ty_len[i]][j, 0]), "lat": str(n[:ty_len[i]][j, 1])})
# s += json.dumps([truth, predict]) + "';"
# with open(os.path.join(save_path, 'show_points.js'), 'w') as show_points:
#     show_points.write(s)
plot(x=lon, y=lat, x_pre=lon_pre, y_pre=lat_pre, file=os.path.join(save_path, 'test_location.png'))
plot(x=x_truth, y=y_truth, x_pre=x_pre, y_pre=y_pre, file=os.path.join(save_path, 'test_mixed.png'))
plot(x=[], y=[], x_pre=x_pre, y_pre=y_pre, file=os.path.join(save_path, 'test_pre.png'))
plot(x=x_truth, y=y_truth, x_pre=[], y_pre=[], file=os.path.join(save_path, 'test_truth.png'))
