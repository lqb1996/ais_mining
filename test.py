import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import configparser
import time
import datetime
from module.lstm_base import *
import torch.nn.utils.rnn as rnn_utils

from joblib import Parallel, delayed
from sklearn.metrics import f1_score, log_loss, classification_report
from sklearn.model_selection import StratifiedKFold

# import lightgbm as lgb
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
    data_x = [sq[0:-1] for sq in data]
    data_y = [sq[1:, 1:3] for sq in data]
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
rnn = LSTMlight4PRE()
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()
rnn = torch.load(cf.get("path", "model_file"))
png_save_path = cf.get("path", "test_png_path")

test_iter = iter(test_loader)
# 测试一个batchsize
tx, tx_len, ty, ty_len = test_iter.next()
if torch.cuda.is_available():
    tx = tx.float().cuda()
    ty = ty.float().cuda()

tx = rnn_utils.pack_padded_sequence(tx, tx_len, batch_first=True)

# (16, 59, 2)
pre_y = rnn(tx)
n_pre = pre_y.cpu().detach().numpy()
n_y = ty.cpu().detach().numpy()

save_path = os.path.join(proDir, png_save_path)
x = np.array([])
y = np.array([])
x_pre = np.array([])
y_pre = np.array([])
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i, n in enumerate(n_pre):
    # print(np.concatenate((n, n_y[i]), axis=1)[:ty_len[i]])
    x = np.concatenate((x, n_y[i].T[0][:ty_len[i]]))
    y = np.concatenate((y, n_y[i].T[1][:ty_len[i]]))
    x_pre = np.concatenate((x_pre, n.T[0][:ty_len[i]]))
    y_pre = np.concatenate((y_pre, n.T[1][:ty_len[i]]))
plot(x=x, y=y, x_pre=x_pre, y_pre=y_pre, file=os.path.join(save_path, 'test.png'))
