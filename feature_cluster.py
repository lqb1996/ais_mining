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

rnn = TransLSTM4PRE(num_hidden_encoder_layers=6)
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()
rnn = torch.load(cf.get("path", "model_file"))
png_save_path = cf.get("path", "test_png_path")
feature_total = None
x_len = []
x = np.array([])
y = np.array([])

for tx, tx_len, ty, ty_len in test_loader:
    if torch.cuda.is_available():
        tx = tx.float().cuda()
        ty = ty.float().cuda()
    with torch.no_grad():
        rnn.eval()
        pre_y, feature = rnn(tx, tx_len)
        t_source = tx.cpu().detach().numpy()
        x_len += tx_len

        for i, n in enumerate(t_source):
            x = np.concatenate((x, t_source[i][:tx_len[i]][:, 0]))
            feature_mean = torch.mean(feature[i][:tx_len[i]].view(tx_len[i], -1), dim=0)
            y = np.concatenate((y, t_source[i][:tx_len[i]][:, 1]))
            if feature_total is None:
                feature_total = feature_mean.cpu().detach().numpy()[np.newaxis, :]
            else:
                feature_total = np.concatenate((feature_total, feature_mean.cpu().detach().numpy()[np.newaxis, :]), axis=0)

estimator = KMeans(n_clusters=5)
kmeans_pred = estimator.fit_predict(feature_total)
c = np.array([])
for i, k in enumerate(kmeans_pred):
    z = np.zeros((x_len[i],), dtype=np.int)
    z[:] = k
    c = np.concatenate((c, z))
cluster_cls = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], c[np.newaxis, :]), axis=0).T
# plt.scatter(x, y, c=c, alpha=0.3)
# plt.savefig('test_kmeans.png')
# print(cluster_cls.shape)
cluster_plot(cluster_cls)
