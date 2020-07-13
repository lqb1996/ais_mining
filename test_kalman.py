import os, sys, glob
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import configparser
from module.lstm_large import *
import torch.nn.utils.rnn as rnn_utils

from utils.BLexchangeXY import *
from utils.XYexchangeBL import *
from DataLoader import *
from KalmanFilter import *
from plot import *

proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "setting.cfg")
cf = configparser.ConfigParser()
cf.read(configPath)
csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
csv_loader = CSVDataSet(csv_path)


# 处理一个batchsize
def collate_fn(data):
    # mmsi,longitude,latitude,cog,rot,trueHeading,sog,time,navStatus,gap,gap,gap
    data.sort(key=lambda x: len(x), reverse=True)
    data_truth = [torch.cat((sq[0:-1, 1:7], sq[0:-1, 9:]), 1) for sq in data]
    data_y = [torch.cat((sq[1:, 1:3], sq[1:, -2:]), 1) for sq in data]
    data_x = []
    nm2m = 1.852 * 5 / 18   # 单位转换参数,km/ms
    for idx, d in enumerate(data):
        offset = None
        gap_time = data[idx][1:, 9].unsqueeze(1)
        lon = torch.from_numpy(KalmanFilter(data[idx][0:-1, 1].numpy()).get_res()).unsqueeze(1)
        lat = torch.from_numpy(KalmanFilter(data[idx][0:-1, 2].numpy()).get_res()).unsqueeze(1)
        cog_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 3].numpy()).get_res()).unsqueeze(1)
        rot_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 4].numpy()).get_res()).unsqueeze(1)
        head_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 5].numpy()).get_res()).unsqueeze(1)
        sog_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 6].numpy()).get_res()).unsqueeze(1)
        gaptime_offset = torch.from_numpy(KalmanFilter(data[idx][1:, 9].numpy()).get_res()).unsqueeze(1)
        offsog_offset = torch.from_numpy(KalmanFilter(data[idx][1:, -4].numpy()).get_res()).unsqueeze(1)
        lon_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, -2].numpy()).get_res()).unsqueeze(1)
        lat_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, -1].numpy()).get_res()).unsqueeze(1)
        lon_gentle = torch.from_numpy(KalmanFilter(data[idx][0:-1, 1].numpy(), R=1*2).get_res()).unsqueeze(1)
        lat_gentle = torch.from_numpy(KalmanFilter(data[idx][0:-1, 2].numpy(), R=1*2).get_res()).unsqueeze(1)
        lon_offset_gentle = torch.from_numpy(KalmanFilter(data[idx][0:-1, -2].numpy(), R=1*2).get_res()).unsqueeze(1)
        lat_offset_gentle = torch.from_numpy(KalmanFilter(data[idx][0:-1, -1].numpy(), R=1*2).get_res()).unsqueeze(1)
        for oi, o in enumerate(d[0:-1, :]):
            v_x = o[6] * math.sin(math.radians(o[3])) * nm2m
            v_y = o[6] * math.cos(math.radians(o[3])) * nm2m
            pre_offset_x = v_x * d[oi+1][9]
            pre_offset_y = v_y * d[oi+1][9]
            pre_offset_lon, pre_offset_lat = transformMercatorToLngLat(pre_offset_x, pre_offset_y)
            if offset is not None:
                offset = torch.cat((offset, torch.unsqueeze(torch.tensor([pre_offset_lon, pre_offset_lat]), 0)), 0)
            else:
                offset = torch.unsqueeze(torch.tensor([pre_offset_lon, pre_offset_lat]), 0)
        data_x.append(torch.cat((d[0:-1, 1:8], d[0:-1, 10:], gaptime_offset,  cog_offset, rot_offset, sog_offset, lon, lat, lon_offset, lat_offset, offset, gap_time), 1))
        # data_x.append(torch.cat((d[0:-1, 3:7], lon, lat, lon_offset, lat_offset, offset, gap_time), 1))
    datax_length = [len(sq) for sq in data_x]
    datay_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    data_truth = rnn_utils.pad_sequence(data_truth, batch_first=True, padding_value=0)
    return data_x, datax_length, data_y, datay_length, data_truth


test_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)

rnn = KalmanTrans4PRE_large(input_size=24, hidden_size=24*2, num_hidden_encoder_layers=12)
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()
rnn = torch.load(cf.get("path", "model_file"))
png_save_path = cf.get("path", "test_png_path")

loss_func = nn.MSELoss()
x_truth = np.array([])
y_truth = np.array([])
x_pre = np.array([])
y_pre = np.array([])
lon = np.array([])
lat = np.array([])
suanhaode_x = np.array([])
suanhaode_y = np.array([])
suanhaode_lon = np.array([])
suanhaode_lat = np.array([])
lon_pre = np.array([])
lat_pre = np.array([])
x_truth_lon = np.array([])
y_truth_lat = np.array([])
s = "let route = '"
truth = []
predict = []
flag = 0

for tx, tx_len, ty, ty_len, ttruth in test_loader:
    if torch.cuda.is_available():
        tx = tx.float().cuda()
        ty = ty.float().cuda()
        ttruth = ttruth.float().cuda()
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
            lon_pre = np.concatenate((lon_pre, n[:ty_len[i]][:, 0]+ttruth[i][:tx_len[i]][:, 0].cpu().detach().numpy()))
            lat_pre = np.concatenate((lat_pre, n[:ty_len[i]][:, 1]+ttruth[i][:tx_len[i]][:, 1].cpu().detach().numpy()))
            if flag < 1 and len(n) > 50 and len(n) < 150:
                route_np = np.array([
                    ty[i][:tx_len[i]][:, 0].cpu().detach().numpy(),
                    ty[i][:tx_len[i]][:, 1].cpu().detach().numpy(),
                    n[:ty_len[i]][:, 0]+(ttruth[i][:tx_len[i]][:, 0].cpu().detach().numpy()),
                    n[:ty_len[i]][:, 1]+(ttruth[i][:tx_len[i]][:, 1].cpu().detach().numpy()),
                    tx[i][:tx_len[i]][:, 7].cpu().detach().numpy()
                ]).T
                loss = None
                offset_loss = loss_func(pre_y[i][:ty_len[i]], ty[i][:tx_len[i]][:, 2:])
                truth_loss = loss_func(pre_y[i][:ty_len[i]] + ttruth[i][:ty_len[i], :2], ty[i][:tx_len[i]][:, :2])
                loss = offset_loss ** 2 + truth_loss ** 2
                if loss < 0.00001:
                    base_time = int(time.mktime(time.strptime('2019-10-01 00:00:00', "%Y-%m-%d %H:%M:%S")))
                    for r in route_np:
                        date = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(r[4] + base_time))
                        truth.append({"lon": str(r[0]), "lat": str(r[1]), "date": date, "type": "truth"})
                        predict.append({"lon": str(r[2]), "lat": str(r[3]), "date": date, "type": "predict"})
                    print(route_np.shape)
                    print(json.dumps([truth, predict]))
                    flag += 1
save_path = os.path.join(proDir, png_save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

s += json.dumps([truth, predict]) + "';"
with open(os.path.join(save_path, 'show_points.js'), 'w') as show_points:
    show_points.write(s)
plot(x=lon, y=lat, x_pre=lon_pre, y_pre=lat_pre, file=os.path.join(save_path, 'test_location.png'))
plot(x=x_truth, y=y_truth, x_pre=x_pre, y_pre=y_pre, file=os.path.join(save_path, 'test_mixed.png'))
plot(x=[], y=[], x_pre=x_pre, y_pre=y_pre, file=os.path.join(save_path, 'test_pre.png'))
