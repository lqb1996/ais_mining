import os, sys, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import configparser
import time
import datetime
from module.bert_lstm import *
from utils.BLexchangeXY import *
from utils.XYexchangeBL import *
import torch.nn.utils.rnn as rnn_utils
from DataLoader import *
from KalmanFilter import *
import torch.optim.lr_scheduler as lr_scheduler

proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "setting.cfg")
cf = configparser.ConfigParser()
cf.read(configPath)
csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
total_np_file = os.path.join(proDir, cf.get("path", "total_np_file"))
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
        gap_time = torch.unsqueeze(data[idx][1:, 9], 1)
        lon = torch.from_numpy(KalmanFilter(data[idx][0:-1, 1].numpy()).get_res()).unsqueeze(1)
        lat = torch.from_numpy(KalmanFilter(data[idx][0:-1, 2].numpy()).get_res()).unsqueeze(1)
        cog_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 3].numpy()).get_res()).unsqueeze(1)
        rot_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 4].numpy()).get_res()).unsqueeze(1)
        sog_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, 6].numpy()).get_res()).unsqueeze(1)
        gaptime_offset = torch.from_numpy(KalmanFilter(data[idx][1:, 9].numpy()).get_res()).unsqueeze(1)
        lon_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, -2].numpy()).get_res()).unsqueeze(1)
        lat_offset = torch.from_numpy(KalmanFilter(data[idx][0:-1, -1].numpy()).get_res()).unsqueeze(1)
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
        # data_x.append(torch.cat((d[0:-1, 3:7], offset, gap_time), 1))
        data_x.append(torch.cat((d[0:-1, 1:8], d[0:-1, 10:], gaptime_offset,  cog_offset, rot_offset, sog_offset, lon, lat, lon_offset, lat_offset, offset, gap_time), 1))
    datax_length = [len(sq) for sq in data_x]
    datay_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    data_truth = rnn_utils.pad_sequence(data_truth, batch_first=True, padding_value=0)
    return data_x, datax_length, data_y, datay_length, data_truth


train_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)
test_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)
save_path = os.path.join(proDir, cf.get("path", "res_path"), time.strftime("%m-%d_%H:%M", time.localtime()))

rnn = Simple_BERT_LSTM4PRE(num_hidden_encoder_layers=6, input_size=24, hidden_size=48)

os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=float(cf.get("super-param", "lr")))  # optimize all cnn parameters
loss_func = nn.MSELoss()
best_loss = 1000
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=1e-06, eps=0)

# 去掉mask并计算损失
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    # crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = loss_func(inp, target).masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


for step in range(int(cf.get("super-param", "epoch"))):
    mean_loss = []
    for batch, (tx, tx_len, ty, ty_len, ttruth) in enumerate(train_loader):  # (batch_size, length_per_sequence, feature_per_words)
        rnn.train()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()
            ttruth = ttruth.float().cuda()

        output, _ = rnn(tx, tx_len)

        # 直接对输出结果计算MSE损失
        # loss = loss_func(output, ty)
        # 分开计算输出结果计算MSE损失
        # lamda = float(cf.get("super-param", "lamda"))
        # loss1 = loss_func(output[:, :2], ty[:, :2])
        # loss2 = loss_func(output[:, -2:], ty[:, -2:])
        # loss = loss1 + lamda * loss2
        loss = None
        for i, y in enumerate(ty):
            offset_loss = loss_func(output[i][:ty_len[i]], y[:ty_len[i], -2:])
            truth_loss = loss_func(output[i][:ty_len[i]]+ttruth[i][:ty_len[i], :2], y[:ty_len[i], :2])
            if loss is not None:
                loss += offset_loss**2 + truth_loss**2
            else:
                loss = offset_loss**2 + truth_loss**2
        loss.backward()  # back propagation, compute gradients
        mean_loss.append(loss.cpu().item())
        if (batch+1) % 128 == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.step()
    optimizer.zero_grad()
    # 基于loss的lr_decay
    scheduler.step(loss)

    with torch.no_grad():
        rnn.eval()
        tx, tx_len, ty, ty_len, ttruth = iter(test_loader).next()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()
            ttruth = ttruth.float().cuda()

        output, _ = rnn(tx, tx_len)
        loss = None
        for i, y in enumerate(ty):
            offset_loss = loss_func(output[i][:ty_len[i]], y[:ty_len[i], -2:])
            truth_loss = loss_func(output[i][:ty_len[i]] + ttruth[i][:ty_len[i], :2], y[:ty_len[i], :2])
            if loss is not None:
                loss += offset_loss ** 2 + truth_loss ** 2
            else:
                loss = offset_loss ** 2 + truth_loss ** 2
        sum = 0
        for item in mean_loss:
            sum += item
        m = sum / len(mean_loss)
        print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item(), 'train_loss : %.4f' % m, 'lr : %.8f' % optimizer.param_groups[0]['lr'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(rnn, os.path.join(save_path, 'lstm4pre%d.pkl' % step))
        with open(os.path.join(save_path, 'lstm4pre.log'), 'a+') as loss4log:
            loss4log.write('epoch : %d  lr : %.8f  train_loss : %.4f  val_loss : %.4f\n' % (step, optimizer.param_groups[0]['lr'], m, loss.cpu().item()))
