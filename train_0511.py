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
import torch.optim.lr_scheduler as lr_scheduler
from DataLoader import *

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
    # longitude,latitude,cog,rot,trueHeading,sog,timegap,gap,gap,gap,longap,latgap
    data_x = [torch.cat((sq[0:-1, 1:7], sq[0:-1, 9:]), 1) for sq in data]
    data_y = [torch.cat((sq[1:, 1:3], sq[1:, -2:]), 1) for sq in data]
    # data_y = [sq[1:, -2:] for sq in data]
    datax_length = [len(sq) for sq in data_x]
    datay_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    return data_x, datax_length, data_y, datay_length


train_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)
test_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)

save_path = os.path.join(proDir, cf.get("path", "res_path"), time.strftime("%m-%d_%H:%M", time.localtime()))
# rnn = LSTM4PRE()
# rnn = LSTMlight4PRE()
# rnn = LSTM_Attention4PRE()
# rnn = ResLSTM_Attention4PRE()
rnn = TransLSTM4PRE(num_hidden_encoder_layers=6)

os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=float(cf.get("super-param", "lr")))  # optimize all cnn parameters
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=1e-06, eps=0)

loss_func = nn.MSELoss()
best_loss = 1000


for step in range(int(cf.get("super-param", "epoch"))):
    mean_loss = []
    for batch, (tx, tx_len, ty, ty_len) in enumerate(train_loader):  # (batch_size, length_per_sequence, feature_per_words)
        rnn.train()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()
        ttruth = tx
        output, _ = rnn(tx, tx_len)
        # 根据预测序列所用到的长短调整loss
        for i, y in enumerate(ty):
            offset_loss = loss_func(output[i][:ty_len[i]], y[:ty_len[i], -2:])
            truth_loss = loss_func(output[i][:ty_len[i]]+ttruth[i][:ty_len[i], :2], y[:ty_len[i], :2])
            # offset_loss = torch.abs(offset_loss*torch.log(offset_loss))
            # truth_loss = torch.abs(truth_loss*torch.log(truth_loss))
            loss = offset_loss + truth_loss
        loss.backward()  # back propagation, compute gradients
        # 使用L2梯度裁剪
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 2)
        mean_loss.append(loss.cpu().item())
        if (batch+1) % 32 == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.step()
    optimizer.zero_grad()
    # 基于loss的lr_decay
    scheduler.step(loss)

    with torch.no_grad():
        rnn.eval()
        tx, tx_len, ty, ty_len = iter(test_loader).next()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()

        output, _ = rnn(tx, tx_len)
        loss = loss_func(output, ty[:, :, -2:])
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
