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
csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
csv_loader = CSVDataSet(csv_path)


# 处理一个batchsize
def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_x = [sq[0:-1] for sq in data]
    data_y = [sq[1:, :2] for sq in data]
    datax_length = [len(sq) for sq in data_x]
    datay_length = [len(sq) for sq in data_x]
    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    return data_x, datax_length, data_y, datay_length


train_loader = DataLoader(dataset=csv_loader,
                          batch_size=int(cf.get("super-param", "batch_size")),
                          collate_fn=collate_fn,
                          shuffle=True)

rnn = LSTM4PRE()

os.environ["CUDA_VISIBLE_DEVICES"] = cf.get("super-param", "gpu_ids")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if torch.cuda.is_available():
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=float(cf.get("super-param", "lr")))  # optimize all cnn parameters
loss_func = nn.MSELoss()

best_loss = 1000


# 去掉mask并计算损失
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    # crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = loss_func(inp, target).masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


if not os.path.exists('weights'):
    os.mkdir('weights')

for step in range(int(cf.get("super-param", "epoch"))):
    mean_loss = []
    for tx, tx_len, ty, ty_len in train_loader:  # (batch_size, length_per_sequence, feature_per_words)
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()

        tx = rnn_utils.pack_padded_sequence(tx, tx_len, batch_first=True)
        output = rnn(tx)
        # out_pad, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        loss = loss_func(output, ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
        mean_loss.append(loss.cpu().item())

    # with torch.no_grad():
    #     for tx, tx_len, ty, ty_len in train_loader:
    #         if torch.cuda.is_available():
    #             tx = tx.float().cuda()
    #             ty = ty.float().cuda()
    #
    #         tx = rnn_utils.pack_padded_sequence(tx, tx_len, batch_first=True)
    #         output = rnn(tx)
    #         loss = loss_func(output, ty)
    #
    #         print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item())
    sum = 0
    for item in mean_loss:
        sum += item
    m = sum / len(mean_loss)
    print('epoch : %d  ' % step, 'train_loss : %.4f' % m)
    torch.save(rnn, 'weights/rnn.pkl'.format(m))
    print('new model saved at epoch {} with val_loss {}'.format(step, m))
