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
    data.sort(key=lambda x: len(x), reverse=True)
    data_x = [sq[0:-1] for sq in data]
    # data_y = [torch.cat((sq[0:-1, 1:3], sq[0:-1, -2:]), 1) for sq in data]
    data_y = [sq[1:, -2:] for sq in data]
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

save_path = os.path.join(proDir, cf.get("path", "res_path"), time.asctime(time.localtime(time.time())))
# rnn = LSTM4PRE()
# rnn = LSTMlight4PRE()
rnn = LSTMTiny4PRE()
# rnn = bilstm_attn(int(cf.get("super-param", "batch_size")), 4, 512, True, 0.4, True, 64, 12)

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


for step in range(int(cf.get("super-param", "epoch"))):
    mean_loss = []
    for tx, tx_len, ty, ty_len in train_loader:  # (batch_size, length_per_sequence, feature_per_words)
        rnn.train()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()

        # with torch.no_grad():
        #     weights = np.tanh(np.arange(ty_len) * (np.e / ty_len))
        #     weights = torch.tensor(weights, dtype=torch.float32, device=device)

        tx = rnn_utils.pack_padded_sequence(tx, tx_len, batch_first=True)
        output = rnn(tx)

        # 直接对输出结果计算MSE损失
        # loss = loss_func(output, ty)
        # 分开计算输出结果计算MSE损失
        # lamda = float(cf.get("super-param", "lamda"))
        # loss1 = loss_func(output[:, :2], ty[:, :2])
        # loss2 = loss_func(output[:, -2:], ty[:, -2:])
        # loss = loss1 + lamda * loss2
        # mask掉多余的padding再计算MSE损失
        loss = 0
        for i, y in enumerate(ty):
            loss += loss_func(output[i][:ty_len[i]], y[:ty_len[i]])
        loss = loss/ty.shape[0]
        # 根据预测序列所用到的长短调整loss
        # loss = (output - ty) ** 2 * weights
        # loss = loss.mean()

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
        mean_loss.append(loss.cpu().item())

    with torch.no_grad():
        rnn.eval()
        tx, tx_len, ty, ty_len = iter(test_loader).next()
        if torch.cuda.is_available():
            tx = tx.float().cuda()
            ty = ty.float().cuda()

        tx = rnn_utils.pack_padded_sequence(tx, tx_len, batch_first=True)
        output = rnn(tx)
        loss = loss_func(output, ty)
        print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item())
        sum = 0
        for item in mean_loss:
            sum += item
        m = sum / len(mean_loss)
        print('epoch : %d  ' % step, 'train_loss : %.4f' % m)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(rnn, os.path.join(save_path, 'lstm4pre%d.pkl' % step))
        with open(os.path.join(save_path, 'lstm4pre.log'), 'a+') as loss4log:
            loss4log.write('epoch : %d  train_loss : %.4f\n' % (step, m))
        print('new model saved at epoch {} with val_loss {}'.format(step, m))
