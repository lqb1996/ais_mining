import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import configparser
import time
import datetime
from module.lstm_base import *

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
csv_loader = CSVDataLoader(csv_path)
# plot(csv_loader.df_total['longitude'], csv_loader.df_total['latitude'])
# print(csv_loader.df_total)


rnn = LSTM()

if torch.cuda.is_available():
    rnn = rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=float(cf.get("super-param", "lr")))  # optimize all cnn parameters
loss_func = nn.MSELoss()

best_loss = 1000


if not os.path.exists('weights'):
    os.mkdir('weights')

for step in range(int(cf.get("super-param", "lr"))):
    for tx, ty in train_loader:
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()

        output = rnn(torch.unsqueeze(tx, dim=2))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()

        print('epoch : %d  ' % step, 'train_loss : %.4f' % loss.cpu().item())

    with torch.no_grad():
        for tx, ty in val_loader:
            if torch.cuda.is_available():
                tx = tx.cuda()
                ty = ty.cuda()

            output = rnn(torch.unsqueeze(tx, dim=2))
            loss = loss_func(torch.squeeze(output), ty)

            print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item())

        if loss.cpu().item() < best_loss:
            best_loss = loss.cpu().item()
            torch.save(rnn, 'weights/rnn.pkl'.format(loss.cpu().item()))
            print('new model saved at epoch {} with val_loss {}'.format(step, best_loss))
