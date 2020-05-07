import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader


class LSTM4PRE(nn.Module):
    def __init__(self):
        super(LSTM4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=2048,
            num_layers=2,
            bidirectional=True,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(4096, 2))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out = self.out(out_pad)
        return out


class LSTMlight4PRE(nn.Module):
    def __init__(self):
        super(LSTMlight4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        print(out_len)
        out = self.out(out_pad)
        return out
