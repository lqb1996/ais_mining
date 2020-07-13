import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
import math


class BERT_LSTM4PRE_featureAT(nn.Module):
    def __init__(self, input_size=24, num_hidden_encoder_layers=12, hidden_size=24, dropout=0.2, multi_head=8):
        super(BERT_LSTM4PRE_featureAT, self).__init__()
        hidden_size = hidden_size if hidden_size is not None else input_size*4
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.multi_head = multi_head

        self.attention = Attention()

        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(input_size, multi_head=multi_head) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_len):
        # x.shape=(batch, len, feature)
        # mask.shape=(batch, feature, len, len)
        # f_mask.shape=(batch, len, feature, feature)
        mask = (x > 0).transpose(1, 2)[:, :1, :].unsqueeze(2).repeat(1, self.multi_head, x.size(1), 1)
        f_mask = (x > 0)[:, :1, :].unsqueeze(2).repeat(1, self.multi_head, x.size(2), 1)
        print(x.shape)
        out_pad = x.transpose(1, 2)
        out_pad, attn = self.attention(out_pad, out_pad, out_pad, f_mask)
        out_pad = out_pad.transpose(1, 2)
        print(out_pad.shape)
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask=mask) + x
        feature = rnn_utils.pack_padded_sequence(out_pad, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm(feature, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out = self.out(r_out1)
        return out, feature


class Comfort_BERT_LSTM4PRE(nn.Module):
    def __init__(self, input_size=7, num_hidden_encoder_layers=2, hidden_size=224, dropout=0.2):
        super(Comfort_BERT_LSTM4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = self.norm2(r_out1)
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask=mask)
        feature = out_pad+x
        feature = rnn_utils.pack_padded_sequence(feature, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm2(feature, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out = self.dropout(self.out(r_out1))
        return out, feature


class Simple_BERT_LSTM4PRE(nn.Module):
    def __init__(self, input_size=24, num_hidden_encoder_layers=12, hidden_size=12, dropout=0.2):
        super(Simple_BERT_LSTM4PRE, self).__init__()
        hidden_size = hidden_size if hidden_size is not None else input_size*4
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(input_size) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        out_pad = x
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask=mask) + x
        feature = rnn_utils.pack_padded_sequence(out_pad, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm(feature, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        # out = self.dropout(self.out(r_out1))
        out = self.out(r_out1)
        return out, feature


class EncoderBlocks(nn.Module):
    def __init__(self, hidden_dim, multi_head=24):
        super(EncoderBlocks, self).__init__()
        self.attention = MultiHeadedAttention(h=multi_head, d_model=hidden_dim)
        self.LN1 = nn.LayerNorm(hidden_dim)
        self.fn1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.elu = nn.ELU()
        self.fn2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.LN2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        out = self.attention(x, x, x, mask=mask)
        cat_out = self.LN1(out)+x
        out = self.fn1(cat_out)
        out = self.elu(out)
        out = self.fn2(out)
        out = self.LN2(out)+cat_out
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
