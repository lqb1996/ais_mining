import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
import math


# training at gpu 1-05080043
class LSTM4PRE(nn.Module):
    def __init__(self, input_size=30, hidden_size=30*2):
        super(LSTM4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Linear(hidden_size, 6)
        )

    def forward(self, x, x_len):
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out = self.out(out_pad)
        return out, h_n


# training at gpu 4-05080108
class LSTMlight4PRE(nn.Module):
    def __init__(self, input_size=7, hidden_size=7*67, num_layers=2):
        super(LSTMlight4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.ELU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, x_len):
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out = self.out(out_pad)
        return out


class LSTM_Attention4PRE(nn.Module):
    def __init__(self, input_size=7, hidden_size=7*64, num_layers=2):
        super(LSTM_Attention4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

        self.attention = SelfAttention(hidden_size*2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out_pad, _ = self.attention(out_pad)
        out = self.out(out_pad)
        return out


class ResLSTM_Attention4PRE(nn.Module):
    def __init__(self):
        super(ResLSTM_Attention4PRE, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=6,
            hidden_size=168,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            # dropout=0.4
        )
        self.lstm2 = nn.LSTM(
            input_size=7,
            hidden_size=168,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            # dropout=0.4
        )

        self.norm1 = nn.LayerNorm(13)
        self.norm2 = nn.LayerNorm(336)

        self.attention = SelfAttention(336)

        self.out = nn.Sequential(
            nn.Linear(336, 2)
        )

    def forward(self, x, x_len):
        x = self.norm1(x)
        x1 = x[:, :, :6]
        x1 = rnn_utils.pack_padded_sequence(x1, x_len, batch_first=True)
        x2 = x[:, :, 6:]
        x2 = rnn_utils.pack_padded_sequence(x2, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x1, None)  # None 表示 hidden state 会用全 0 的 state
        r_out2, (h_n, h_c) = self.lstm2(x2, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        r_out2, out_len = rnn_utils.pad_packed_sequence(r_out2, batch_first=True)
        r_out1 = r_out1 + r_out2
        out_pad = self.norm2(r_out1)
        out_pad, _ = self.attention(out_pad)
        out = self.out(out_pad)
        return out


class EncoderBlocks(nn.Module):
    def __init__(self, hidden_dim, multi_head=30, dropout=0.2):
        super(EncoderBlocks, self).__init__()
        self.attention = MultiHeadedAttention(h=multi_head, d_model=hidden_dim, dropout=dropout)
        self.LN1 = nn.LayerNorm(hidden_dim)
        self.fn1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.elu = nn.ELU()
        self.fn2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.LN2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        out = self.attention(x, x, x, mask=mask)
        cat_out = self.LN1(out+x)
        out = self.fn1(cat_out)
        # out = self.dropout1(self.elu(out))
        # out = self.dropout2(self.fn2(out))
        out = self.elu(out)
        out = self.fn2(out)
        out = self.LN2(out+cat_out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        # outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        return outputs, weights


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

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class LSTMTrans4PRE(nn.Module):
    def __init__(self, input_size=7, num_hidden_encoder_layers=2, hidden_size=224):
        super(LSTMTrans4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size*2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = r_out1
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad+r_out1, mask)
        feature = out_pad+r_out1
        out = self.out(feature)
        return out, feature


class Res_LSTMTrans4PRE(nn.Module):
    def __init__(self, input_size=7, num_hidden_encoder_layers=2, hidden_size=224):
        super(Res_LSTMTrans4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size*2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = r_out1
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask)+r_out1
        feature = out_pad
        out = self.out(feature)
        return out, feature


class KalmanTrans4PRE(nn.Module):
    def __init__(self, input_size=11, num_hidden_encoder_layers=2, hidden_size=352, dropout=0):
        super(KalmanTrans4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size * 2, dropout=dropout) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = r_out1
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask) + r_out1
        feature = out_pad
        out = self.out(feature)
        return out, feature


class KalmanTrans4PRE_large(nn.Module):
    def __init__(self, input_size=15, num_hidden_encoder_layers=2, hidden_size=15*32):
        super(KalmanTrans4PRE_large, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size * 2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ELU(),
            nn.Linear(128, 6)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2).unsqueeze(2).repeat(1, 1, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = r_out1
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask) + r_out1
        feature = out_pad
        out = self.out(feature)
        return out, feature


class KalmanTrans4PRE_cat(nn.Module):
    def __init__(self, input_size=15, num_hidden_encoder_layers=2, hidden_size=15 * 32, multi_head=24, dropout=0.4):
        super(KalmanTrans4PRE_cat, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.multi_head = multi_head
        self.a_block = EncoderBlocks(input_size+hidden_size*2, multi_head=multi_head, dropout=dropout)
        self.linear = nn.Linear(input_size+hidden_size*2, hidden_size*2)
        self.transformer_blocks_lstm = nn.ModuleList(
            [EncoderBlocks(hidden_size * 2, multi_head=multi_head, dropout=dropout) for _ in range(num_hidden_encoder_layers)])
        self.transformer_blocks_input = nn.ModuleList(
            [EncoderBlocks(input_size, multi_head=multi_head, dropout=dropout) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2)[:, :1, :].unsqueeze(2).repeat(1, self.multi_head, x.size(1), 1)
        r_out1 = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(r_out1, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        r_out2 = r_out1
        x_encode = x
        for trans_lstm in self.transformer_blocks_lstm:
            r_out2 = trans_lstm.forward(r_out2, mask) + r_out1
        for trans_in in self.transformer_blocks_input:
            x_encode = trans_in.forward(x_encode, mask) + x
        feature = torch.cat((r_out2, x_encode), 2)
        feature = self.linear(self.a_block(feature, mask))
        out = self.out(feature)
        return out, feature


class KalmanTrans4PRE_add(nn.Module):
    def __init__(self, input_size=30, num_hidden_encoder_layers=12, hidden_size=4*30, multi_head=30, dropout=0.15):
        super(KalmanTrans4PRE_add, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.multi_head = multi_head
        self.a_block = EncoderBlocks(input_size, multi_head=multi_head, dropout=dropout)
        self.linear = nn.Linear(input_size, hidden_size*2)
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size * 2, multi_head=multi_head, dropout=dropout) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 6)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2)[:, :1, :].unsqueeze(2).repeat(1, self.multi_head, x.size(1), 1)
        r_out1 = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(r_out1, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        x_encode = self.a_block(x, mask=mask)
        x_encode = self.linear(x_encode)
        out_pad = r_out1 + x_encode
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask) + x_encode
        feature = out_pad
        out = self.out(feature)
        return out, feature


class KalmanTrans4PRE_light(nn.Module):
    def __init__(self, input_size=15, num_hidden_encoder_layers=2, hidden_size=15*32, multi_head=24):
        super(KalmanTrans4PRE_light, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.multi_head = multi_head
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size * 2, multi_head=multi_head) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        mask = (x > 0).transpose(1, 2)[:, :1, :].unsqueeze(2).repeat(1, self.multi_head, x.size(1), 1)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = r_out1
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad, mask) + r_out1
        feature = out_pad
        out = self.out(feature)
        return out, feature
