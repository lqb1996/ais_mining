import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
import math


# training at gpu 1-05080043
class LSTM4PRE(nn.Module):
    def __init__(self):
        super(LSTM4PRE, self).__init__()

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.out = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out = self.out(out_pad)
        return out


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


class TransLSTM4PRE(nn.Module):
    def __init__(self, input_size=7, num_hidden_encoder_layers=2, hidden_size=224):
        super(TransLSTM4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        # self.lstm2 = nn.LSTM(
        #     input_size=input_size-6,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     bidirectional=True,
        #     batch_first=True,
        #     dropout=0.4
        # )

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(hidden_size*2)
        # self.attention = SelfAttention(336)
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size*2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_len):
        x = self.norm1(x)
        x = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True)
        r_out1, (h_n, h_c) = self.lstm1(x, None)  # None 表示 hidden state 会用全 0 的 state
        r_out1, out_len = rnn_utils.pad_packed_sequence(r_out1, batch_first=True)
        out_pad = self.norm2(r_out1)
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad+r_out1)
        feature = out_pad+r_out1
        out = self.out(feature)
        return out, feature


class TransLSTM_BN4PRE(nn.Module):
    def __init__(self, input_size=13, num_hidden_encoder_layers=2, hidden_size=168):
        super(TransLSTM_BN4PRE, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.lstm2 = nn.LSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(hidden_size*2)
        # self.attention = SelfAttention(336)
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size*2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
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
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad+r_out1)
        feature = out_pad+r_out1
        out = self.out(feature)
        return out, feature

class EncoderBlocks(nn.Module):
    def __init__(self, hidden_dim):
        super(EncoderBlocks, self).__init__()
        self.attention = SelfAttention(hidden_dim)
        self.LN1 = nn.LayerNorm(hidden_dim)
        self.fn1 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fn2 = nn.Linear(hidden_dim, hidden_dim)
        self.LN2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.attention(x)
        cat_out = self.LN1(out+x)
        out = self.fn1(cat_out)
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

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, dropout=self.dropout)

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


class TransLSTM4CLS(nn.Module):
    def __init__(self, input_size=13, num_hidden_encoder_layers=2, hidden_size=168):
        super(TransLSTM4CLS, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.lstm2 = nn.LSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(hidden_size*2)
        # self.attention = SelfAttention(336)
        self.transformer_blocks = nn.ModuleList(
            [EncoderBlocks(hidden_size*2) for _ in range(num_hidden_encoder_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ELU(),
            nn.Linear(128, 2)
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
        for transformer in self.transformer_blocks:
            out_pad = transformer.forward(out_pad+r_out1)
        out = self.out(out_pad+r_out1)
        return out


class bilstm_attn(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda, attention_size, sequence_length):
        super(bilstm_attn, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        #print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits
