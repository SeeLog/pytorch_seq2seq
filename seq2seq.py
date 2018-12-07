import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from my_utils import device, MAX_LENGTH, __PAD_ID__


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=__PAD_ID__)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True)
        self.linear_h = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_batch, input_lens):
        batch_size = input_batch.shape[1]

        embedded = self.embedding(input_batch)  # (s, b) -> (s, b, h)
        output, (hidden_h, hidden_c) = self.lstm(embedded)

        hidden_h = hidden_h.transpose(1, 0)     # (2, b, h) -> (b, 2, h)
        hidden_h = hidden_h.reshape(batch_size, -1) # (b, 2, h) -> (b, 2h)
        hidden_h = F.dropout(hidden_h, p=0.5, training=self.training)
        hidden_h = self.linear_h(hidden_h)      # (b, 2h) -> (b, h)
        hidden_h = F.relu(hidden_h)
        hidden_h = hidden_h.unsqueeze(0)        # (b, h) -> (1, b, h)

        hidden_c = hidden_c.transpose(1, 0)
        hidden_c = hidden_c.reshape(batch_size, -1) # (b, 2, h) -> (b, 2h)
        hidden_c = F.dropout(hidden_c, p=0.5, training=self.training)
        hidden_c = self.linear_c(hidden_c)
        hidden_c = F.relu(hidden_c)
        hidden_c = hidden_c.unsqueeze(0)

        return output, (hidden_h, hidden_c)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN1(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, max_length=MAX_LENGTH):
        super(AttnDecoderRNN1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attn = nn.Linear(embedding_size + 2 * hidden_size, max_length)
        self.attn_combine = nn.Linear(embedding_size + 2 * hidden_size, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param input: (b)
        :param hidden: ((b, h), (b, h))
        :param encoder_outputs: (il, b, 2h)
        :return: (b, o), ((b, h), (b, h)), (b, il)
        """

        input_length = encoder_outputs.shape[0]
        # padding
        #print("input", input_length)
        #print("PAD", self.max_length - input_length)
        encoder_outputs = torch.cat([
            encoder_outputs,
            torch.zeros(
                self.max_length - input_length,
                encoder_outputs.shape[1],
                encoder_outputs.shape[2],
                device=device
            )
        ], dim=0)   # (il, b, 2h), (ml-il, b, 2h) -> (ml, b, 2h)

        #print("encoder_outputs", encoder_outputs.shape)

        drop_encoder_outputs = F.dropout(encoder_outputs, p=0.1, training=self.training)

        #print("drop_encoder_outputs", drop_encoder_outputs.transpose(0, 1).shape)

        # embedding
        embedded = self.embedding(input)    # (b) -> (b, e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)

        emb_hidden = torch.cat([embedded, hidden[0], hidden[1]], dim=1)     # (b, e), ((b, h), (b, h)) -> (b, e+2h)

        attn_weights = self.attn(emb_hidden)    # (b, e+2h) -> (b, ml)
        attn_weights = F.softmax(attn_weights, dim=1)

        #print("attn_weigths", attn_weights.unsqueeze(1).shape)

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),              # (b, 1, ml)
            drop_encoder_outputs.transpose(0, 1)    # (b, ml, 2h)
        )

        attn_applied = F.dropout(attn_applied, p=0.1, training=self.training)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)      # ((b, e), (b, 2h)) -> (b, e+2h)
        output = self.attn_combine(output)      # (b, e+2h) -> (b, h)
        output = F.dropout(output, p=0.5, training=self.training)

        output = F.relu(output)
        hidden = self.lstm(output, hidden)      # (b, h), ((b, h), (b, h)) -> (b, h)((b, h), (b, h))

        output = F.log_softmax(self.out(hidden[0]), dim=1)  # (b, h) -> (b, o)

        return output, hidden, attn_weights     # (b, o), (b, h), (b, il)

class AttnDecoderRNN2(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_size, output_size, pad_token=0):
        super(AttnDecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTMCell(emb_size, hidden_size)

        self.score_w = nn.Linear(2*hidden_size, 2*hidden_size)
        self.attn_w = nn.Linear(4*hidden_size, attn_size)
        self.out_w = nn.Linear(attn_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param: input: (b)
        :param: hidden: ((b,h),(b,h))
        :param: encoder_outputs: (il,b,2h)

        :return: (b,o), ((b,h),(b,h)), (b,il)
        """

        embedded = self.embedding(input)  # (b) -> (b,e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)

        hidden = self.lstm(embedded, hidden)  # (b,e),((b,h),(b,h)) -> ((b,h),(b,h))
        decoder_output = torch.cat(hidden, dim=1)  # ((b,h),(b,h)) -> (b,2h)
        decoder_output = F.dropout(decoder_output, p=0.5, training=self.training)

        # score
        score = self.score_w(decoder_output)  # (b,2h) -> (b,2h)
        scores = torch.bmm(
            encoder_outputs.transpose(0, 1),  # (b,il,2h)
            score.unsqueeze(2)  # (b,2h,1)
        )  # (b,il,1)
        attn_weights = F.softmax(scores, dim=1)  # (b,il,1)

        # context
        context = torch.bmm(
            attn_weights.transpose(1, 2),  # (b,1,il)
            encoder_outputs.transpose(0, 1)  # (b,il,2h)
        )  # (b,1,2h)
        context = context.squeeze(1)  # (b,1,2h) -> (b,2h)

        concat = torch.cat((context, decoder_output), dim=1)  # ((b,2h),(b,2h)) -> (b,4h)
        #concat = F.dropout(concat, p=0.5, training=self.training)

        attentional = self.attn_w(concat)  # (b,4h) -> (b,a)
        attentional = F.tanh(attentional)
        #attentional = F.dropout(attentional, p=0.5, training=self.training)

        output = self.out_w(attentional)  # (b,a) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights.squeeze(2)  # (b,o), ((b,h),(b,h)), (b,il)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding: nn.Embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn: nn.Linear = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


#encoder = EncoderRNN(256, 256)