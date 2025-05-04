import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dimension):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(dimension, 1)

    def forward(self, input_ids, ids_len):
        ids_embedded = self.embedding(input_ids)
        packed_input = pack_padded_sequence(ids_embedded, ids_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), ids_len - 1, :self.dimension]
        ids_dropout = self.dropout(out_forward)
        ids_fc = self.fc(ids_dropout)
        ids_fc = torch.squeeze(ids_fc, 1)
        return torch.sigmoid(ids_fc)
