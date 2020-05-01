import torch
import torch.nn as nn

class MyModel(nn.Module):
        def __init__(self, input_size, output_size, embedding_layer):
            super().__init__()

            # Parameters
            self.hidden_dim = HParam.hidden_dim
            self.n_layers = HParam.n_layers

            self.word_embedding = embedding_layer

            # RNN Layer
            self.gru = nn.GRU(HParam.embedding_size, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=HParam.bidirectional)


            self.gru_out_dim = self.hidden_dim*2 if (HParam.bidirectional) else self.hidden_dim

            # Fully connected layer
            self.fc = nn.Linear(self.gru_out_dim, int(self.hidden_dim/2))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(HParam.dropout)


            self.fc2 = nn.Linear(int(self.hidden_dim/2), output_size)

        def forward(self, x):

            batch_size = x.size(0)

            # Hidden state initialization for the first input
            hidden = self.init_hidden(batch_size)

            # Inputs of the model transformed into word embeddings
            embeddings = self.word_embedding(x)

            gru_out, h = self.gru(embeddings, hidden)


            # Reshaping the outputs such that it can be fit into the fully connected layer
            out = gru_out.contiguous().view(-1, self.gru_out_dim)

            out = self.fc(out)
            out = self.relu(out)
            dropout = self.dropout(out)

            out = self.fc2(out)
            dropout = self.dropout(out)
            out = dropout

            return out

        def init_hidden(self, batch_size):
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            # We'll send the tensor holding the hidden state to the device we specified earlier as well

            mult = 2 if (HParam.bidirectional) else 1

            hidden = torch.zeros(self.n_layers*mult, batch_size, self.hidden_dim)
            return hidden

class HParam ():
    window_size = 100
    window_shift = 100
    embedding_size = 300
    batch_size = 100
    hidden_dim = 100
    dropout = 0.15
    n_layers = 1
    lr=4e-4
    n_iter = 3
    bidirectional=True
    freeze=False
