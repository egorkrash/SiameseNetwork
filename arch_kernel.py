import torch
from torch import nn


class KernelSiameseNetwork(nn.Module):
    """
    New (second) version of Siamese network which supports semantic kernels
    """
    def __init__(self, context_encoder, context_dim, query_dim):
        super(KernelSiameseNetwork, self).__init__()
        self.context_encoder = context_encoder
        # siamese network layers
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(context_dim + query_dim, 1024)
        self.linear_2 = nn.Linear(1024, 128)
        self.linear_3 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, context, query_pos_repr, query_neg_repr=None, train=True):
        # take both queries while training and only one while testing to assign a score
        # (second input just ignored if train=False)
        context_repr = self.context_encoder(context)
        siamese_inp_pos = torch.cat([query_pos_repr, context_repr], dim=-1)
        score_pos = self.linear_3(self.relu(self.linear_2(self.relu(self.dropout(self.linear_1(siamese_inp_pos))))))
        # in testing phase model takes only text and query
        # (to assign a score "how good is the given query in the context of given text")
        if train:
            assert query_neg_repr is not None, "you have to provide a second input"
            siamese_inp_neg = torch.cat([query_neg_repr, context_repr], dim=-1)
            score_neg = self.linear_3(self.relu(self.linear_2(self.relu(self.dropout(self.linear_1(siamese_inp_neg))))))
            return score_pos - score_neg
        else:
            return score_pos


class QueryEncoder(nn.Module):
    def __init__(self, emb_matrix, hidden_size=64):
        super(QueryEncoder, self).__init__()

        self.embedding, num_embeddings, embedding_dim = self.create_emb_layer(emb_matrix, True)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=1,
                          bidirectional=True, batch_first=True)

    @staticmethod
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def forward(self, inputs, lengths):
        # X = app vector
        embedded = self.embedding(inputs)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                                           batch_first=True, enforce_sorted=False)
        output, hn = self.gru(embedded)
        output = torch.cat([hn[0], hn[1]], dim=-1)
        return output


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=1864, hidden_size=64):
        super(ContextEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, hidden_size * 2)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, inputs):
        return self.leaky_relu(self.linear2(self.dropout(self.leaky_relu(self.linear1(inputs)))))
