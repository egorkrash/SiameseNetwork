from __future__ import print_function
from arch import *
from arch import SiameseNetwork
from utils import *
import numpy as np
import pickle as pkl
import torch


class Model(object):
    def __init__(self, checkpoint_path='./weights/params_wval_9.pt'):
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        # load embedding matrices
        context_emb_matrix = np.load('data/context_emb_matr.npy')
        query_emb_matrix = np.load('data/query_emb_matr.npy')
        # send them to gpu
        context_emb_matrix = torch.tensor(context_emb_matrix, dtype=torch.float32, device=self.device)
        query_emb_matrix = torch.tensor(query_emb_matrix, dtype=torch.float32, device=self.device)

        # initialize encoders for texts and queries
        context_enc = Encoder(context_emb_matrix)
        query_enc = Encoder(query_emb_matrix)
        # initialize network
        self.net = SiameseNetwork(context_enc, query_enc, 128, 128)
        # load from checkpoint
        self.net = load_model(self.net, checkpoint_path, self.device)
        # send it to gpu
        self.net.to(self.device)
        # load dicts
        self.context_word_idx, query_word_idx = pkl.load(open('data/word_idx_dicts.pkl', 'rb'))
        # self.query_idx_word = dict(zip(query_word_idx.values(), query_word_idx.keys()))

        mask = pkl.load(open('data/mask_queries.pkl', 'rb'))
        with open('bank_queries.txt', 'r', encoding='utf-8') as f:
            # load all queries in memory
            self.bank_queries = list(map(lambda x: x.strip('\n'), f.readlines()))
            self.bank_queries = [self.bank_queries[x] for x in mask]

        # load preprocessed bank of queries
        self.all_queries = np.load('data/all_keywords_keys.npy')
        self.all_queries = list(map(lambda x: text_to_idx(x.split(), query_word_idx), self.all_queries))

        assert len(self.all_queries) == len(self.bank_queries)
        assert (np.array(list(map(len, self.all_queries))) == 0).sum() == 0, \
            'each query must contain at least 4 symbols'

    def predict(self, description, top_n=300):
        # preprocess description
        text = text2canonicals(description)
        # convert to indices
        text = text_to_idx(text, self.context_word_idx)
        assert len(text) > 0, 'description must contain at least 4 symbols'
        # make pairs (text, query) to feed to the network
        data4prediction = make_pairs4prediction(text, self.all_queries)
        # initialize generator
        del self.all_queries  # to free some memory
        generator = iterate_minibatches(data4prediction, 256,
                                        self.device, shuffle=False, train=False)
        predictions = []
        # set model in evaluation mode
        self.net.eval()
        # make predictions
        for sample in generator:
            # unpack sample
            context, clen, q_cand, q_candlen = sample
            outputs = self.net(context, clen, q_cand, q_candlen, train=False)
            outputs = list(map(lambda x: x[0], outputs.tolist()))
            predictions.extend(outputs)

        # sort them
        sorted_predictions = sorted(zip(predictions, range(len(predictions))),
                                    reverse=True)

        # q4predictions = self.all_queries[:len(predictions)]
        q4predictions = self.bank_queries[:len(predictions)]
        inds = list(map(lambda x: x[1], sorted_predictions))
        # select top N predictions and return
        qselected = [q4predictions[x] for x in inds]
        # list(map(lambda x: ' '.join(list(map(lambda y: self.query_idx_word.get(y), x))), qselected))[:top_n]
        return qselected[:top_n]
