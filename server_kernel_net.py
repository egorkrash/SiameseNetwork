from __future__ import print_function
from arch_kernel import *
from arch_kernel import KernelSiameseNetwork
from utils import *
import numpy as np
import pickle as pkl
import torch


class Model(object):
    def __init__(self, checkpoint_path='./weights/params_0'):
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        # load embedding matrices
        query_emb_matrix = np.load('data/query_emb_matr.npy')
        # send them to gpu
        query_emb_matrix = torch.tensor(query_emb_matrix, dtype=torch.float32, device=self.device)

        # initialize encoders for texts and queries
        context_enc = ContextEncoder()
        # initialize network
        self.net = KernelSiameseNetwork(context_enc, 128, 128)
        self.query_enc = QueryEncoder(query_emb_matrix)
        # load from checkpoint
        self.net, self.query_enc = load_model(self.net, self.query_enc, checkpoint_path, self.device)
        # send it to gpu
        self.net.to(self.device)
        self.query_enc.to(self.device)
        self.net.eval()
        self.query_enc.eval()

        # load dicts
        # self.context_word_idx, query_word_idx = pkl.load(open('data/word_idx_dicts.pkl', 'rb'))
        # self.query_idx_word = dict(zip(query_word_idx.values(), query_word_idx.keys()))

        mask = pkl.load(open('data/mask_queries.pkl', 'rb'))
        with open('bank_queries.txt', 'r', encoding='utf-8') as f:
            # load all queries in memory
            self.bank_queries = list(map(lambda x: x.strip('\n'), f.readlines()))
            self.bank_queries = [self.bank_queries[x] for x in mask]

        # load preprocessed bank of queries
        #self.all_queries = np.load('data/all_keywords_keys.npy', allow_pickle=True)
        self.all_queries = pkl.load(open('data/queries_encodings.pkl', 'rb'))

        assert len(self.all_queries) == len(self.bank_queries)
        assert (np.array(list(map(len, self.all_queries))) == 0).sum() == 0, \
            'each query must contain at least 4 symbols'

    def predict(self, kernel, category, name, top_n=300):
        # preprocess input
        kernel_vector = process_new_input(name, kernel, category)
        # convert to indices
        assert len(kernel) > 0, 'kernel must contain at least 4 symbols'
        assert len(category) > 0
        assert len(name) > 0
        # make pairs (text, query) to feed to the network
        data4prediction = make_pairs4prediction(kernel_vector, self.all_queries)
        # initialize generator
        #generator = iterate_minibatches(data4prediction, 256,
        #                                self.device, shuffle=False, train=False)
        generator = iterate_encoding_minibatches(data4prediction, 256, self.device)
        predictions = []
        # make predictions
        for sample in generator:
            # unpack sample
            #context, query_pos, qposlen = sample
            context, clen, query_repr = sample
            outputs = self.net(context, query_repr, train=False)
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
