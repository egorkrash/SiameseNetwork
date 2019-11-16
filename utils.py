from __future__ import print_function
import re
import pickle as pkl
import hashlib
import pymorphy2
import numpy as np
import multiprocessing
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import string

# morph analyzer for text lemmatization
morph = pymorphy2.MorphAnalyzer()


# function for performing parallel computing on cpu
def parallelization(func, massive, jobs=None, tq=True):
    num_cores = multiprocessing.cpu_count() if jobs is None else jobs
    if tq:
        results = np.array(Parallel(n_jobs=num_cores)(delayed(func)(i) for i in tqdm(massive)))
        return results
    else:
        results = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in massive)
        return results


def _word2canonical4w2v(word):
    elems = morph.parse(word)
    my_tag = ''
    res = []
    for elem in elems:
        if 'VERB' in elem.tag or 'GRND' in elem.tag or 'INFN' in elem.tag:
            my_tag = 'V'
        if 'NOUN' in elem.tag:
            my_tag = 'S'
        normalised = elem.normalized.word
        res.append((normalised, my_tag))
    tmp = list(filter(lambda x: x[1] != '', res))
    if len(tmp) > 0:
        return tmp[0]
    else:
        return res[0]


def word2canonical(word):
    return _word2canonical4w2v(word)[0]


def get_words(text, filter_short_words=False):
    if filter_short_words:
        return filter(lambda x: len(x) > 3, re.findall(r'(?u)\w+', text))
    else:
        return re.findall(r'(?u)\w+', text)


def text2canonicals(text, add_word=False, filter_short_words=True):
    words = []
    for word in get_words(text, filter_short_words=filter_short_words):
        words.append(word2canonical(word.lower()))
        if add_word:
            words.append(word.lower())
    return words


def remove_punct(s, translator):
    return s.translate(translator)


def preprocess_keywords(arr, lemmatizer, stop_words):
    """
    function that makes lemmatization and other preprocessings for given keywords

    arr: iterable, list of all keywords
    lemmatizer: pymorphy2 lemmatizer
    stop_words: iterable, list of all stop words

    return: list of all the unique keywords
    """
    # remove number
    pattern = '[0-9]'
    arr = [re.sub(pattern, '', i) for i in arr]

    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    arr = [remove_punct(i, translator) for i in arr]

    # remove stop words
    for stop_word in stop_words:
        for i in range(len(arr)):
            arr[i] = arr[i].replace(stop_word, '')

    # remove odd whitespaces
    arr = [i.lstrip().rstrip() for i in arr]

    for i in tqdm(range(len(arr))):
        lemmatize_words = []
        for word in arr[i].split(' '):
            lemmatize_word = lemmatizer.parse(word)[0].normal_form
            lemmatize_words.append(lemmatize_word)
        arr[i] = ' '.join(lemmatize_words)

    unique_queries, mask = np.unique(arr, return_index=True)
    return unique_queries[1:], mask[1:]


def calculate_intersection(keywords_1, keywords_2):
    """
    function that calculates intersection between keywords_1 and keywords_2
    keywords_1: iterable, list of all the correct keywords
    keywords_2: iterable, list of predicted keywords

    return: float, normalized value of intersection
    """
    intersection = len(set(keywords_1) & set(keywords_2))
    normalizer = len(set(keywords_1))

    return intersection / normalizer


def build_weight_matrix(word2vec, target_vocab, emb_dim=300):
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = word2vec.get_vector(word)  # word2vec[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    return weights_matrix


def get_vocab(texts):
    return Counter([word for text in texts for word in text]).keys()


def get_queries_vocab(queries):
    return Counter([word for qs in queries for query in qs for word in query]).keys()


def text_to_idx(text, word_idx):
    return list(map(lambda x: word_idx.get(x) if word_idx.get(x) is not None else len(word_idx) + 1, text))


def filter_zero_length(data, name='train', train=True, verbose=True):
    i = 0
    cnt = 0
    length = len(data)
    while i < len(data):
        if train:
            t, q, _q = data[i]
            if len(q) == 0 or len(_q) == 0:
                data.pop(i)
                cnt += 1
                i -= 1
        else:
            t, q = data[i]
            if len(q) == 0:
                data.pop(i)
                cnt += 1
                i -= 1
        i += 1

    if verbose:
        print('{}/{} zero length samples found in {} set'.format(cnt, length, name))
    return data


def make_dataset(texts, queries, nb_train_samples=None, num_neg_samples=5):
    # construct a dataset in a format of (context, query_positive, query_negative)
    # assuming texts[i] maps to queries[i]
    assert len(texts) == len(queries)
    train_data = []
    q_space = [q for subspace in queries for q in subspace]
    # we have len(q_space) queries at all
    # let's just sample all negatives in one run
    negatives = np.random.choice(q_space, len(q_space) * num_neg_samples)
    # now write it all into train data
    k = 0
    for i in tqdm(range(len(texts))):
        for j in range(len(queries[i])):
            # distribute negatives
            z = 0
            while k < len(negatives) and z < num_neg_samples:
                # append our triplet
                train_data.append([texts[i], queries[i][j], negatives[k]])
                k += 1
                z += 1

    # if number of train samples is not specified then return whole preprocessed dataset
    if nb_train_samples is not None:
        return train_data[:nb_train_samples]
    return train_data


def sample_negatives(neg_space, n_samples):
    # TODO: probabilities depending on length or frequency (f_i^(3/4) / sum(f_i^(3/4)))
    return np.random.choice(neg_space, n_samples)


def make_pairs4prediction(text, queries):
    pairs = []
    for query in queries:
        pairs.append((text, query))
    return pairs


def checksum_match():
    # calculate hash for current bank
    hasher = hashlib.md5()
    with open('bank_queries.txt', 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    md5hash = hasher.hexdigest()
    # compare that hash with previous
    with open('data/queries_md5hash', 'r', encoding='utf-8') as f:
        if md5hash != f.readline():
            # bank of queries is changed, we need to preprocess it again
            return False, md5hash
        return True, None


def update_queries(query_enc=None, device=None):
    print('Bank of queries has changed, preprocessing...')
    with open('bank_queries.txt', 'r', encoding='utf-8') as f:
        # load all queries in memory
        queries = list(map(lambda x: x.strip('\n'), f.readlines()))

    # preprocess queries
    stop_words = ['бесплатно', 'скачать', 'на русском', 'онлайн', 'русский']
    queries_preprocessed, mask = preprocess_keywords(queries, morph, stop_words)
    # save them
    np.save('data/all_keywords_keys.npy', queries_preprocessed, allow_pickle=True)
    pkl.dump(mask, open('data/mask_queries.pkl', 'wb'))
    if query_enc is not None:
        assert device is not None
        calculate_queries_encodings(query_enc, device, queries_preprocessed)


def calculate_queries_encodings(model, device, queries, batch_size=256):
    _, query_word_idx = pkl.load(open('data/word_idx_dicts.pkl', 'rb'))
    all_queries = list(map(lambda x: text_to_idx(x.split(), query_word_idx), queries))

    queries_encodings = []
    for start_idx in range(0, len(all_queries) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = all_queries[excerpt]
        queries_tensor = torch.tensor(pad_sequence(batch), dtype=torch.long, device=device)
        queries_len = torch.tensor(list(map(len, queries_tensor)), dtype=torch.int32, device=device)
        queries_repr = model(queries_tensor, queries_len)
        queries_repr = queries_repr.detach().cpu()
        queries_encodings.append(queries_repr)
    queries_encodings = np.array([x for btch in queries_encodings for x in btch])
    # save tensors for queries
    np.save('data/queries_encodings.npy', queries_encodings, allow_pickle=True)


def pad_sequence(array):
    # array is array of arrays
    maxlen = np.max(list(map(len, array)))
    matrix = np.zeros((len(array), maxlen))
    for i, subarr in enumerate(array):
        # post padding
        subarr = np.pad(subarr, (0, maxlen - len(subarr)), 'constant')
        matrix[i] = subarr
    return matrix


def iterate_encoding_minibatches(inputs, batchsize, device):
    # inputs are pairs (vector of words indexes, query tensor)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        batch = inputs[excerpt]
        context, query_repr = zip(*batch)
        query_repr = torch.stack(query_repr)
        query_repr = query_repr.to(device)
        clen = torch.tensor(list(map(len, context)), dtype=torch.int32, device=device)
        context = torch.tensor(pad_sequence(context), dtype=torch.long, device=device)
        yield context, clen, query_repr


def iterate_minibatches(inputs, batchsize, device, shuffle=False, train=True):
    vector_dict = pkl.load(open('data/vectors_dict.pkl', 'rb'))
    if shuffle:
        # shuffle indices
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            batch = [inputs[x] for x in excerpt]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            batch = inputs[excerpt]

        if train:
            context, q_pos, q_neg = zip(*batch)
        else:
            context, q_pos = zip(*batch)
        # calculate lengths which will be used in rnn
        context = np.array([vector_dict[x] for x in context])
        qposlen = torch.tensor(list(map(len, q_pos)), dtype=torch.int32, device=device)
        context = torch.tensor(context, dtype=torch.float32, device=device)
        q_pos = torch.tensor(pad_sequence(q_pos), dtype=torch.long, device=device)

        if train:
            qneglen = torch.tensor(list(map(len, q_neg)), dtype=torch.int32, device=device)
            q_neg = torch.tensor(pad_sequence(q_neg), dtype=torch.long, device=device)
            yield context, q_pos, qposlen, q_neg, qneglen
        else:
            yield context, q_pos, qposlen


def save_model(state_dict, path):
    torch.save(state_dict, path)


def load_model(model, path, device):
    checkpoint = torch.load(path,  map_location=device)
    model.load_state_dict(checkpoint)
    return model
