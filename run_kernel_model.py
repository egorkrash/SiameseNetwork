from __future__ import print_function
from arch_kernel import *
from utils import *
import argparse
import numpy as np
import warnings
import pickle as pkl
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F


# define our loss
def cost_function(x):
    return torch.mean(-F.logsigmoid(x))  # negative log likelihood (pairwised loss)


def train(net, query_enc, train_data, test_data, device, args):
    optimizer = optim.Adam(list(net.parameters()) + list(query_enc.parameters()), lr=args.lr)  # weight_decay=0.0005)
    print('Start training...')
    # set models in training mode
    net.train()
    query_enc.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        i = 0
        # initialize data generator
        generator = iterate_minibatches(train_data, args.batch_size, device=device, shuffle=args.shuffle,)
        for sample in generator:
            # parse the sample
            context, clen, q_pos, qposlen, q_neg, qneglen = sample
            # zero gradients
            optimizer.zero_grad()
            query_pos_repr = query_enc(q_pos, qposlen)
            query_neg_repr = query_enc(q_neg, qneglen)
            # get predictions (difference in scores)
            outputs = net(context, clen, query_pos_repr, query_neg_repr)
            # calculate loss
            loss = cost_function(outputs)
            # backpropagate
            loss.backward()
            # make a step towards anti-gradient
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.log_interval == args.log_interval - 1:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0

            if i % args.val_interval == args.val_interval - 1:
                test(net, test_data, args.test_batch_size, device, args.shuffle)
                net.train()
            i += 1

        # save model each args.save_each epochs
        if args.save_model and epoch % args.save_each == args.save_each - 1:
            path = 'weights/new_params_wval_{}.pt'.format(epoch)
            save_model(net.state_dict(), path)
            print('Model saved in {}'.format(path))

    print('Finished Training')


def test(net, query_enc, test_data, batch_size, device, shuffle=False, portion=None, use_query_encodings=True):
    if portion is None:
        portion = batch_size * 200
    # make a random sample from test data of size portion
    assert portion <= len(test_data)
    inds = np.arange(len(test_data))
    np.random.shuffle(inds)
    test_data = [test_data[x] for x in inds[:portion]]
    print('Testing on {} samples'.format(len(test_data)))
    generator = iterate_minibatches(test_data, batch_size, device, shuffle=shuffle,
                                    use_query_encodings=use_query_encodings)
    cnt = 0
    running_loss = 0
    # set model in evaluation mode
    net.eval()
    query_enc.eval()
    with torch.no_grad():
        for sample in generator:
            if use_query_encodings:
                context, clen, query_pos_repr, query_neg_repr = sample
                outputs = net(context, clen, query_pos_repr, query_neg_repr)
            else:
                # parse the sample
                context, clen, q_pos, qposlen, q_neg, qneglen = sample
                # get predictions
                query_pos_repr = query_enc(q_pos, qposlen)
                query_neg_repr = query_enc(q_neg, qneglen)
                outputs = net(context, clen, query_pos_repr, query_neg_repr)
            # calculate loss
            loss = cost_function(outputs)
            running_loss += loss.item()
            cnt += 1
    print('Test loss %.3f' % (running_loss / cnt))


def predict(net, description, device, batch_size=512, top_n=300):
    context_word_idx, query_word_idx = pkl.load(open('data/word_idx_dicts.pkl', 'rb'))
    # preprocess description
    text = text2canonicals(description)
    # check that bank of queries is not changed
    match, md5hash = checksum_match()

    if not match:
        # update data/all_keywords_keys.npy
        update_queries()
        # rewrite new hash
        with open('data/queries_md5hash', 'w', encoding='utf-8') as f:
            f.write(md5hash)

    # load preprocessed bank of queries
    all_queries = np.load('data/all_keywords_keys.npy')  # TODO: put here matching tensors
    # convert to indices
    text = text_to_idx(text, context_word_idx)
    all_queries = list(map(lambda x: text_to_idx(x.split(), query_word_idx), all_queries))
    # make pairs (text, query) to feed to the network
    data4prediction = make_pairs4prediction(text, all_queries)
    # filter zero length samples
    data4prediction = filter_zero_length(data4prediction, name='eval', train=False, verbose=False)
    # initialize generator
    generator = iterate_minibatches(data4prediction, batch_size,
                                    device, shuffle=False, train=False)
    predictions = []
    # set model in evaluation mode
    net.eval()
    # make predictions
    print('Making predictions...')
    for sample in generator:
        # unpack sample
        context, clen, q_cand, q_candlen = sample
        outputs = net(context, clen, q_cand, q_candlen, train=False)
        outputs = list(map(lambda x: x[0], outputs.tolist()))
        predictions.extend(outputs)

    # sort them
    sorted_predictions = sorted(zip(predictions, range(len(predictions))),
                                reverse=True)
    q4predictions = all_queries[:len(predictions)]
    inds = list(map(lambda x: x[1], sorted_predictions))
    # select top N predictions and return
    qselected = [q4predictions[x] for x in inds]
    query_idx_word = dict(zip(query_word_idx.values(), query_word_idx.keys()))
    return list(map(lambda x: ' '.join(list(map(lambda y: query_idx_word.get(y), x))), qselected))[:top_n]


def main():
    # training settings
    parser = argparse.ArgumentParser(description='Siamese Network')

    parser.add_argument('--save-model', default=True,
                        help='for saving the current model each "save-each" epochs')

    parser.add_argument('--save-each', type=int, default=5, metavar='N',
                        help='number of epochs to wait before making checkpoint (default 5)')

    parser.add_argument('--load-model', action='store_true', default=False,
                        help='whether to load checkpoint of a trained model or not')

    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='path to state dict (checkpoint) for model (is needed only if --load-model is True)')

    parser.add_argument('--make-prediction', action='store_true', default=False,
                        help='make predictions using data from predicton-data-path')

    parser.add_argument('--prediction-data-path', type=str, default='./testdesc.txt',
                        help='path to data for predicting queries')

    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate on the full test set')

    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')

    parser.add_argument('--shuffle', default=True,
                        help='shuffle batches while training (default True)')

    parser.add_argument('--nneg-samples', type=int, default=10,
                        help='number of negative samples for one positive (default 10)')

    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='index of gpu to use (default 0)')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status (default 200)')

    parser.add_argument('--val-interval', type=int, default=5000, metavar='N',
                        help='how many batches to wait before performing validation (default 5000)')

    args = parser.parse_args()
    is_trained = False  # becomes true when model is trained or loaded
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # load embedding matrices
    context_emb_matrix = np.load('data/context_emb_matr.npy')
    query_emb_matrix = np.load('data/query_emb_matr.npy')
    # send them to gpu
    context_emb_matrix = torch.tensor(context_emb_matrix, dtype=torch.float32, device=device)
    query_emb_matrix = torch.tensor(query_emb_matrix, dtype=torch.float32, device=device)

    # initialize encoders for texts and queries
    context_enc = QueryEncoder(context_emb_matrix)
    # initialize network
    query_enc = QueryEncoder(query_emb_matrix)
    net = KernelSiameseNetwork(context_enc, 128, 128)
    # load from checkpoint
    if args.load_model:
        print('Loading model...')
        net = load_model(net, args.checkpoint_path, device)
        is_trained = True

    # send it to gpu
    query_enc.to(device)
    net.to(device)

    if args.train or args.eval:
        # load preprocessed texts and queries
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        print('Loading data and forming a training set...')
        # call load_data with allow_pickle implicitly set to true
        texts = np.load('data/all_descriptions_keys.npy')
        queries = np.load('data/matched_keywords.npy')
        queries = list(map(lambda x: list(map(lambda y: y.split(), x)), queries))
        # load dicts for mapping words to indices
        context_word_idx, query_word_idx = pkl.load(open('data/word_idx_dicts.pkl', 'rb'))
        # map to indices
        samples = list(map(lambda x: text_to_idx(x, context_word_idx), texts))
        queries = list(map(lambda x: list(map(lambda y: text_to_idx(y, query_word_idx), x)), queries))
        # split to train and test sets of apps
        train_samples, test_samples, train_queries, test_queries = train_test_split(samples, queries,
                                                                                    test_size=0.05,
                                                                                    random_state=args.seed)
        # construct triplets (text, positive query, negative query)
        train_data = make_dataset(train_samples, train_queries, num_neg_samples=args.nneg_samples)
        test_data = make_dataset(test_samples, test_queries, num_neg_samples=args.nneg_samples)
        # filter samples which have texts or queries of length 0
        train_data = filter_zero_length(train_data, name='train', verbose=False)
        test_data = filter_zero_length(test_data, name='test', verbose=False)
        # shuffle the data sets
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        print('Amount of training samples: {}'.format(len(train_data)))
        print('Amount of testing samples: {}'.format(len(test_data)))

        # train the network
        if args.train:
            train(net, query_enc, train_data, test_data, device, args)
            is_trained = True

        # evaluate
        if args.eval:
            if not is_trained:
                warnings.warn('Model is not trained or loaded')
            test(net, test_data, args.test_batch_size, device, portion=len(test_data))

    if args.make_prediction:
        assert is_trained, 'Model have to be trained or loaded before making predictions'
        with open(args.prediction_data_path, 'r', encoding='utf-8') as f:
            description = f.read().strip()
        predictions = predict(net, description, device)
        with open('testpreds.txt', 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')
        print('Predictions were saved in testpreds.txt')


if __name__ == "__main__":
    main()
