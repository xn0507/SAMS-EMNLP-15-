import cPickle
import gzip
import os

import numpy
import theano


def prepare_data(seqs, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    The mask is to judge the padding.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask




def load_data(path="ppdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here ppdb)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    f = open(path, 'rb')
    train_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_xp = []
        new_train_set_y = []
        for x, xp, y in zip(train_set[0], train_set[1], train_set[2]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_xp.append(xp)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_xp, new_train_set_y)
        del new_train_set_x, new_train_set_xp, new_train_set_y
    
    # split training set into validation set
    train_set_x, train_set_xp, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_xp = [train_set_xp[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_xp = [train_set_xp[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    
    train_set = (train_set_x, train_set_xp, train_set_y)
    valid_set = (valid_set_x, valid_set_xp, valid_set_y)
    test_set = train_set

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]
    
    test_set_x, test_set_xp, test_set_y = test_set
    valid_set_x, valid_set_xp, valid_set_y = valid_set
    train_set_x, train_set_xp, train_set_y = train_set
    
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)
    train_set_xp = remove_unk(train_set_xp)
    valid_set_xp = remove_unk(valid_set_xp)
    test_set_xp = remove_unk(test_set_xp)

    train = (train_set_x, train_set_xp, train_set_y)
    valid = (valid_set_x, valid_set_xp, valid_set_y)
    test = (test_set_x, test_set_xp, test_set_y)

    return train, valid, test
