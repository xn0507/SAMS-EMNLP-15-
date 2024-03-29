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

    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        for l, s in zip(lengths, seqs):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask




def load_data(path="sentences.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here sentences)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = train_set
    f.close()
    if maxlen:
        new_train_set_x = []
        for x in train_set:
            if len(x) < maxlen:
                new_train_set_x.append(x)
        train_set = new_train_set_x
        del new_train_set_x

    # split training set into validation set
    train_set_x = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]

    train_set = train_set_x
    valid_set = valid_set_x

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x= test_set
    valid_set_x = valid_set
    train_set_x = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]

    train = train_set_x
    valid = valid_set_x
    test = test_set_x
    print len(train)

    return train, valid, test
