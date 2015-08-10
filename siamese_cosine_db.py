from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import ppdb

datasets = {'ppdb': (ppdb.load_data, ppdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params['Semb1'] = (0.01 * randn).astype(config.floatX)
    params['Semb2'] = (0.01 * randn).astype(config.floatX)
    params['Semb3'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
   
    # classifier
    params['wcosine'] = numpy.cast['float64'](0.01)
    params['bcosine'] = numpy.cast['float64'](0.01)
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            print ('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]



layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask_x, xp, mask_xp, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask_x, xp, mask_xp, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask_x, xp, mask_xp, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask_x, xp, mask_xp, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    x = tensor.matrix('x', dtype='int64')
    xp = tensor.matrix('xp', dtype='int64')
    mask_x = tensor.matrix('mask_x', dtype=config.floatX)
    mask_xp = tensor.matrix('mask_xp', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps_x = x.shape[0]
    n_samples_x = x.shape[1]

    Wemb = tparams['Wemb'][x.flatten()].reshape([n_timesteps_x, n_samples_x, options['dim_proj']])
    Semb1 = tparams['Semb1'][x.flatten()].reshape([n_timesteps_x,n_samples_x,options['dim_proj']])
    Semb2 = tparams['Semb2'][x.flatten()].reshape([n_timesteps_x,n_samples_x,options['dim_proj']])
    Semb3 = tparams['Semb3'][x.flatten()].reshape([n_timesteps_x,n_samples_x,options['dim_proj']])
    context = theano.tensor.sum(Wemb, axis=1, keepdims=True)
    score1 = theano.tensor.sum(Semb1 * context, axis=2, keepdims=True)+1e-8
    score2 = theano.tensor.sum(Semb2 * context, axis=2, keepdims=True)+1e-8
    score3 = theano.tensor.sum(Semb3 * context, axis=2, keepdims=True)+1e-8
    normalization = score1+score2+score3
    score1 = score1/normalization
    score2 = score2/normalization
    score3 = score3/normalization
    emb_x = score1*Semb1 + score2*Semb2 + score3*Semb3
    proj_x = get_layer(options['encoder'])[1](tparams, emb_x, options,
                                            prefix=options['encoder'],
                                            mask=mask_x)
    if options['encoder'] == 'lstm':
        proj_x = (proj_x * mask_x[:, :, None]).sum(axis=0)
        proj_x = proj_x / mask_x.sum(axis=0)[:, None]

    f_pred = theano.function([x, mask_x], proj_x, name='f_pred')

    n_timesteps_p = xp.shape[0]
    n_samples_p = xp.shape[1]

    Wemb_p = tparams['Wemb'][xp.flatten()].reshape([n_timesteps_p, n_samples_p, options['dim_proj']])
    Semb1_p = tparams['Semb1'][xp.flatten()].reshape([n_timesteps_p,n_samples_p,options['dim_proj']])
    Semb2_p = tparams['Semb2'][xp.flatten()].reshape([n_timesteps_p,n_samples_p,options['dim_proj']])
    Semb3_p = tparams['Semb3'][xp.flatten()].reshape([n_timesteps_p,n_samples_p,options['dim_proj']])

    context_p = theano.tensor.sum(Wemb_p, axis=1, keepdims=True)
    score1_p = theano.tensor.sum(Semb1_p * context_p, axis=2, keepdims=True)+1e-8
    score2_p = theano.tensor.sum(Semb2_p * context_p, axis=2, keepdims=True)+1e-8
    score3_p = theano.tensor.sum(Semb3_p * context_p, axis=2, keepdims=True)+1e-8
    normalization_p = score1_p + score2_p + score3_p
    score1_p = score1_p / normalization_p
    score2_p = score2_p / normalization_p
    score3_p = score3_p / normalization_p
    emb_xp = score1_p * Semb1_p + score2_p * Semb2_p + score3_p * Semb3_p
    proj_xp = get_layer(options['encoder'])[1](tparams, emb_xp, options,
                                          prefix=options['encoder'],
                                          mask=mask_xp)
    if options['encoder'] == 'lstm':
        proj_xp = (proj_xp * mask_xp[:, :, None]).sum(axis=0)
        proj_xp = proj_xp / mask_xp.sum(axis=0)[:, None]

    unit_x = proj_x / tensor.sqrt((proj_x ** 2).sum(axis=1,keepdims=True))
    unit_xp = proj_xp / tensor.sqrt((proj_xp ** 2).sum(axis=1,keepdims=True))
    dist = (unit_x * unit_xp).sum(axis=1)
    pred_prob = tensor.nnet.sigmoid(tparams['wcosine'] * dist + tparams['bcosine'])

    cost = (y - pred_prob).sum() **2

    return x, mask_x, xp, mask_xp, y, f_pred, cost

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-1.0 * z))

def pred_error(tparams, f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask_x = prepare_data([data[0][t] for t in valid_index])
        xp, mask_xp = prepare_data([data[1][t] for t in valid_index])
        y = numpy.array(data[2])[valid_index]
        
        preds_x = f_pred(x, mask_x)
        preds_xp = f_pred(xp, mask_xp)
    
        unit_x = preds_x / numpy.sqrt((preds_x ** 2).sum(axis=1,keepdims=True))
        unit_xp = preds_xp / numpy.sqrt((preds_xp ** 2).sum(axis=1,keepdims=True))
        dist = (unit_x * unit_xp).sum(axis=1)
        pred_prob = sigmoid(tparams['wcosine'].get_value() * dist + tparams['bcosine'].get_value())
        err = y - pred_prob
        valid_err += (err.sum()) ** 2
    valid_err = valid_err / len(data[0])

    return valid_err


def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=62,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='ppdb',
    decay_c=0.01,
    # Parameter for extra option
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, valid, test = load_data(n_words=n_words, valid_portion=0.1,maxlen=None)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx], [test[2][n] for n in idx])

    ydim = 1

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # build the symbolic graph
    (x, mask_x, xp, mask_xp, y, f_pred, cost) = build_model(tparams, model_options)
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = (tparams['wcosine'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    f_cost = theano.function([x, mask_x, xp, mask_xp, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask_x, xp, mask_xp, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask_x, xp, mask_xp, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train[2][t] for t in train_index]
                x = [train[0][t]for t in train_index]
                xp = [train[1][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask_x = prepare_data(x)
                xp, mask_xp = prepare_data(xp)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask_x, xp, mask_xp, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    train_err = pred_error(tparams, f_pred, prepare_data, train, kf)
                    valid_err = pred_error(tparams, f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(tparams, f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(tparams, f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(tparams, f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(tparams, f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100,
        test_size=2,
        reload_model=True
    )
