

from collections import OrderedDict
import cPickle as pkl
import sys
import time
import random
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import copy
import sentences

datasets = {'sentences': (sentences.load_data, sentences.prepare_data)}

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


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    model_options = locals().copy()
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    #zeros = numpy.zeros_like(randn)
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params['Semb1'] = (0.01 * randn).astype(config.floatX)
    params['Semb2'] = (0.01 * randn).astype(config.floatX)
    params['Semb3'] = (0.01 * randn).astype(config.floatX)
    #params['Cemb1'] = (zeros).astype(config.floatX)
    #params['Cemb2'] = (zeros).astype(config.floatX)
    #params['Cemb3'] = (zeros).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

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


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, x_p, cost):
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
    f_grad_shared = theano.function([x, mask, x_p], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, x_p, cost):
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

    f_grad_shared = theano.function([x, mask, x_p], cost, updates=zgup + rg2up,
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


def rmsprop(lr, tparams, grads, x, mask, x_p, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, x_p], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    x_p = tensor.matrix('x_p', dtype='int64')

    #x and x_p have the same shape

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    Wemb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb1 = tparams['Semb1'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb2 = tparams['Semb2'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb3 = tparams['Semb3'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb1 = tparams['Cemb1'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb2 = tparams['Cemb2'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb3 = tparams['Cemb3'][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    context = theano.tensor.sum(Wemb, axis=1, keepdims=True)
    score1 = theano.tensor.sum(Semb1 * context, axis=2, keepdims=True)+1e-8
    score2 = theano.tensor.sum(Semb2 * context, axis=2, keepdims=True)+1e-8
    score3 = theano.tensor.sum(Semb3 * context, axis=2, keepdims=True)+1e-8
    normalization = score1+score2+score3
    score1 = score1/normalization
    score2 = score2/normalization
    score3 = score3/normalization
    #Cemb1 += score1*context
    #Cemb2 += score2*context
    #Cemb3 += score3*context
    emb = score1*Semb1 + score2*Semb2 + score3*Semb3
    #get_layer(options['encoder'])[1]=function call lstm_layer, proj is the composite
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.dot(proj, tparams['U']) + tparams['b']

    #function calls for scoring
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    
    #similarly for x_p
    Wemb_p = tparams['Wemb'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb1_p = tparams['Semb1'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb2_p = tparams['Semb2'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    Semb3_p = tparams['Semb3'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb1_p = tparams['Cemb1'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb2_p = tparams['Cemb2'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    #Cemb3_p = tparams['Cemb3'][x_p.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])
    context_p = theano.tensor.sum(Wemb_p, axis=1, keepdims=True)
    score1_p = theano.tensor.sum(Semb1_p * context_p, axis=2, keepdims=True)+1e-8
    score2_p = theano.tensor.sum(Semb2_p * context_p, axis=2, keepdims=True)+1e-8
    score3_p = theano.tensor.sum(Semb3_p * context_p, axis=2, keepdims=True)+1e-8
    normalization_p = score1_p + score2_p + score3_p
    score1_p = score1_p / normalization_p
    score2_p = score2_p / normalization_p
    score3_p = score3_p / normalization_p
    emb_p = score1_p * Semb1_p + score2_p * Semb2_p + score3_p * Semb3_p
    #get_layer(options['encoder'])[1]=function call lstm_layer, proj is the composite
    proj_p = get_layer(options['encoder'])[1](tparams, emb_p, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj_p = (proj_p * mask[:, :, None]).sum(axis=0)
        proj_p = proj_p / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj_p = dropout_layer(proj_p, use_noise, trng)

    pred_p = tensor.dot(proj_p, tparams['U']) + tparams['b']


    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    diff = pred - pred_p
    dist = diff + off
    cost = tensor.maximum(0, 1 - dist).mean()

    return use_noise, x, mask, x_p, f_pred_prob, f_pred, cost



def reorder(x, seed):
    for l in x:
        random.seed(seed)
        random.shuffle(l)

def corrupt(x, seed, n_words):
    for i in range(len(x)):
        if len(x[i])<1: continue
        for interval in range(len(x[i]) // 10):
            random.seed(seed)
            pos = random.randint(interval*10, interval*10+9)
            emb_subst = random.randint(0, n_words)
            x[i][pos] = emb_subst
        random.seed(seed)
        emb_subst = random.randint(0, n_words)
        x[i][len(x[i])-1] = emb_subst


def pred_error(f_pred_prob, prepare_data, data, iterator, seed, n_words, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x = [data[t] for t in valid_index]
        x_p1 = copy.deepcopy(x)
        x_p2 = copy.deepcopy(x)
        reorder(x_p1,seed)
        corrupt(x_p2, seed, n_words)
        x, mask = prepare_data(x)
        x_p1, _mask = prepare_data(x_p1)
        x_p2, _mask = prepare_data(x_p2)

        preds = f_pred_prob(x, mask)
        preds_p1 = f_pred_prob(x_p1, mask)
        preds_p2 = f_pred_prob(x_p2, mask)
        diff1 = x-x_p1
        dist1 = diff1
        cost1 = numpy.maximum(0, 1 - dist1).sum()
        diff2 = x-x_p2
        dist2 = diff2
        cost2 = numpy.maximum(0, 1 - dist2).sum()
        
        valid_err += cost1 + cost2

    valid_err = valid_err / len(data)

    return valid_err




def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=62,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=30,  # Compute the validation error after this number of update.
    saveFreq=100,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=2,  # The batch size during training.
    valid_batch_size=2,  # The batch size used for validation/test set.
    dataset='sentences',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    print "program starts"
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, valid, test = load_data(n_words=n_words, valid_portion=0.1,
                                   maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = [test[n] for n in idx]

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

    # use_noise is for dropout
    (use_noise, x, mask, x_p, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, x_p, mask], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, x_p, mask], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, x_p, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid), valid_batch_size)   #returns batch_idx, bath_examples
    kf_test = get_minibatches_idx(len(test), valid_batch_size)

    print "%d train examples" % len(train)
    print "%d valid examples" % len(valid)
    print "%d test examples" % len(test)

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train) / batch_size
    if saveFreq == -1:
        saveFreq = len(train) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [train[t]for t in train_index]
                
                x_p1 = copy.deepcopy(x)
                x_p2 = copy.deepcopy(x)
                reorder(x_p1, SEED)
                corrupt(x_p2, SEED, n_words)

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask = prepare_data(x)
                x_p1, _mask = prepare_data(x_p1)
                x_p2, _mask = prepare_data(x_p2)
                n_samples += x.shape[1]

                cost1 = f_grad_shared(x, mask, x_p1)
                f_update(lrate)

                cost2 = f_grad_shared(x, mask, x_p2)
                f_update(lrate)

                cost=cost1+cost2

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
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred_prob, prepare_data, train, kf, SEED, n_words)
                    valid_err = pred_error(f_pred_prob, prepare_data, valid,
                                           kf_valid, SEED, n_words)
                    test_err = pred_error(f_pred_prob, prepare_data, test, kf_test, SEED, n_words)

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

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train), batch_size)
    train_err = pred_error(f_pred_prob, prepare_data, train, kf_train_sorted, SEED, n_words)
    valid_err = pred_error(f_pred_prob, prepare_data, valid, kf_valid, SEED, n_words)
    test_err = pred_error(f_pred_prob, prepare_data, test, kf_test, SEED, n_words)

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
        max_epochs=5000,
        test_size=2,
        reload_model=True
    )
