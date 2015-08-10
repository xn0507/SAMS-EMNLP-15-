

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
from theano.printing import debugprint
#config.compute_test_value = 'raise'
datasets = {'sentences': (sentences.load_data, sentences.prepare_sentence)}

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
    Global parameter. For the embeding and the classifier.
    model_options = locals().copy()
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
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


def param_init_recnn(options, params, prefix='recnn'):
    """
    Init the RecNN parameter:
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def recnn_layer(tparams, node_feature, kids, options, prefix='recnn', mask_k=None):
    
    if node_feature.ndim == 2:
        n_timesteps = node_feature.shape[0]
        print 'yes'
    else:
        n_samples = 1

    assert mask_k is not None

    #def _input(k_slice, reshaped_node_feature_slice):
       # return reshaped_node_feature_slice[k_slice.flatten()]

    def _step(m_, k_, step_, node_feature):
        # m_: mask slice of a single timestep

        inputs = tensor.concatenate([node_feature[k_[0]],node_feature[k_[1]]])

        h = tensor.tanh(tensor.dot (tparams[_p(prefix, 'W')], inputs) +
                   tparams[_p(prefix, 'b')])

        
        #logic here: if mask_k is 1, replace the nodel value with the compositional result; if mask_k is 0, keep the previous node value
        previous_p = node_feature[step_]
        new_p = m_ * h + (1. - m_) * previous_p
        
        step = step_+1

        return step, tensor.set_subtensor(node_feature[step_], new_p)

    rval, updates = theano.scan(_step,
                                sequences=[mask_k, kids],
                                outputs_info=[tensor.alloc((0)),
                                              node_feature],
                                name=_p(prefix, '_layers'))
    #last row of the last result
    return rval[1][-1][-1]


# ff: Feed Forward (normal neural net), only useful to put after recnn
#     before the classifier.
layers = {'recnn': (param_init_recnn, recnn_layer)}


def sgd(lr, tparams, grads, x, kids, mask, x_p, cost):
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
    f_grad_shared = theano.function([x, kids, mask, x_p], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, kids, mask, x_p, cost):
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

    f_grad_shared = theano.function([x, kids, mask, x_p], cost, updates=zgup + rg2up,
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

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.vector('x', dtype='int64')
    x.tag.test_value = numpy.random.randint(4, size=5)
    x_p = tensor.vector('x_p', dtype='int64')
    x_p.tag.test_value = numpy.random.randint(4, size=5)
    k = tensor.matrix('k', dtype='int64')
    k.tag.test_value = numpy.random.randint(4, size=(5,2))
    mask_k = tensor.vector('mask', dtype=config.floatX)
    mask_k.tag.test_value = numpy.random.randint(2, size=5)
    #x and x_p have the same shape
    n_timesteps = x.shape[0]

    #shared variable
    node_feature = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                options['dim_proj']])
    #get_layer(options['encoder'])[1]=function call recnn_layer, proj is the composite
    proj = get_layer(options['encoder'])[1](tparams, node_feature, k, options,
                                            prefix=options['encoder'],
                                            mask_k=mask_k)

    #print 'pred:', proj.shape.tag.test_value
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.dot(proj, tparams['U']) + tparams['b']   #is this the right scoring function? 
    #the result is a list with a single element
    pred = pred[0]

    #function calls for scoring
    f_pred_prob = theano.function([x, k, mask_k], pred, name='f_pred_prob')
    #theano.printing.debugprint(f_pred)
    #similarly for x_p
    node_feature_p = tparams['Wemb'][x_p.flatten()].reshape([n_timesteps,
                                                options['dim_proj']])
    #get_layer(options['encoder'])[1]=function call recnn_layer, proj is the composite
    proj_p = get_layer(options['encoder'])[1](tparams, node_feature_p, k, options,
                                            prefix=options['encoder'],
                                            mask_k=mask_k)

    if options['use_dropout']:
        proj_p = dropout_layer(proj_p, use_noise, trng)

    pred_p = tensor.dot(proj_p, tparams['U']) + tparams['b']
    pred_p = pred_p[0]

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    margin = 1
    diff = pred - pred_p
    dist = diff + off
    cost = tensor.clip(1 - dist, 0, 1e999)
    #g = tensor.grad(proj.mean(), tparams['recnn_W'])
    #print 'cost:', g.tag.test_value
    return use_noise, x, k, mask_k, x_p, f_pred_prob, cost

def reorder(x, seed):
    random.seed(seed)
    random.shuffle(x)

def corrupt(x, seed, n_words):
    if len(x)<1: return
    for interval in range(len(x) // 10):
        random.seed(seed)
        pos = random.randint(interval*10, interval*10+9)
        emb_subst = random.randint(0, n_words)
        x[pos] = emb_subst
    random.seed(seed)
    emb_subst = random.randint(0, n_words)
    x[len(x)-1] = emb_subst


def pred_error(f_pred_prob, prepare_sentence, data, iterator, seed, n_words, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    """
    valid_err = 0
    for _, valid_index in iterator:
        x_batch, k_batch = [data[0][t] for t in valid_index], [data[1][t] for t in valid_index]
        for x, k in zip(x_batch, k_batch):
            x_p1 = copy.deepcopy(x)
            x_p2 = copy.deepcopy(x)
            reorder(x_p1,seed)
            corrupt(x_p2, seed, n_words)
            x, kx, mask_k = prepare_sentence(x,k)
            x_p1, _k, _mask_k = prepare_sentence(x_p1,k)
            x_p2, _k, _mask_k = prepare_sentence(x_p2,k)

            preds = f_pred_prob(x, kx, mask_k)
            preds_p1 = f_pred_prob(x_p1, kx, mask_k)
            preds_p2 = f_pred_prob(x_p2, kx, mask_k)
            cost1 = preds > preds_p1
            cost2 = preds > preds_p2
        
            valid_err += cost1 + cost2

    valid_err = valid_err * 1.0 / len(data[0])

    return valid_err

def train_recnn(
    dim_proj=128,  # word embeding dimension and recnn number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=1.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.000001,  # Learning rate for sgd, perhaps 0.000001 is better
    n_words=62,  # Vocabulary size
    optimizer=sgd,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='recnn',  # TODO: can be removed must be recnn.
    saveto='recnn_model.npz',  # The best model will be saved there
    validFreq=30,  # Compute the validation error after this number of update.
    saveFreq=100,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=2,  # The batch size during training.
    valid_batch_size=2,  # The batch size used for validation/test set.
    dataset='sentences',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    print "program starts"
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_sentence = get_dataset(dataset)

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
        test = [test[0][n] for n in idx], [test[1][n] for n in idx]

    ydim = 1

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('recnn_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, k, mask_k, x_p, f_pred_prob, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, k, mask_k, x_p], cost, name='f_cost')
    #print tparams.keys()

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, k, mask_k, x_p], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, k, mask_k, x_p, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)   #returns batch_idx, bath_examples
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
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)

            for _, train_index in kf:
                
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x_batch, k_batch  = [train[0][t]for t in train_index], [train[1][t]for t in train_index]
                n_samples += len(x_batch)
                
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                for x, k in zip (x_batch, k_batch):
                    uidx += 1
                    x_p1 = copy.deepcopy(x)
                    x_p2 = copy.deepcopy(x)
                    reorder(x_p1, SEED)
                    corrupt(x_p2, SEED, n_words)
                    x, kx, mask_k = prepare_sentence(x,k)
                    x_p1, _kx, _mask_k = prepare_sentence(x_p1,k)
                    x_p2, _kx, _mask_k = prepare_sentence(x_p2,k)
                    

                    cost1 = f_grad_shared(x, kx, mask_k, x_p1)
                    #print f_pred_prob(x, kx, mask_k), f_pred_prob(x_p1, kx, mask_k)
                    f_update(lrate)
                    

                    cost2 = f_grad_shared(x, kx, mask_k, x_p2)
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

            #print 'Seen %d samples' % n_samples*batch_size
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
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred_prob, prepare_sentence, train, kf_train_sorted, SEED, n_words)
    valid_err = pred_error(f_pred_prob, prepare_sentence, valid, kf_valid, SEED, n_words)
    test_err = pred_error(f_pred_prob, prepare_sentence, test, kf_test, SEED, n_words)

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
    train_recnn(
        max_epochs=5000,
        test_size=2,
    )
