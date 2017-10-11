import theano
import theano.tensor as T


def sample_gumbel(shape, srng, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = srng.uniform(size=shape, low=0., high=1.)
    return -T.log(-T.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, srng):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, srng=srng)
    return T.nnet.softmax(y / temperature)


def gumbel_softmax(logits, temperature, srng, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, srng=srng)
    if hard:
        k = logits.shape[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = T.cast(T.equal(y, T.max(y, axis=1, keep_dims=True)), y.dtype)
        y = theano.gradient.zero_grad(y_hard - y) + y
    return y
