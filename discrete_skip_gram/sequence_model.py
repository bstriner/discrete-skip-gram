import os

# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import theano
import theano.tensor as T
from keras.initializations import zero, glorot_uniform
from keras.layers import Input
import numpy as np
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from .backend import cumprod, cumsum


def inner_function(xprev, h, z,
                   W_h, U_h, V_h, b_h,
                   W_f, b_f,
                   W_i, b_i,
                   W_c, b_c,
                   W_o, b_o,
                   W_j, b_j,
                   W_v, b_v
                   ):
    hh = T.tanh(T.dot(h, W_h) + U_h[xprev, :] + T.dot(z, V_h) + b_h)
    f = T.nnet.sigmoid(T.dot(hh, W_f) + b_f)
    i = T.nnet.sigmoid(T.dot(hh, W_i) + b_i)
    o = T.nnet.sigmoid(T.dot(hh, W_c) + b_c)
    w = T.tanh(T.dot(hh, W_o) + b_o)
    h_t = f * h + i * w
    h1 = o * h_t
    h2 = T.tanh(T.dot(h1, W_j) + b_j)
    y_t = T.nnet.softmax(T.dot(h2, W_v) + b_v)
    switch = T.eq(xprev, 1).dimshuffle((0, 'x'))
    ending = T.concatenate((T.ones((1,)), T.zeros((y_t.shape[1] - 1,)))).dimshuffle(('x', 0))
    y_tt = (1 - switch) * y_t + switch * ending
    return h_t, y_tt


# seq, prior, non-seq
def likelihood_function(xprev, x, h, y, z,
                        W_h, U_h, V_h, b_h,
                        W_f, b_f,
                        W_i, b_i,
                        W_c, b_c,
                        W_o, b_o,
                        W_j, b_j,
                        W_v, b_v):
    """
    Returns sequences of likelihoods given sequences of X
    xprev = previous output (n,) int [sequence] (k+2)
    x = output (n,) int [sequence] (k+1)
    h = hidden state (n, hidden_dim) [prior]
    y = last discriminator value (unused) [prior]
    z = context [non-sequence]
    """
    h_t, y_t = inner_function(xprev, h, z, W_h, U_h, V_h, b_h,
                              W_f, b_f,
                              W_i, b_i,
                              W_c, b_c,
                              W_o, b_o,
                              W_j, b_j,
                              W_v, b_v)
    y_tt = y_t[T.arange(y_t.shape[0]), x]
    return h_t, y_tt


# seq, prior, non-seq
def policy_function(rng, h, xprev, z,
                    W_h, U_h, V_h, b_h,
                    W_f, b_f,
                    W_i, b_i,
                    W_c, b_c,
                    W_o, b_o,
                    W_j, b_j,
                    W_v, b_v):
    """
    Creates sequence of x given rng
    rng = random [0-1] [sequence]
    h = hidden state (n, hidden_dim) [prior]
    xprev = output (n,) int [prior] (k+2)
    z = context [non-sequence]
    """
    h_t, y_t = inner_function(xprev, h, z, W_h, U_h, V_h, b_h,
                              W_f, b_f,
                              W_i, b_i,
                              W_c, b_c,
                              W_o, b_o,
                              W_j, b_j,
                              W_v, b_v)
    p_t = cumsum(y_t)
    gt = T.gt(rng.dimshuffle((0, 'x')), p_t)
    x_t = T.sum(gt, axis=1)
    x_t = T.clip(x_t, 0, p_t.shape[1] - 1)
    x_t += np.int32(1)
    x_t = T.cast(x_t, "int32")
    return h_t, x_t


class SequenceModel(object):
    def __init__(self, name, k, depth, latent_dim, hidden_dim):
        self.depth = depth
        self.k = k
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Hidden representation
        self.W_h = glorot_uniform((hidden_dim, hidden_dim), "{}_W_h".format(name))  # h, (hidden_dim, hidden_dim)
        self.U_h = glorot_uniform((k + 2, hidden_dim), "{}_U_h".format(name))  # x, (k+1, hidden_dim)
        self.V_h = glorot_uniform((latent_dim, hidden_dim), "{}_V_h".format(name))  # z (latent_dim, hidden_dim)
        self.b_h = zero((hidden_dim,), "{}_b_h".format(name))  # (hidden_dim,)

        # Forget gate
        self.W_f = glorot_uniform((hidden_dim, hidden_dim), "{}_W_f".format(name))  # z, (latent_dim, hidden_dim)
        self.b_f = zero((hidden_dim,), "{}_b_f".format(name))  # (hidden_dim,)
        # Input gate
        self.W_i = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_i = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Write gate
        self.W_w = glorot_uniform((hidden_dim, hidden_dim), "{}_W_w".format(name))  # z, (latent_dim, hidden_dim)
        self.b_w = zero((hidden_dim,), "{}_b_w".format(name))  # (hidden_dim,)
        # Output
        self.W_o = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_o = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Hidden state
        self.W_j = glorot_uniform((hidden_dim, hidden_dim), "{}_W_j".format(name))  # z, (latent_dim, hidden_dim)
        self.b_j = zero((hidden_dim,), "{}_b_j".format(name))  # (hidden_dim,)
        # y predictions
        self.W_y = glorot_uniform((hidden_dim, k + 1), "{}_W_y".format(name))  # z, (latent_dim, hidden_dim)
        self.b_y = zero((k + 1,), "{}_b_y".format(name))  # (hidden_dim,)
        # self.clip_params = [self.W_h, self.U_h, self.W_f, self.W_i, self.W_w, self.W_o, self.W_j, self.W_y]
        self.params = [self.W_h, self.U_h, self.V_h, self.b_h,
                       self.W_f, self.b_f,
                       self.W_i, self.b_i,
                       self.W_w, self.b_w,
                       self.W_o, self.b_o,
                       self.W_j, self.b_j,
                       self.W_y, self.b_y]

    def partial_likelihood(self, x, z):
        # xprev, x, h, y, z
        n = x.shape[0]
        xprev = T.concatenate((T.zeros((n, 1), dtype='int32'), 1 + x[:, :-1]), axis=1)
        xr = T.transpose(x, (1, 0))
        xprevr = T.transpose(xprev, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n,), dtype='float32')]
        (_, pr), _ = theano.scan(likelihood_function, sequences=[xprevr, xr], outputs_info=outputs_info,
                                 non_sequences=[z] + self.params)
        p = T.transpose(pr, (1, 0))
        return p

    def likelihood(self, x, z):
        return cumprod(self.partial_likelihood(x, z))

    def nll(self, x, z):
        return -T.log(self.likelihood(x, z))

    def policy(self, rng, z):
        # rng, h, xprev, z,
        n = rng.shape[0]
        rngr = T.transpose(rng, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n,), dtype='int32')]
        (_, xr), _ = theano.scan(policy_function, sequences=[rngr], outputs_info=outputs_info,
                                 non_sequences=[z] + self.params)
        x = T.transpose(xr, (1, 0)) - 1
        return theano.gradient.zero_grad(x)
