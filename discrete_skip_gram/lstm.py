import os

# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import theano
import theano.tensor as T
from keras.initializers import zero, glorot_uniform
from keras.layers import Input
import numpy as np
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from .backend import cumprod, cumsum

import keras.backend as K
def lstm_function(x, h, y,
                  W_h, U_h, b_h,
                  W_f, b_f,
                  W_i, b_i,
                  W_c, b_c,
                  W_o, b_o,
                  W_j, b_j,
                  W_v, b_v
                  ):
    """
    x = input (int) [sequence]
    h = hidden state [prior]
    y = prior output [prior] (unused)
    :return:
    """
    hh = T.tanh(T.dot(h, W_h) + U_h[x, :] + b_h)
    f = T.nnet.sigmoid(T.dot(hh, W_f) + b_f)
    i = T.nnet.sigmoid(T.dot(hh, W_i) + b_i)
    o = T.nnet.sigmoid(T.dot(hh, W_c) + b_c)
    w = T.tanh(T.dot(hh, W_o) + b_o)
    h_t = f * h + i * w
    h1 = o * h_t
    h2 = T.tanh(T.dot(h1, W_j) + b_j)
    # y_t = T.tanh(T.dot(h2, W_v) + b_v)
    y_t = T.dot(h2, W_v) + b_v
    return h_t, y_t

def make_W(shape, name):
    return K.variable(glorot_uniform()(shape), name=name)
def make_b(shape, name):
    return K.variable(zero()(shape), name=name)


class LSTM(object):
    """
    h = hidden state
    x = input (int)
    """

    def __init__(self, name, k, hidden_dim):
        self.k = k
        self.hidden_dim = hidden_dim

        # Hidden representation
        self.W_h = make_W((hidden_dim, hidden_dim), "{}_W_h".format(name))  # h, (hidden_dim, hidden_dim)
        self.U_h = make_W((k + 1, hidden_dim), "{}_U_h".format(name))  # x, (k+1, hidden_dim)
        self.b_h = make_b((hidden_dim,), "{}_b_h".format(name))  # (hidden_dim,)

        # Forget gate
        self.W_f = make_W((hidden_dim, hidden_dim), "{}_W_f".format(name))
        self.b_f = make_b((hidden_dim,), "{}_b_f".format(name))
        # Input gate
        self.W_i = make_W((hidden_dim, hidden_dim), "{}_W_i".format(name))
        self.b_i = make_b((hidden_dim,), "{}_b_i".format(name))
        # Write gate
        self.W_w = make_W((hidden_dim, hidden_dim), "{}_W_w".format(name))
        self.b_w = make_b((hidden_dim,), "{}_b_w".format(name))
        # Output
        self.W_o = make_W((hidden_dim, hidden_dim), "{}_W_o".format(name))
        self.b_o = make_b((hidden_dim,), "{}_b_o".format(name))
        # Hidden state
        self.W_j = make_W((hidden_dim, hidden_dim), "{}_W_j".format(name))
        self.b_j = make_b((hidden_dim,), "{}_b_j".format(name))
        # y predictions
        self.W_y = make_W((hidden_dim, hidden_dim), "{}_W_y".format(name))
        self.b_y = make_b((hidden_dim,), "{}_b_y".format(name))
        # self.clip_params = [self.W_h, self.U_h, self.W_f, self.W_i, self.W_w, self.W_o, self.W_j, self.W_y]
        self.params = [self.W_h, self.U_h, self.b_h,
                       self.W_f, self.b_f,
                       self.W_i, self.b_i,
                       self.W_w, self.b_w,
                       self.W_o, self.b_o,
                       self.W_j, self.b_j,
                       self.W_y, self.b_y]

    def call(self, x):
        n = x.shape[0]
        xr = T.transpose(x, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'), T.zeros((n, self.hidden_dim), dtype='float32')]
        (_, yr), _ = theano.scan(lstm_function, sequences=[xr], outputs_info=outputs_info, non_sequences=self.params)
        y = T.transpose(yr, (1, 0, 2))
        return y[:, -1, :]

    def regularization_loss(self, W_regularizer):
        ws = [self.W_h, self.U_h, self.W_f, self.W_i, self.W_w, self.W_o, self.W_j, self.W_y]
        loss = 0.0
        for w in ws:
            loss += W_regularizer(w)
        return loss
