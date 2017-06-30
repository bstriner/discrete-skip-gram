import itertools

import numpy as np
import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from .utils import build_embedding
from .utils import leaky_relu
from .utils import uniform_smoothing, softmax_nd


def merge_losses(a, b):
    c = {}
    for k in set(a.keys() + b.keys()):
        c[k] = []
        if k in a:
            for v in a[k]:
                c[k].append(v)
        if k in b:
            for v in b[k]:
                c[k].append(v)
    return c


def mean_tensors(x):
    l = len(x)
    if l == 0:
        raise ValueError("Error")
    if l == 1:
        return x[0]
    else:
        return add_tensors(x) / np.float32(l)


def add_tensors(x):
    l = len(x)
    if l == 0:
        raise ValueError("Error")
    elif l == 1:
        return x[0]
    else:
        mid = int(l / 2)
        return add_tensors(x[:mid]) + add_tensors(x[mid:])


class SkipgramLookaheadFlatLayer(Layer):
    """
    
    """

    def __init__(self,
                 z_k,
                 z_depth,
                 y_k,
                 units,
                 embedding_units,
                 lookahead_depth,
                 hidden_layers=2,
                 inner_activation=leaky_relu,
                 layernorm=False,
                 batchnorm=False,
                 negative_sampling=None,
                 floating='float64',
                 embeddings_initializer='random_uniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        assert lookahead_depth == z_depth
        self.negative_sampling = negative_sampling
        self.floating = floating
        self.z_k = z_k
        self.z_depth = z_depth
        self.lookahead_depth = lookahead_depth
        self.layernorm = layernorm
        self.batchnorm = batchnorm
        self.y_k = y_k
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation
        self.units = units
        self.embedding_units = embedding_units
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        self.p_yzs = []
        self.p_masks = []
        Layer.__init__(self)

    def build(self, (pz, y)):
        assert (len(pz) == 3)  # n, z_depth, z_k
        assert pz[1] == self.z_depth
        assert pz[2] == self.z_k
        assert (len(y) == 2)  # n, 1

        self.p_yzs = []
        self.p_masks = []
        for i in range(self.z_depth):
            p_yz = build_embedding(self, (int(np.power(2, i + 1)), self.y_k), "p_yz_{}".format(i), dtype=self.floating)
            self.p_yzs.append(p_yz)
            masks = []
            combos = list(itertools.product(list(range(self.z_k)), repeat=i + 1))
            for j in range(i + 1):
                mask = []
                for k in range(self.z_k):
                    m = np.array([c[j] == k for c in combos]).reshape((1, 1, -1))
                    mask.append(m)
                mask = np.concatenate(mask, axis=1)  # (1, z_k, buckets)
                masks.append(mask)
            masks = np.concatenate(masks, axis=0)  # (depth, z_k, buckets)
            if self.floating == 'float32':
                masks = T.constant(masks.astype(np.float32), name="mask_{}".format(i))
            else:
                masks = T.constant(masks.astype(np.float64), name="mask_{}".format(i))
            self.p_masks.append(masks)
        self.built = True

    def compute_output_shape(self, (pz, y)):
        assert (len(pz) == 3)  # n, z_depth, z_k
        assert pz[1] == self.z_depth
        assert pz[2] == self.z_k
        assert (len(y) == 2)  # n, 1
        return y[0], pz[1]

    def calc_loss(self, p_z, y, depth):
        # p0: (n,)
        # p_z: (n, z_depth, z_k)
        p_z_t = p_z[:, :(depth + 1), :]  # (n, depth, z_k)
        m = self.p_masks[depth]  # (depth, z_k, buckets)
        t = (p_z_t.dimshuffle((0, 1, 2, 'x'))) * (m.dimshuffle(('x', 0, 1, 2)))  # (n, depth, z_k, buckets)
        pm = T.prod(T.sum(t, axis=2), axis=1)  # (n, buckets)
        mode = 0
        if mode == 0:
            # log(p*q)
            pyz = uniform_smoothing(softmax_nd(self.p_yzs[depth]))  # (buckets, y_k)
            pyz_t = T.transpose(pyz, (1, 0))[T.flatten(y), :]  # (n, buckets)
            loss = -T.log(T.sum(pyz_t * pm, axis=1, keepdims=True))  # (n,1)
        elif mode == 1:
            # p*log(q)
            pyz = uniform_smoothing(softmax_nd(self.p_yzs[depth]), 1e-6)  # (buckets, y_k)
            nllt = -T.log(T.transpose(pyz, (1, 0))[T.flatten(y), :])  # (n, buckets)
            loss = T.sum(nllt * pm, axis=1, keepdims=True)  # (n,1)
        elif mode == 2:
            # log(p) * log(q)
            #t = (T.log(p_z_t.dimshuffle((0, 1, 2, 'x'))) +
            #     T.log(m.dimshuffle(('x', 0, 1, 2))))  # (n, depth, z_k, buckets)
            #pm = T.sum(T.sum(t, axis=2), axis=1)  # (n, buckets)
            pyz = uniform_smoothing(softmax_nd(self.p_yzs[depth]))  # (buckets, y_k)
            nllt = -T.log(T.transpose(pyz, (1, 0))[T.flatten(y), :])  # (n, buckets)
            loss = T.sum(nllt * T.log(pm), axis=1, keepdims=True)  # (n,1)
        return loss

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        (p_z, y) = inputs
        # p_z: p(z): (n, z_depth, z_k)
        # y: ngram: (n, 1) int32
        losses = [self.calc_loss(p_z=p_z, y=y, depth=i) for i in range(self.z_depth)]
        lossarray = T.concatenate(losses, axis=1)  # (n, z_depth)
        return lossarray
