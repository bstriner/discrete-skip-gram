import numpy as np
import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from .utils import build_embedding
from .utils import softmax_nd


class SkipgramValidationFlatLayer(Layer):
    """
    
    """

    def __init__(self,
                 z_k,
                 z_depth,
                 y_k,
                 floating='float64',
                 embeddings_initializer='random_uniform', embeddings_regularizer=None):
        self.floating = floating
        self.z_k = z_k
        self.z_depth = z_depth
        self.y_k = y_k
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        self.p_yzs = []
        self.p_masks = []
        Layer.__init__(self)

    def build(self, (pz, y)):
        assert (len(pz) == 2)  # n, z_depth
        assert pz[1] == self.z_depth
        assert (len(y) == 2)  # n, 1

        self.p_yzs = []
        for i in range(self.z_depth):
            p_yz = build_embedding(self, (int(np.power(2, i + 1)), self.y_k), "p_yz_{}".format(i), dtype=self.floating)
            self.p_yzs.append(p_yz)
        self.built = True

    def compute_output_shape(self, (pz, y)):
        assert (len(pz) == 2)  # n, z_depth
        assert pz[1] == self.z_depth
        assert (len(y) == 2)  # n, 1
        return y[0], pz[1]

    def calc_loss1(self, z, y, depth):
        # p0: (n,)
        # p_z: (n, z_depth, z_k)
        z_t = z[:, :(depth + 1)]  # (n, depth)
        # mask = T.power(self.z_k, T.arange(depth + 1, dtype='int32'))  # (depth,)
        _mask = np.power(self.z_k, np.arange(depth + 1, dtype=np.int32)).astype(np.int32)
        mask = T.constant(_mask)  # (depth,)
        buckets = T.sum(z_t * (mask.dimshuffle(('x', 0))), axis=1)  # (n,)
        pyz1 = self.p_yzs[depth]  # (buckets, y_k)
        pyz2 = pyz1[buckets, :]  # (n, y_k)
        pyz3 = softmax_nd(pyz2)  # (n, y_k)
        pyz4 = pyz3[T.arange(y.shape[0]), T.flatten(y)]  # (n,)
        loss = -T.log(pyz4)  # (n,)
        return loss

    def calc_loss2(self, z, y, depth):
        # p0: (n,)
        # p_z: (n, z_depth, z_k)
        z_t = z[:, :(depth + 1)]  # (n, depth)
        # mask = T.power(self.z_k, T.arange(depth + 1, dtype='int32'))  # (depth,)
        _mask = np.power(self.z_k, np.arange(depth + 1, dtype=np.int32)).astype(np.int32)
        mask = T.constant(_mask)  # (depth,)
        buckets = T.sum(z_t * (mask.dimshuffle(('x', 0))), axis=1)  # (n,)
        pyz = softmax_nd(self.p_yzs[depth])  # (buckets, y_k)
        pyz_t = pyz[buckets, T.flatten(y)]
        eps = 1e-5
        loss = -T.log(pyz_t+eps)  # (n,)
        return loss

    def calc_loss(self, z, y, depth):
        if np.power(self.z_k, depth) >= 256:
            return self.calc_loss2(z, y, depth)
        else:
            return self.calc_loss1(z, y, depth)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        (z, y) = inputs
        # z: encoding: (n, z_depth) [int 0-k]
        # y: ngram: (n, 1) int32
        losses = [self.calc_loss2(z=z, y=y, depth=i) for i in range(self.z_depth)]
        lossarray = T.stack(losses, axis=1)  # (n, z_depth)
        return lossarray
