import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from .utils import build_bias, build_embedding
from .utils import leaky_relu
from .utils import uniform_smoothing, softmax_nd
from ..units.mlp_unit import MLPUnit


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


def add_tensors(x):
    l = len(x)
    if l == 0:
        raise ValueError("Error")
    elif l == 1:
        return x[0]
    elif l == 2:
        return x[0] + x[1]
    else:
        mid = int(l / 2)
        return add_tensors(x[:mid]) + add_tensors(x[mid:])


class SkipgramLookaheadLayer(Layer):
    """
    
    """

    def __init__(self,
                 z_k,
                 z_depth,
                 lookahead_depth,
                 y_k,
                 units,
                 embedding_units,
                 hidden_layers=2,
                 inner_activation=leaky_relu,
                 layernorm=False,
                 batchnorm=False,
                 negative_sampling=None,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.negative_sampling = negative_sampling
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
        Layer.__init__(self)

    def build(self, (pz, y)):
        assert (len(pz) == 3)  # n, z_depth, z_k
        assert pz[1] == self.z_depth
        assert pz[2] == self.z_k
        assert (len(y) == 2)  # n, 1
        self.z_embedding = build_embedding(self, (self.z_k, self.embedding_units), 'z_embedding')
        self.rnn = MLPUnit(self,
                           input_units=[self.units, self.embedding_units],
                           units=self.units,
                           output_units=self.units,
                           inner_activation=self.inner_activation,
                           hidden_layers=self.hidden_layers,
                           layernorm=self.layernorm,
                           batchnorm=self.batchnorm,
                           name="rnn")
        self.mlp = MLPUnit(self,
                           input_units=[self.units],
                           units=self.units,
                           output_units=self.y_k,
                           inner_activation=self.inner_activation,
                           hidden_layers=self.hidden_layers,
                           layernorm=self.layernorm,
                           batchnorm=self.batchnorm,
                           name="mlp")
        self.h0 = build_bias(self, (self.units,), name="h0")
        self.built = True

    def compute_output_shape(self, (pz, y)):
        assert (len(pz) == 3)  # n, z_depth, z_k
        assert pz[1] == self.z_depth
        assert pz[2] == self.z_k
        assert (len(y) == 2)  # n, 1
        return y[0], pz[1]

    def recurse(self, h0, p0, p_z, y, depth, max_depth):
        # h0: (1, units)
        # p_z: (n, depth, z_k)
        # y: (n, 1)
        if depth >= max_depth:
            return {}
        hd = self.rnn.call([h0, self.z_embedding], self.rnn.non_sequences)  # (z_k, units)
        h1 = h0 + hd  # (z_k, units)
        y1 = self.mlp.call([h1], self.mlp.non_sequences)  # (z_k, y_k)
        p1 = uniform_smoothing(softmax_nd(y1))  # (z_k, y_k)
        nll1 = -T.log(p1)  # (z_k, y_k)
        nllt = T.transpose(nll1, (1, 0))[T.flatten(y), :]  # (n, z_k)
        pzt = p0.dimshuffle((0, 'x')) * (p_z[:, depth, :])  # (n, z_k)
        loss = T.sum(pzt * nllt, axis=1, keepdims=True)  # (n, 1)
        losses = {depth: [loss]}
        for z in range(self.z_k):
            h2 = T.reshape(h1[z, :], (1, -1))
            sublosses = self.recurse(h0=h2, p_z=p_z, y=y, depth=depth + 1, p0=pzt[:, z], max_depth=max_depth)
            losses = merge_losses(losses, sublosses)
        return losses

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        (p_z, y) = inputs
        # p_z: p(z): (n, z_depth, z_k)
        # y: ngram: (n, 1) int32
        n = y.shape[0]
        p0 = T.ones(shape=(n,), dtype='float32')  # (n,)

        # h0 = T.extra_ops.repeat(self.h0, self.z_k, axis=0)
        h0 = self.h0.dimshuffle(('x', 0))
        losses = self.recurse(h0=h0, p0=p0, p_z=p_z, y=y, depth=0, max_depth=self.z_depth)
        assert len(losses.keys()) == self.z_depth
        lossvals = [add_tensors(losses[k]) for k in range(self.z_depth)]
        lossarray = T.concatenate(lossvals, axis=1)  # (n, z_depth)
        return lossarray
