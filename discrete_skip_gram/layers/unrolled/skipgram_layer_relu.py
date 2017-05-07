import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from ..utils import W, b, pair, shift_tensor
from ...units.mlp_unit import MLPUnit


class SkipgramLayerRelu(Layer):
    """
    Given a flattened context, calculate NLL of a series
    """

    def __init__(self, k, units, embedding_units, mean=True,
                 layernorm=False,
                 inner_activation = T.nnet.relu,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.k = k
        self.units = units
        self.layernorm = layernorm
        self.inner_activation = inner_activation
        self.embedding_units = embedding_units
        self.mean = mean
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):

        y_embedding = W(self, (self.k+1, self.embedding_units))
        self.rnn = MLPUnit(self,
                           input_units=[self.units, self.embedding_units, input_dim],
                           units=self.units,
                           output_units=self.units,
                           inner_activation=self.inner_activation,
                           name="rnn")
        self.mlp = MLPUnit(self,
                           input_units=[self.units, self.embedding_units],
                           units=self.units,
                           inner_activation=self.inner_activation,
                           output_units=self.units,
                           output_activation=T.nnet.softmax,
                           name="rnn")

        self.non_sequences = ([y_embedding]+
            self.rnn.non_sequences +
            self.mlp.non_sequences)
        h0 = b(self, (1, self.units), "h0")
        self.h0 = h0
        self.built = True

    def build(self, (z, x)):
        assert (len(z) == 2)
        assert (len(x) == 2)
        input_dim = z[1]
        self.build_params(input_dim)

    #    def compute_mask(self, inputs, mask=None):
    #        print ("Compute mask {}".format(mask))
    #        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        x = input_shape[1]
        assert (len(z) == 2)
        assert (len(x) == 2)
        if self.mean:
            return (x[0], 1)
        else:
            return x

    def step(self, y0, y1, h0, z, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        rnnparams = params[idx:(idx+self.rnn.count)]
        idx += self.rnn.count
        mlpparams = params[idx:(idx+self.mlp.count)]
        idx += self.mlp.count
        assert idx == len(params)

        yembedded = y_embedding[y0,:]
        hd = self.rnn.call([h0, yembedded, z], rnnparams)
        h1 = hd+h0
        p1 = self.mlp.call([h1], mlpparams)
        eps = 1e-7
        nll1 = -T.log(eps+p1[T.arange(p1.shape[0]), y1])
        # nll1 = T.reshape(nll1,(-1,1))
        return h1, nll1

    def call(self, (z, y)):
        # z: input context: n, input_dim
        # y: ngram: n, depth int32
        # print "Z NDIM: {}".format(z.ndim)
        yr = T.transpose(y, (1, 0))
        yshifted = shift_tensor(y)
        yshiftedr = T.transpose(yshifted, (1, 0))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[yshiftedr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class SkipgramPolicyLayer(Layer):
    def __init__(self, layer, srng, depth):
        self.layer = layer
        self.srng = srng
        self.depth = depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.depth)

    def step(self, rng, h0, y0, z, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        rnnparams = params[idx:(idx + self.rnn.count)]
        idx += self.rnn.count
        mlpparams = params[idx:(idx + self.mlp.count)]
        idx += self.mlp.count
        assert idx == len(params)

        yembedded = y_embedding[y0, :]
        hd = self.rnn.call([h0, yembedded, z], rnnparams)
        h1 = hd + h0
        p1 = self.mlp.call([h1], mlpparams)
        c1 = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, z):
        # z: input context: n, input_dim
        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
