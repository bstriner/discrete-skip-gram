import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair, shift_tensor, leaky_relu, layer_norm
from ..units.mlp_unit import MLPUnit


class HighwayLayerDiscrete(Layer):
    """
    Highway rnn with discrete inputs
    """

    def __init__(self,
                 units,
                 embedding_units,
                 k,
                 hidden_layers=2,
                 layernorm=False,
                 inner_activation=leaky_relu,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='random_uniform', bias_regularizer=None,
                 **kwargs):
        self.units = units
        self.embedding_units = embedding_units
        self.k = k
        self.layernorm = layernorm
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.x_embedding = W(self, (self.k, self.embedding_units), name="x_embedding")
        self.mlp = MLPUnit(self,
                           input_units=[self.units, self.embedding_units],
                           units=self.units,
                           output_units=self.units,
                           hidden_layers=self.hidden_layers,
                           inner_activation=self.inner_activation,
                           layernorm=self.layernorm,
                           name="mlp")
        self.non_sequences = [self.x_embedding] + self.mlp.non_sequences
        self.h0 = b(self, (1, self.units), "h0")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape[0], input_shape[1], self.units

    def step(self, x, y0, *params):
        ind = 0
        x_embedding = params[ind]
        ind += 1
        mlpparams = params[ind:(ind + self.mlp.count)]
        ind += self.mlp.count
        assert ind == len(params)

        xe = x_embedding[x, :]
        hd = self.mlp.call([y0, xe], mlpparams)
        y1 = hd + y0
        return y1

    def call(self, x):
        # x: (n, depth)
        xr = T.transpose(x, (1, 0))  # (depth, n)
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0)]
        yr, _ = theano.scan(self.step,
                            sequences=[xr],
                            outputs_info=outputs_info,
                            non_sequences=self.non_sequences)
        # yr: (depth, n, units)
        y = T.transpose(yr, (1, 0, 2))  # (n, depth, units)
        return y
