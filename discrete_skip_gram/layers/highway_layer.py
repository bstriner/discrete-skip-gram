import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import build_kernel, build_bias, build_pair, shift_tensor, leaky_relu
from ..units.mlp_unit import MLPUnit


class HighwayLayer(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self,
                 units,
                 hidden_layers=2,
                 inner_activation=leaky_relu,
                 layernorm=False,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='random_uniform', bias_regularizer=None,
                 **kwargs):
        self.units = units
        self.layernorm=layernorm
        self.hidden_layers = hidden_layers
        self.inner_activation = inner_activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = False
        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        input_dim = input_shape[2]
        self.mlp = MLPUnit(self,
                           input_units=[self.units, input_dim],
                           units=self.units,
                           output_units=self.units,
                           hidden_layers=self.hidden_layers,
                           inner_activation=self.inner_activation,
                           layernorm=self.layernorm,
                           name="mlp")
        self.h0 = build_bias(self, (1, self.units), "h0")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return input_shape[0], input_shape[1], self.units

    def step(self, x, y0, *params):
        hd = self.mlp.call([y0, x], params)
        y1 = hd + y0
        return y1

    def call(self, x):
        # x: (n, depth, input_dim)
        xr = T.transpose(x, (1, 0, 2))  # (depth, n, input_dim)
        n = x.shape[0]
        outputs_info = [T.extra_ops.repeat(self.h0, n, axis=0)]
        yr, _ = theano.scan(self.step,
                            sequences=[xr],
                            outputs_info=outputs_info,
                            non_sequences=self.mlp.non_sequences)
        # yr: (depth, n, units)
        y = T.transpose(yr, (1, 0, 2))  # (n, depth, units)
        return y
