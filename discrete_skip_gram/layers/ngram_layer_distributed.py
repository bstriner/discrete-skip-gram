import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair, shift_tensor
from .ngram_layer import NgramLayer


class NgramLayerDistributed(NgramLayer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self, *args, **kwargs):
        NgramLayer.__init__(self,*args, **kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        x = input_shape[1]
        assert (len(z) == 3)
        assert (len(x) == 2)
        input_dim = z[2]
        self.build_params(input_dim)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        x = input_shape[1]
        assert (len(z) == 3)
        assert (len(x) == 2)
        return x[0], z[1], x[1]

    def wrapper_step(self, z, xprev, x, h0, *params):
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[xprev, x], outputs_info=outputs_info,
                                    non_sequences=[z] + list(params))
        return nllr

    """
    def call(self, (z, x)):
        # z: input context: n, depth, input_dim
        # x: ngram: n, depth int32
        # output: n, z_depth, window*2
        zflat = T.reshape(z, (z.shape[0]*z.shape[1],z.shape[2]))
        xr = T.transpose(x, (1, 0))
        xshifted = shift_tensor(x)
        xshiftedr = T.transpose(xshifted, (1, 0))
        xsrr = T.repeat(xshiftedr, z.shape[1], axis=1)
        xrr = T.repeat(xr, z.shape[1], axis=1)
        n = z.shape[0]
        outputs_info = [None]
        nllr, _ = theano.scan(self.step, sequences=[xsrr, xrr], outputs_info=outputs_info,
                              non_sequences=[zflat] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0, 2))
        return nll
    """
    def call(self, (z, x)):
        # z: input context: n, depth, input_dim
        # x: ngram: n, depth int32
        # output: n, z_depth, window*2
        zr = T.transpose(z, (1, 0, 2))
        xr = T.transpose(x, (1, 0))
        xshifted = shift_tensor(x)
        xshiftedr = T.transpose(xshifted, (1, 0))
        outputs_info = [None]
        nllr, _ = theano.scan(self.wrapper_step, sequences=[zr], outputs_info=outputs_info,
                              non_sequences=[xshiftedr, xr, self.h0] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0, 2))
        return nll
