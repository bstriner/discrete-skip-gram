import theano
import theano.tensor as T
from keras.layers import Layer
from keras.engine import InputSpec
from keras import initializers, regularizers
from .utils import W, b, pair, shift_tensor
import keras.backend as K
from ..units.dense_unit import DenseUnit
from ..units.lstm_unit import LSTMUnit
class SkipgramHSMLayer(Layer):
    """
    Given a flattened representation of x, encode as a series of symbols
    """

    def __init__(self,
                 units,
                 code_depth,
                 mean = True,
                 embedding_initializer='RandomUniform',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None):
        self.units = units
        self.code_depth = code_depth
        self.mean = mean
        self.embedding_initializer = initializers.get(embedding_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False
        Layer.__init__(self)

    def build_params(self, input_dim):

        # Embedding step
        y_embedding = self.add_weight((3, self.units),
                            initializer=self.embedding_initializer,
                            name="y_embedding")
        self.embedding_lstm = LSTMUnit(self, self.units, "embedding_lstm")
        self.embedding_d1 = DenseUnit(self, self.units, self.units, "embedding_d1", activation=T.tanh)
        self.embedding_d2 = DenseUnit(self, self.units, self.units, "embedding_d2")
        embedding_params = ([y_embedding] +
                             self.embedding_lstm.non_sequences +
                             self.embedding_d1.non_sequences +
                             self.embedding_d2.non_sequences)
        self.embedding_count = len(embedding_params)

        # Prediction step
        yp_embedding = self.add_weight((3, self.units),
                                      initializer=self.embedding_initializer,
                                      name="yp_embedding")
        self.prediction_lstm = LSTMUnit(self, self.units, "prediction_lstm")
        self.prediction_d1 = DenseUnit(self, self.units, self.units, "prediction_d1", activation=T.tanh)
        self.prediction_d2 = DenseUnit(self, self.units, 1, "prediction_d2", activation=T.nnet.sigmoid)
        prediction_params = ([yp_embedding] +
                                  self.prediction_lstm.non_sequences +
                                  self.prediction_d1.non_sequences +
                                  self.prediction_d2.non_sequences)
        self.prediction_count = len(prediction_params)

        # Main lstm
        z_W = W(self, (input_dim, self.units), "z_W")
        self.outer_lstm = LSTMUnit(self, self.units, "outer_lstm")

        self.non_sequences = ([
            self.embedding_lstm.h0,
            self.prediction_lstm.h0] +
            embedding_params +
            prediction_params +
            [z_W] +
            self.outer_lstm.non_sequences)
        self.built = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 2)
        assert (len(y) == 3)
        input_dim = z[1]
        self.build_params(input_dim)


    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        z = input_shape[0]
        y = input_shape[1]
        assert (len(z) == 2)
        assert (len(y) == 3) # (n, y depth, code depth)
        if self.mean:
            return y[0], 1
        else:
            return y[0], y[1]

    def step_embedding(self, y, h0, *params):
        idx = 0
        y_embedding = params[idx]
        idx += 1
        yh = y_embedding[y,:]
        h1, o1 = self.embedding_lstm.call(h0, yh, params[idx:(idx+self.embedding_lstm.count)])
        idx += self.embedding_lstm.count
        t1 = self.embedding_d1.call(o1, params[idx:(idx+2)])
        idx += 2
        y1 = self.embedding_d2.call(t1, params[idx:(idx+2)])
        return h1, y1

    def step_predict(self, yp0, yp1, h0, *params):
        idx = 0
        yp_embedding = params[idx]
        idx += 1
        yh = yp_embedding[yp0,:]
        h1, o1 = self.prediction_lstm.call(h0, yh, params[idx:(idx+self.prediction_lstm.count)])
        idx += self.prediction_lstm.count
        t1 = self.prediction_d1.call(o1, params[idx:(idx+2)])
        idx += 2
        p1 = self.prediction_d2.call(t1, params[idx:(idx+2)])
        sign = (yp1*2)-1
        nll = T.flatten(-T.log((1-yp1)+(sign*p1)))
        return h1, nll

    def step(self, y0, y1, h0, z, *params):
        #y0 (n, hsm depth) [0-2]
        #y1 (n, hsm depth) [0-1]
        # Unpack parameters
        idx = 0
        embedding_h0 = params[idx]
        idx += 1
        prediction_h0 = params[idx]
        idx += 1
        embedding_params = params[idx:(idx+self.embedding_count)]
        idx += self.embedding_count
        prediction_params = params[idx:(idx+self.prediction_count)]
        idx += self.prediction_count
        z_W = params[idx]
        idx += 1
        outer_lstm_params = params[idx:(idx+self.outer_lstm.count)]
        idx += self.outer_lstm.count
        assert(idx == len(params))


        n = y0.shape[0]

        # Run embedding lstm over input embeddings
        y0r = T.transpose(y0, (1,0)) # hsm depth, n (int32, 0-2)
        outputs_info = [
            T.extra_ops.repeat(embedding_h0, n, axis=0),
            None
        ]
        (_, yembedded), _ = theano.scan(self.step_embedding, sequences=[y0r], outputs_info=outputs_info,
                                        non_sequences=embedding_params)

        # Embedding of Z
        zembedded = T.dot(z, z_W)

        # Run main lstm
        h1, o1= self.outer_lstm.call(h0, yembedded+zembedded, outer_lstm_params)

        # Run prediction lstm
        y1r = T.transpose(y1, (1,0)) # hsm depth, n (int32, 0-1)
        y1s = shift_tensor(y1)
        y1sr = T.transpose(y1s, (1,0))
        outputs_info = [
            T.extra_ops.repeat(prediction_h0, n, axis=0),
            None
        ]
        (_, nllparts), _ = theano.scan(self.step_predict, sequences=[y1r, y1sr], outputs_info=outputs_info,
                                        non_sequences=prediction_params)
        # nll (coding depth, n)
        nll = T.sum(nllparts, axis=0) # (n,)
        return h1, nll

    def call(self, (z, y)):
        # z: input context: n, input_dim
        # y: ngram encoded: n, y depth, hsm depth int32
        yr = T.transpose(y, (1, 0, 2))
        ys = shift_tensor(y)
        ysr = T.transpose(ys, (1, 0, 2))
        n = z.shape[0]
        outputs_info = [T.extra_ops.repeat(self.outer_lstm.h0, n, axis=0),
                        None]
        (hr, nllr), _ = theano.scan(self.step, sequences=[ysr, yr], outputs_info=outputs_info,
                                    non_sequences=[z] + self.non_sequences)
        nll = T.transpose(nllr, (1, 0))
        if self.mean:
            nll = T.mean(nll, axis=1, keepdims=True)
        return nll


class NgramLayerGenerator(Layer):
    def __init__(self, layer, srng, depth):
        self.layer = layer
        self.srng=srng
        self.depth=depth
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = False
        Layer.__init__(self)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.built = True

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return mask

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape[0], self.depth

    def step(self, rng, h0, x0, z, *params):
        (h_W, h_U, h_V, h_b,
         f_W, f_b,
         i_W, i_b,
         c_W, c_b,
         o_W, o_b,
         t_W, t_b,
         y_W, y_b) = params
        h = T.tanh(h_W[x0, :] + T.dot(h0, h_U) + T.dot(z, h_V) + h_b)
        f = T.nnet.sigmoid(T.dot(h, f_W) + f_b)
        i = T.nnet.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.nnet.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(o * h1, t_W) + t_b)
        p1 = T.nnet.softmax(T.dot(t, y_W) + y_b)
        c1 = T.cumsum(p1, axis=1)
        y1 = T.sum(T.gt(rng.dimshuffle((0, 'x')), c1), axis=1) + 1
        y1 = T.cast(y1, 'int32')
        return h1, y1

    def call(self, z):
        # z: input context: n, input_dim
        # rng: rng: n, depth float32

        n = z.shape[0]
        rngr = self.srng.uniform(low=0, high=1, dtype='float32', size=(self.depth, n))
        outputs_info = [T.extra_ops.repeat(self.layer.h0, n, axis=0),
                        T.zeros((n,), dtype='int32')]
        (hr, yr), _ = theano.scan(self.step, sequences=[rngr], outputs_info=outputs_info,
                                  non_sequences=[z] + self.layer.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y
