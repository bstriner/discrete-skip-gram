import theano
import theano.tensor as T

from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from keras.optimizers import Adam
from ..layers.utils import build_embedding, sigmoid_smoothing, build_bias, leaky_relu
from ..units.mlp_unit import MLPUnit


class SequentialEmbeddingBalancedBinary(Layer):
    """
    Shift dim 2 and pad with learned bias
    """

    def __init__(self,
                 x_k,
                 z_depth,
                 units,
                 srng,
                 batchnorm=True,
                 norm_batch_size=512,
                 inner_activation=leaky_relu,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zero', bias_regularizer=None,
                 embeddings_initializer='random_uniform', embeddings_regularizer=None):
        self.srng = srng
        self.norm_batch_size = norm_batch_size
        self.x_k = x_k
        self.z_depth = z_depth
        self.batchnorm = batchnorm
        self.inner_activation = inner_activation
        self.units = units
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = [InputSpec(ndim=2)]
        self.supports_masking = False
        self.embedding = None
        self.h0 = None
        self.rnn = None
        self.mlp = None
        super(SequentialEmbeddingBalancedBinary, self).__init__()

    def compute_mask(self, inputs, mask=None):
        print ("Compute mask {}".format(mask))
        return [mask, mask]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        self.embedding = build_embedding(self, (self.x_k, self.z_depth), "embedding")
        self.h0 = build_bias(self, (1, self.units), "h0", trainable=False)
        self.rnn = MLPUnit(self, input_units=[self.units, 1], units=self.units, output_units=self.units,
                           hidden_layers=2, inner_activation=self.inner_activation, trainable=False,
                           batchnorm=self.batchnorm, name="rnn")
        self.mlp = MLPUnit(self, input_units=[self.units], units=self.units, output_units=1,
                           hidden_layers=2, inner_activation=self.inner_activation, trainable=False,
                           output_activation=T.nnet.sigmoid,
                           batchnorm=self.batchnorm, name="mlp")
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 1
        return [(input_shape[0], self.z_depth), (input_shape[0], self.z_depth)]

    def step(self, pz, h0, *params):
        # pz (n,)
        # h0 (n, units)
        # sequences, priors, non-sequences
        idx = 0
        rnn_params = params[idx:(idx + self.rnn.count)]
        idx += self.rnn.count
        mlp_params = params[idx:(idx + self.mlp.count)]
        idx += self.mlp.count
        assert idx == len(params)

        # predict median from h0
        pred_z = T.flatten(self.mlp.call([h0], mlp_params))  # (n,)
        # loss is MAE
        z_loss = T.abs_(pred_z - pz)  # (n,)
        # z is relative to median
        z = theano.gradient.zero_grad(T.cast(T.gt(pz, pred_z), 'float32'))  # (n,)
        z_input = T.reshape(z, (-1, 1))
        # h1 incorporates z
        h1 = self.rnn.call([h0, z_input], rnn_params)  # (n, units)
        print "Ndim: {}, {}, {}, {}, {}".format(pz.ndim, h0.ndim, h1.ndim, z.ndim, z_loss.ndim)
        return h1, z, z_loss

    def scan(self, pz):
        # pz_raw: (n, z_depth)
        pzr = T.transpose(pz, (1, 0))  # z_depth, n
        outputs_info = [T.extra_ops.repeat(self.h0, pz.shape[0], axis=0), None, None]
        non_sequences = self.rnn.non_sequences + self.mlp.non_sequences
        (_, zr, z_lossr), _ = theano.scan(self.step,
                                          sequences=[pzr],
                                          outputs_info=outputs_info,
                                          non_sequences=non_sequences)
        # zr (z_depth, n)
        # z_lossr (z_depth, n)
        z = T.transpose(zr, (1, 0))
        z_loss = T.transpose(z_lossr, (1, 0))
        return z, z_loss

    def call(self, inputs, **kwargs):
        # normalization batch
        norm_idx = self.srng.random_integers(low=0, high=self.x_k - 1, size=(self.norm_batch_size,))  # (n,)
        norm_batch = self.embedding[norm_idx, :]  # (n, depth)
        pz_norm = sigmoid_smoothing(T.nnet.sigmoid(norm_batch))  # (n, z_depth)
        _, z_loss = self.scan(pz_norm)
        loss = T.mean(T.sum(z_loss, axis=1), axis=0)
        subopt = Adam(1e-3)
        params = [self.h0] + self.rnn.non_sequences + self.mlp.non_sequences
        updates = subopt.get_updates(params, {}, loss)
        self.add_update(updates=updates)

        # training batch
        x = T.flatten(inputs)  # (n,)
        train_batch = self.embedding[x, :]  # (n, depth)
        pz_train = sigmoid_smoothing(T.nnet.sigmoid(train_batch))  # (n, z_depth)
        z_train, _ = self.scan(pz_train)
        return [pz_train, z_train]
