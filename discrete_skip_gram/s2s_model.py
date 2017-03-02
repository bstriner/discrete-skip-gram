from .sequence_model import SequenceModel
from .lstm import LSTM
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from keras.optimizers import RMSprop
import os
import h5py


class S2SModel(object):
    def __init__(self, x_k, x_depth, z_k, z_depth, hidden_dim):
        self.z_model = SequenceModel("z_model", z_k, z_depth, hidden_dim, hidden_dim)
        self.x_model = SequenceModel("x_model", x_k, x_depth, hidden_dim, hidden_dim)
        self.x_lstm = LSTM("x_lstm", x_k, x_depth, hidden_dim)
        self.z_lstm = LSTM("z_lstm", z_k, z_depth, hidden_dim)

        x_input = T.imatrix("x_input")
        x_noised_input = T.imatrix("x_noised_input")
        z_input = T.imatrix("z_input")

        srng = RandomStreams(seed=234)

        def encode(x):
            rng = srng.uniform(size=(x.shape[0], z_depth), low=0, high=1, dtype='float32')
            return self.z_model.policy(rng, self.x_lstm.call(x))

        def decode(z):
            rng = srng.uniform(size=(z.shape[0], x_depth), low=0, high=1, dtype='float32')
            return self.x_model.policy(rng, self.z_lstm.call(z))

        def z_p(x, z):
            return self.z_model.likelihood(z, self.x_lstm.call(x))

        def x_p(x, z):
            return self.x_model.likelihood(x, self.z_lstm.call(z))

        x_gen = decode(z_input)
        z_gen = encode(x_input)

        x_loss = T.mean(-T.log(x_p(x_noised_input, z_gen)), axis=None) + \
                 T.mean(-T.log(1 - x_p(x_gen, z_input)), axis=None)
        # test w/ x_input and x_noised_input
        z_loss = T.mean(-T.log(1 - z_p(x_noised_input, z_gen)), axis=None) + \
                 T.mean(-T.log(z_p(x_gen, z_input)), axis=None)

        x_opt = RMSprop(1e-4)
        z_opt = RMSprop(1e-4)
        x_updates = x_opt.get_updates(self.x_model.params + self.z_lstm.params, {}, x_loss)
        z_updates = z_opt.get_updates(self.z_model.params + self.x_lstm.params, {}, z_loss)
        updates = x_updates + z_updates
        self.train_f = theano.function([x_input, z_input, x_noised_input], [x_loss, z_loss], updates=updates)
        self.encode_f = theano.function([x_input], [z_gen])
        self.decode_f = theano.function([z_input], [x_gen])

        x_autoencoded = decode(z_gen)

        self.autoencode_f = theano.function([x_input], [x_autoencoded])
        self.all_params = self.x_model.params + self.z_lstm.params + \
                          self.z_model.params + self.x_lstm.params

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with h5py.File(path, "w") as f:
            for p in self.all_params:
                f.create_dataset(p.name, data=p.get_value())

    def load(self, path):
        with h5py.File(path, 'r') as f:
            for p in self.all_params:
                p.set_value(f[p.name][:])

    def train_batch(self, x_input, z_input, x_noised_input):
        return self.train_f(x_input, z_input, x_noised_input)

    def autoencode(self, x):
        return self.autoencode_f(x)[0]

    def encode(self, x):
        return self.encode_f(x)[0]

    def decode(self, z):
        return self.decode_f(z)[0]
