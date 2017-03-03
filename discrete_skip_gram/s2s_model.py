from .sequence_model import SequenceModel
from .lstm import LSTM
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from keras.optimizers import RMSprop, Adam
import os
import h5py
import numpy as np

class S2SModel(object):
    def __init__(self, x_k, x_depth, z_k, z_depth, hidden_dim, lr, regularizer=None,
                 encode_deterministic=False, decode_deterministic=True,
                 adversarial_x=False, adversarial_z=False):
        self.x_model = SequenceModel("x_model", x_k, x_depth, hidden_dim, hidden_dim)
        self.z_model = SequenceModel("z_model", z_k, z_depth, hidden_dim, hidden_dim)
        self.x_lstm = LSTM("x_lstm", x_k, x_depth, hidden_dim)
        self.z_lstm = LSTM("z_lstm", z_k, z_depth, hidden_dim)

        x_input = T.imatrix("x_input")
        x_noised_input = T.imatrix("x_noised_input")
        z_input = T.imatrix("z_input")

        srng = RandomStreams(seed=234)

        def encode(x):
            if encode_deterministic:
                return self.z_model.policy_deterministic(self.x_lstm.call(x))
            else:
                rng = srng.uniform(size=(x.shape[0], z_depth), low=0, high=1, dtype='float32')
                return self.z_model.policy(rng, self.x_lstm.call(x))

        def decode(z):
            if decode_deterministic:
                return self.x_model.policy_deterministic(self.z_lstm.call(z))
            else:
                rng = srng.uniform(size=(z.shape[0], x_depth), low=0, high=1, dtype='float32')
                return self.x_model.policy(rng, self.z_lstm.call(z))

        def z_p(x, z):
            return self.z_model.likelihood(z, self.x_lstm.call(x))

        def x_p(x, z):
            return self.x_model.likelihood(x, self.z_lstm.call(z))

        x_gen = decode(z_input)
        z_gen = encode(x_input)

        eps = 1e-8
        x_loss = T.mean(-T.log(eps+x_p(x_noised_input, z_gen)), axis=None)
        if adversarial_x:
            x_loss += T.mean(-T.log(eps+1 - x_p(x_gen, z_input)), axis=None)
        z_loss = T.mean(-T.log(eps+z_p(x_gen, z_input)), axis=None)
        if adversarial_z:
            z_loss += T.mean(-T.log(eps+1 - z_p(x_noised_input, z_gen)), axis=None)

        reg_loss = 0.0
        if regularizer:
            reg_loss += self.x_model.regularization_loss(regularizer)
            reg_loss += self.z_model.regularization_loss(regularizer)
            reg_loss += self.x_lstm.regularization_loss(regularizer)
            reg_loss += self.z_lstm.regularization_loss(regularizer)
        # x_opt = RMSprop(1e-4)
        # z_opt = RMSprop(1e-4)
        # x_updates = x_opt.get_updates(self.x_model.params + self.z_lstm.params, {}, x_loss)
        # z_updates = z_opt.get_updates(self.z_model.params + self.x_lstm.params, {}, z_loss)
        # updates = x_updates + z_updates
        loss = x_loss + z_loss + reg_loss
        opt = Adam(lr)
        self.all_params = self.x_model.params + self.z_lstm.params + \
                          self.z_model.params + self.x_lstm.params
        updates = opt.get_updates(self.all_params, {}, loss)
        self.train_f = theano.function([x_input, z_input, x_noised_input], [x_loss, z_loss], updates=updates)
        self.encode_f = theano.function([x_input], [z_gen])
        self.decode_f = theano.function([z_input], [x_gen])

        x_autoencoded = decode(z_gen)

        self.autoencode_f = theano.function([x_input], [x_autoencoded])
        names = [p.name for p in self.all_params]
        fail = False
        for name in set(names):
            count = names.count(name)
            if count > 1:
                print("Duplicate name: {} x {}".format(name, count))
                fail = True
        if fail:
            raise ValueError("Duplicate name of parameter")

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
        xloss, zloss = self.train_f(x_input, z_input, x_noised_input)
        if not (np.all(np.isfinite(xloss)) and np.all(np.isfinite(zloss))):
            raise ValueError("NaN loss! xloss: {}, zloss: {}".format(xloss, zloss))
        return xloss, zloss

    def autoencode(self, x):
        return self.autoencode_f(x)[0]

    def encode(self, x):
        return self.encode_f(x)[0]

    def decode(self, z):
        return self.decode_f(z)[0]
