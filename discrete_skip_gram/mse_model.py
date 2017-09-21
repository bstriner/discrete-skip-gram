import csv

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from keras.initializers import RandomUniform
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from tensor_util import leaky_relu
from .plot_util import write_image
from .tensor_util import save_weights, load_latest_weights
from .util import make_path


class MSEModel(object):
    def __init__(self,
                 z_k,
                 classifier,
                 encoder,
                 generator,
                 opt,
                 encoding_units,
                 units=512,
                 activation=leaky_relu,
                 reg_weight_encoding=1e-3,
                 reg_weight_grad=1e-3,
                 initializer=RandomUniform(minval=-0.05, maxval=0.05),
                 pz_regularizer=None):
        self.z_k = z_k
        self.activation = activation
        self.generator = generator
        srng = RandomStreams(123)
        x_input = T.fmatrix(name='x_input')  # (n, input_units)
        n = x_input.shape[0]
        pz = classifier.call(x_input)  # (n, z_k)
        encoding = encoder.call(x_input)  # (n, encoding_units)

        generator_z_embedding = K.variable(initializer((z_k, units)))
        generator_encoding_weight = K.variable(initializer((encoding_units, units)))
        generator_b = K.variable(initializer((units,)))
        generator_ctx = activation((T.dot(encoding, generator_encoding_weight).dimshuffle((0, 'x', 1))) +
                                   (generator_z_embedding.dimshuffle(('x', 0, 1))) +
                                   generator_b)  # (n, z_k, units)
        generated = generator.call(generator_ctx)  # (n, z_k, input_dim)

        mse = T.sum(T.square(generated - (x_input.dimshuffle((0, 'x', 1)))), axis=2)  # (n, z_k)
        loss_mse = T.mean(T.sum(mse * pz, axis=1), axis=0)
        params = ([generator_z_embedding, generator_encoding_weight, generator_b] +
                  generator.params + classifier.params +
                  encoder.params)

        # regularize PZ
        if pz_regularizer:
            loss_reg_pz = pz_regularizer(pz)
        else:
            loss_reg_pz = T.constant(0)

        # regularize encoding
        if reg_weight_encoding > 0:
            loss_reg_enc = reg_weight_encoding * T.mean(T.sum(T.square(encoding), axis=1), axis=0)
        else:
            loss_reg_enc = T.constant(0)

        # regularize generator
        if reg_weight_grad > 0:
            samples = 64
            idx1 = srng.random_integers(low=0, high=n - 1, size=(samples,))
            idx2 = srng.random_integers(low=0, high=n - 1, size=(samples,))
            s1 = encoding[idx1, :]
            s2 = encoding[idx2, :]
            alphas = srng.uniform(low=0, high=1, size=(samples,)).dimshuffle((0, 'x'))
            esamp = (alphas * s1) + ((1 - alphas) * s2)  # (samples, input_units)
            zsamp = srng.random_integers(low=0, high=z_k - 1, size=(samples,))

            sequences = [esamp, zsamp]
            outputs_info = None
            non_sequences = [generator_z_embedding, generator_encoding_weight, generator_b] + generator.params
            g, _ = theano.scan(self.scan_grad,
                               sequences=sequences,
                               outputs_info=outputs_info,
                               non_sequences=non_sequences)
            assert g.ndim == 1

            loss_reg_grad = reg_weight_grad * T.sum(g)
        else:
            loss_reg_grad = T.constant(0)

        loss_tot = loss_mse + loss_reg_pz + loss_reg_enc + loss_reg_grad
        updates = opt.get_updates(loss_tot, params=params)
        outputs = [loss_mse, loss_reg_pz, loss_reg_enc, loss_reg_grad, loss_tot]
        self.fun_train = theano.function([x_input], outputs, updates=updates)
        self.weights = params + opt.weights

        self.fun_pz = theano.function([x_input], pz)
        zmax = T.argmax(pz, axis=1)
        xgensel = generated[T.arange(n), zmax, :]
        self.fun_autoencode = theano.function([x_input], xgensel)

    def scan_grad(self, esamp, zsamp, gz, ge, gb, *gparams):
        ctx = self.activation(T.dot(esamp, ge) + gz[zsamp, :] + gb)
        gen = self.generator.call_on_params(ctx, gparams)  # (input_dim,)
        assert gen.ndim == 1
        g, _ = theano.scan(lambda i, g, e: T.sum(T.square(T.grad(g[i], e))),
                           outputs_info=[None],
                           sequences=[T.arange(gen.shape[0])],
                           non_sequences=[gen, esamp])
        assert g.ndim == 1
        return T.sum(g)

    def visualize_classifier(self,
                             x,
                             output_path,
                             samples=5):
        pz = self.fun_pz(x)  # (n, pz)
        z = np.argmax(pz, axis=1)  # (n,)
        imgs = []
        for i in range(self.z_k):
            idx = np.where(z == i)[0]
            size = min(idx.shape[0], samples)
            if size > 0:
                sel = np.random.choice(idx, size=size, replace=False)
                imz = x[sel, :]
                if size < samples:
                    imz = np.concatenate((imz, np.zeros((samples - size, x.shape[1]))))
                imgs.append(imz)  # (samples, input_units)
            else:
                imgs.append(np.zeros((samples, x.shape[1])))
        img = np.stack(imgs, axis=0)  # (z_k, samples, input_dim)
        img = np.reshape(img, (self.z_k, samples, 28, 28))
        img = np.transpose(img, (0, 2, 1, 3))
        img = np.reshape(img, (self.z_k * 28, samples * 28))
        write_image(img, output_path)

    def visualize_autoencoder(self,
                              x,
                              output_path,
                              columns=2,
                              rows=5):
        n = x.shape[0]
        samples = rows * columns
        idx = np.random.random_integers(low=0, high=n - 1, size=(samples,))
        xs = x[idx, :]  # (samples, input_dim)
        ae = self.fun_autoencode(xs)
        img = np.stack((xs, ae), axis=0)  # (2, samples, input_dim)
        img = np.reshape(img, (2, rows, columns, 28, 28))
        img = np.transpose(img, (1, 3, 2, 0, 4))  # (rows,28, cols, 2, 28)
        img = np.reshape(img, (rows * 28, columns * 28 * 2))
        write_image(img, output_path)

    def train(self,
              x,
              output_path,
              epochs,
              batches,
              batch_size=128):
        assert x.ndim == 2
        initial_epoch = load_latest_weights(output_path, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            it1 = tqdm(range(initial_epoch, epochs), desc='Training')
            histfile = "{}/history.csv".format(output_path)
            make_path(histfile)
            with open(histfile, 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'MSE', 'PZ Reg', 'Enc Reg', 'Grad Reg', 'Loss'])
                f.flush()
                for e in it1:
                    it2 = tqdm(range(batches), desc='Epoch {}'.format(e))
                    data = [[] for _ in range(5)]
                    for b in it2:
                        ids = np.random.random_integers(low=0, high=x.shape[0] - 1, size=(batch_size,))
                        d = self.fun_train(x[ids, :])
                        for a, b in zip(data, d):
                            a.append(b)
                        stats = [np.asscalar(np.mean(a)) for a in data]
                        it2.desc = ('Epoch {}, MSE {:.03f}, PZ Reg {:.03f}, Enc Reg {:.03f}, ' +
                                    'Grad Reg {:.03f}, Loss {:03f}').format(
                            e, *stats
                        )

                    self.visualize_classifier(x, "{}/classifier-{:08d}.png".format(output_path, e))
                    self.visualize_autoencoder(x, "{}/autoencoder-{:08d}.png".format(output_path, e))
                    pz = self.fun_pz(x)
                    np.savetxt("{}/encodings-{:08d}.txt".format(output_path, e), pz)
                    stats = [np.asscalar(np.mean(a)) for a in data]
                    w.writerow([e] + stats)
                    f.flush()
                    save_weights("{}/model-{:08d}.h5".format(output_path, e), self.weights)
