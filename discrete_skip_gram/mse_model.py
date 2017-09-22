import csv

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from keras.initializers import RandomUniform
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
                 input_units,
                 activation=leaky_relu,
                 reg_weight_encoding=1e-3,
                 initializer=RandomUniform(minval=-0.05, maxval=0.05),
                 pz_regularizer=None,
                 gen_regularizer=None):
        self.z_k = z_k
        self.activation = activation
        self.generator = generator
        x_input = T.fmatrix(name='x_input')  # (n, input_units)
        n = x_input.shape[0]
        pz = classifier.call(x_input)  # (n, z_k)
        encoding = encoder.call(x_input)  # (n, encoding_units)

        generator_z_embedding = K.variable(initializer((z_k, input_units)))
        assert encoding.ndim == 2
        assert generator_z_embedding.ndim == 2
        zr = T.repeat(generator_z_embedding.dimshuffle(('x', 0, 1)), repeats=n, axis=0)
        er = T.repeat(encoding.dimshuffle((0, 'x', 1)), repeats=z_k, axis=1)
        ctx = T.concatenate((zr, er), axis=2)
        # ctx_units = encoding_units+input_units
        # (n, z_k, input_units+encoding_units)

        blob = generator.call(ctx)  # (n, z_k, input_dim+encodingunits)
        generated = T.nnet.sigmoid(blob[:, :, :input_units])

        mse = T.sum(T.square(generated - (x_input.dimshuffle((0, 'x', 1)))), axis=2)  # (n, z_k)
        loss_mse = T.mean(T.sum(mse * pz, axis=1), axis=0)
        params = ([generator_z_embedding] +
                  generator.params +
                  classifier.params +
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
        loss_reg_gen = T.constant(0)
        if gen_regularizer:
            for w in generator.ws:
                loss_reg_gen += gen_regularizer(w)

        loss_tot = loss_mse + loss_reg_pz + loss_reg_enc + loss_reg_gen
        updates = opt.get_updates(loss_tot, params=params)
        outputs = [loss_mse, loss_reg_pz, loss_reg_enc, loss_reg_gen, loss_tot]
        self.fun_train = theano.function([x_input], outputs, updates=updates)
        self.weights = params + opt.weights

        self.fun_pz = theano.function([x_input], pz)
        zmax = T.argmax(pz, axis=1)  # (n,)
        xgensel = generated[T.arange(n), zmax, :]

        #m = T.mean(generator_z_embedding, axis=1, keepdims=True)
        #s = T.std(generator_z_embedding, axis=1, keepdims=True)
        #proto = T.nnet.sigmoid((generator_z_embedding-m)/(s+1e-9))
        proto = T.nnet.sigmoid(generator_z_embedding)
        xprotosel = proto[zmax, :]
        self.fun_autoencode = theano.function([x_input], [xprotosel, xgensel])

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
                              samples=10):
        n = x.shape[0]
        idx = np.random.random_integers(low=0, high=n - 1, size=(samples,))
        xs = x[idx, :]  # (samples, input_dim)
        proto, ae = self.fun_autoencode(xs)
        img = np.stack((xs, proto, ae), axis=0)  # (3, samples, input_dim)
        img = np.reshape(img, (3, samples, 28, 28))
        img = np.transpose(img, (1, 2, 0, 3))  # (samples,28, 3, 28)
        img = np.reshape(img, (samples * 28, 3 * 28))
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
                # loss_mse, loss_reg_pz, loss_reg_enc, loss_reg_gen, loss_tot
                w.writerow(['Epoch', 'MSE', 'PZ Reg', 'Enc Reg', 'Gen Reg', 'Loss'])
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
                                    'Gen Reg {:.03f}, Loss {:03f}').format(
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
