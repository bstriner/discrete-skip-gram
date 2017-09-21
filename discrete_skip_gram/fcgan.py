import csv

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from tensor_util import leaky_relu
from .plot_util import write_image
from .tensor_util import save_weights, load_latest_weights
from .util import make_path
from keras.initializers import RandomUniform

class FCGAN(object):
    def __init__(self,
                 z_k,
                 classifier,
                 generator,
                 discriminator,
                 optg,
                 optd,
                 input_units,
                 rng_units=256,
                 units=512,
                 activation=leaky_relu,
                 iwgan_weight=1e-3,
                 initializer=RandomUniform(minval=-0.05, maxval=0.05),
                 pz_regularizer=None):
        self.z_k = z_k
        self.discriminator = discriminator
        self.activation = activation
        xreal = T.fmatrix(name='input_units')  # (n, input_units)
        n = xreal.shape[0]
        pz = classifier.call(xreal)  # (n, z_k)

        z_embedding_g = K.variable(initializer((z_k, units)))

        srng = RandomStreams(123)
        rnd = srng.uniform(low=-1, high=1, size=(n, rng_units))
        rng_embedding = K.variable(initializer((rng_units, units)))

        ctx_b = K.variable(initializer((units,)))
        ctx_r = T.dot(rnd, rng_embedding)  # (n, units)

        gen_ctx = (z_embedding_g.dimshuffle(('x', 0, 1))) + (ctx_r.dimshuffle((0, 'x', 1))) + ctx_b  # (n, z_k, units)
        gen_ctx = activation(gen_ctx)  # (n, z_k, units)

        xgen = generator.call(gen_ctx)  # (n, z_k, output_units)
        params_g = [z_embedding_g, rng_embedding, ctx_b] + generator.params + classifier.params

        z_embedding_d = K.variable(initializer( (z_k, units)))
        x_w = K.variable(initializer((input_units, units)))
        ctx_db = K.variable(initializer((units,)))

        ctx_d_gen = (z_embedding_d.dimshuffle(('x', 0, 1))) + T.dot(xgen, x_w) + ctx_db  # (n, z_k, units)
        ctx_d_gen = activation(ctx_d_gen)

        ygen = discriminator.call(ctx_d_gen)  # (n, z_k, 1)
        params_d = [z_embedding_d, x_w, ctx_db] + discriminator.params

        ctx_d_real = (z_embedding_d.dimshuffle(('x', 0, 1))) + (T.dot(xreal, x_w).dimshuffle((0, 'x', 1))) + ctx_db
        ctx_d_real = activation(ctx_d_real)
        yreal = discriminator.call(ctx_d_real)  # (n, z_k, 1)
        assert yreal.ndim == 3
        yreal = yreal[:, :, 0]
        ygen = ygen[:, :, 0]
        ywreal = T.mean(T.sum(yreal * pz, axis=1))  # (n,)
        ywgen = T.mean(T.sum(ygen * pz, axis=1))  # (n,)

        lossg = ywreal - ywgen
        lossd = ywgen - ywreal

        # generator regularization
        if pz_regularizer:
            reg_lossg = pz_regularizer(pz)
        else:
            reg_lossg = T.constant(0)
        lossgtot = lossg + reg_lossg

        # discriminator regularization (IWGAN)
        selrng = srng.uniform(low=0, high=1, size=(n,))
        csum = T.cumsum(pz, axis=1)  # (n, z_k)
        sel = T.cast(T.sum(T.gt(selrng.dimshuffle((0, 'x')), csum), axis=1), 'int32')  # (n,)
        selxgen = xgen[T.arange(n), sel, :]  # (n, input_dim)
        alphas = srng.uniform(low=0, high=1, size=(n,)).dimshuffle((0,'x'))
        assert selxgen.ndim == 2
        assert xreal.ndim == 2
        mixes = (alphas * selxgen) + ((1 - alphas) * xreal)  # (n, input_units)
        g1, _ = theano.scan(self.scan_grad, sequences=[mixes, sel], non_sequences=params_d, outputs_info=[None])
        g2, _ = theano.scan(self.scan_grad, sequences=[xreal, sel], non_sequences=params_d, outputs_info=[None])
        g3, _ = theano.scan(self.scan_grad, sequences=[selxgen, sel], non_sequences=params_d, outputs_info=[None])
        # g: (n,) l2 of grads
        assert g1.ndim == 1
        assert g2.ndim == 1
        assert g3.ndim == 1
        reg_lossd = iwgan_weight * (
            T.mean(T.square(1. - g1), axis=None) +
            T.mean(T.square(1. - g2), axis=None) +
            T.mean(T.square(1. - g3), axis=None) ) / 3
        lossdtot = lossd + reg_lossd
        assert lossgtot.ndim == 0
        assert lossdtot.ndim == 0
        gup = optg.get_updates(lossgtot, params_g)
        dup = optd.get_updates(lossdtot, params_d)
        outputs = [lossg, reg_lossg, lossd, reg_lossd]
        self.traing = theano.function([xreal], outputs, updates=gup)
        self.traind = theano.function([xreal], [], updates=dup)
        self.weights = params_d + params_g + optd.weights + optg.weights

        self.fun_pz = theano.function([xreal], pz)

        z_input = T.ivector()  # (n,)
        zrnd = srng.uniform(low=-1, high=1, size=(z_input.shape[0], rng_units))
        zctx = T.dot(zrnd, rng_embedding) + z_embedding_g[z_input, :] + ctx_b
        zctx = activation(zctx)
        zgen = generator.call(zctx)
        self.fun_gen = theano.function([z_input], zgen)

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

    def visualize_generator(self,
                            output_path,
                            samples=5):
        zs = np.arange(self.z_k)
        zs = np.repeat(np.reshape(zs, (-1, 1)), axis=1, repeats=samples)
        zin = np.reshape(zs, (-1,))
        img = self.fun_gen(zin)
        img = np.reshape(img, (self.z_k, samples, 28, 28))
        img = np.transpose(img, (0, 2, 1, 3))
        img = np.reshape(img, (self.z_k * 28, samples * 28))
        write_image(img, output_path)

    def scan_grad(self, x, z, *params):
        ze = params[0]
        x_w = params[1]
        ctx_db = params[2]
        dparams = params[3:]

        ctx = T.dot(x, x_w) + ze[z, :] + ctx_db
        ctx = self.activation(ctx)
        y = self.discriminator.call_on_params(ctx, dparams)
        assert y.ndim == 1

        g = T.grad(y[0], x)
        g2 = T.sum(T.square(g), axis=None)  # l2
        return g2

    def train(self,
              x,
              output_path,
              epochs,
              batches,
              discriminator_batches,
              batch_size=128):
        assert x.ndim == 2
        initial_epoch = load_latest_weights(output_path, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            it1 = tqdm(range(initial_epoch, epochs), desc='Training')
            histfile = "{}/history.csv".format(output_path)
            make_path(histfile)
            with open(histfile, 'ab') as f:
                w = csv.writer(f)
                w.writerow(['Epoch', 'G Loss', 'G Reg', 'D Loss', 'D Reg'])
                f.flush()
                for e in it1:
                    it2 = tqdm(range(batches), desc='Epoch {}'.format(e))
                    data = [[] for _ in range(4)]
                    for b in it2:
                        for i in range(discriminator_batches):
                            ids = np.random.random_integers(low=0, high=x.shape[0] - 1, size=(batch_size,))
                            self.traind(x[ids, :])
                        ids = np.random.random_integers(low=0, high=x.shape[0] - 1, size=(batch_size,))
                        d = self.traing(x[ids, :])
                        for a, b in zip(data, d):
                            a.append(b)
                        stats = [np.asscalar(np.mean(a)) for a in data]
                        it2.desc = 'Epoch {}, gloss {:.03f}, greg {:.03f}, dloss {:.03f}, dreg {:.03f}'.format(
                            e, *stats
                        )

                    self.visualize_classifier(x, "{}/classifier-{:08d}.png".format(output_path, e))
                    self.visualize_generator("{}/generator-{:08d}.png".format(output_path, e))
                    stats = [np.asscalar(np.mean(a)) for a in data]
                    w.writerow([e] + stats)
                    f.flush()
                    save_weights("{}/model-{:08d}.h5".format(output_path, e), self.weights)
                    pz = self.fun_pz(x)
                    np.savetxt("{}/encodings-{:08d}.txt".format(output_path, e), pz)
