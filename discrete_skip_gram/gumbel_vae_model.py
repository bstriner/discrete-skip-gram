# adapted from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
import csv
import os

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import tqdm

from discrete_skip_gram.tensor_util import bernoulli_px
from .gumbel_util import gumbel_softmax
from .plot_util import write_image
from .tensor_util import load_latest_weights
from .tensor_util import tensor_one_hot, save_weights
from .util import make_path, generate_batches


class GumbelVaeModel(object):
    def __init__(self,
                 encoder,
                 decoder,
                 opt,
                 tau0,
                 taurate,
                 taumin,
                 K=10,  # number of classes
                 N=30,  # number of categorical distributions
                 eps=1e-9
                 ):
        srng = RandomStreams(123)
        # input image x (shape=(batch_size,784))
        x = T.imatrix(name="x")
        # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
        # unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
        logits_y = T.reshape(encoder.call(x), (-1, K))
        q_y = T.nnet.softmax(logits_y)
        log_q_y = T.log(q_y + eps)
        # temperature
        iteration = theano.shared(0, name='iteration')
        iter_updates = [(iteration, iteration + 1)]
        tau = T.max(T.stack((taumin, tau0 * T.exp(-taurate * iteration))))
        # sample and reshape back (shape=(batch_size,N,K))
        # set hard=True for ST Gumbel-Softmax
        yraw = gumbel_softmax(logits_y, tau, hard=False, srng=srng)
        yflat = T.reshape(yraw, (-1, N * K))
        # generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
        logits_x = decoder.call(yflat)
        # (shape=(batch_size,784))
        p_x = T.nnet.sigmoid(logits_x)
        p_xt = bernoulli_px(p_x, x)

        kl = T.mean(T.sum(q_y * (log_q_y - T.log(1.0 / K)), axis=1), axis=0)

        elbo = T.mean(-T.sum(T.log(eps + p_xt), axis=1), axis=0)
        loss = elbo - kl
        params = encoder.params + decoder.params
        updates = opt.get_updates(loss, params)
        self.train_function = theano.function([x], [elbo, kl, loss, tau], updates=updates + iter_updates)

        ymax = T.argmax(logits_y, axis=1)
        yonehot = T.reshape(tensor_one_hot(ymax, K), (-1, N * K))
        pxval = T.nnet.sigmoid(decoder.call(yonehot))
        pxtval = bernoulli_px(pxval, x)
        valloss = T.mean(-T.sum(T.log(eps + pxtval), axis=1), axis=0)
        self.validate_function = theano.function([x], [elbo, kl, loss, valloss])

        rnd = srng.uniform(low=0., high=1., size=x.shape)
        sampled = T.gt(pxval, rnd)
        self.autoencode_function = theano.function([x], sampled)
        self.tau_function = theano.function([], tau)
        self.weights = params + opt.weights + [iteration]

    def validate(self, xtest, batch_size):
        data = [[] for _ in range(4)]
        for batch in generate_batches(xtest, batch_size):
            datum = self.validate_function(batch)
            for i in range(4):
                data[i].append(datum[i])
        means = [np.mean(d) for d in data]
        return means

    def write_autoencoded(self, path, xtest, samples=10):
        idx = np.random.choice(np.arange(xtest.shape[0]), size=(samples,))
        x = xtest[idx, :]
        xsamp = self.autoencode_function(x)
        x1 = x.reshape((-1, 28, 28))
        x2 = xsamp.reshape((-1, 28, 28))
        img = np.concatenate((x1, x2), axis=2)
        # img = np.transpose(img, (0,2))
        img = np.reshape(img, (samples * 28, 28 * 2))
        img = img.astype(np.float32)
        img = np.clip(img, 0., 1.)
        #write_image(img=img, outputpath=path)
        np.save(path+'.npy', img)

    def train(self,
              output_path,
              epochs,
              batches,
              batch_size,
              xtrain,
              xtest):
        initial_epoch = load_latest_weights(output_path, r'model-(\d+).h5', self.weights)
        if initial_epoch < epochs:
            history_file = os.path.join(output_path, 'history.csv')
            make_path(history_file)
            idx = np.arange(xtrain.shape[0])
            with open(history_file, 'ab') as f:
                w = csv.writer(f)
                w.writerow(["Epoch",
                            "ELBO",
                            "KL",
                            "Loss",
                            "Test ELBO",
                            "Test KL",
                            "Test Loss",
                            "Test Val Loss",
                            "Temperature"])
                for epoch in tqdm(range(epochs), desc="Training"):
                    it = tqdm(range(batches), desc="Epoch {}".format(epoch))
                    data = [[] for _ in range(4)]
                    for batch in it:
                        ids = np.random.choice(idx, size=(batch_size,))
                        batchx = xtrain[ids, :]
                        elbo, kl, loss, tau = self.train_function(batchx)
                        for i, v in enumerate([elbo, kl, loss, tau]):
                            data[i].append(v)
                        it.desc = "Epoch {} ELBO {:.04f} KL {:.04f} Loss {:.04f} Tau {:.04f}".format(
                            epoch,
                            np.asscalar(elbo),
                            np.asscalar(kl),
                            np.asscalar(loss),
                            np.asscalar(tau)
                        )
                    trainelbo, trainkl, trainloss, traintau = [np.mean(d) for d in data]
                    testelbo, testkl, testloss, testvalloss = self.validate(xtest=xtest, batch_size=batch_size)
                    row = [
                        trainelbo, trainkl, trainloss,
                        testelbo, testkl, testloss, testvalloss,
                        self.tau_function()]
                    w.writerow([epoch] + [np.asscalar(r) for r in row])
                    self.write_autoencoded(
                        path=os.path.join(output_path, 'autoencoded-{:08d}.png'.format(epoch)),
                        xtest=xtest)

                    save_weights(os.path.join(output_path, 'model-{:08d}.h5'.format(epoch)), weights=self.weights)
