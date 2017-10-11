import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import numpy as np
import theano.tensor as T
from keras.optimizers import Adam

from discrete_skip_gram.gumbel_vae_model import GumbelVaeModel
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.mlp2 import MLP2


def main():
    epochs = 1000
    batches = 2048
    batch_size = 64
    tau0 = 5.
    taurate = 3e-6
    taumin = 0.5
    output_path = '../output/mnist/gumbel_vae'
    data = np.load('../output/mnist/mnist.npz')
    xtrain, xtest = data["xtrain"], data["xtest"]
    initializer = uniform_initializer(0.05)
    K = 10  # number of classes
    N = 20  # number of categorical distributions
    encoder = MLP2(input_units=28 * 28,
                   units=[512, 256, K * N],
                   hidden_activation=T.nnet.relu,
                   initializer=initializer
                   )
    decoder = MLP2(input_units=K * N,
                   units=[256, 512, 28 * 28],
                   hidden_activation=T.nnet.relu,
                   initializer=initializer
                   )
    model = GumbelVaeModel(
        encoder=encoder,
        decoder=decoder,
        opt=Adam(1e-3),
        K=K,
        N=N,
        tau0=tau0,
        taumin=taumin,
        taurate=taurate
    )
    model.train(output_path=output_path,
                epochs=epochs,
                batches=batches,
                batch_size=batch_size,
                xtrain=xtrain,
                xtest=xtest)


if __name__ == '__main__':
    main()
