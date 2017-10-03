import os

#os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.reinforce_model import ReinforceModel
from discrete_skip_gram.reinforce_parameterizations.highway_parameterization import HighwayParameterization
from discrete_skip_gram.mlp import MLP
from discrete_skip_gram.tensor_util import leaky_relu, softmax_nd


def main():
    # hyperparameters
    epochs = 1000
    batches = 1024
    z_k = 256
    initializer = uniform_initializer(0.0001)
    srng = RandomStreams(123)
    opt = Adam(5e-7)
    activation = leaky_relu
    depth = 1
    units = 1024
    embedding_units = 128
    # build and train
    mlp_p = MLP(input_units=units,
                output_units=z_k,
                hidden_units=units,
                hidden_depth=depth,
                hidden_activation=activation,
                output_activation=softmax_nd
                )
    mlp_h = MLP(input_units=units,
                output_units=units,
                hidden_units=units,
                hidden_depth=depth,
                hidden_activation=activation
                )
    outputpath = "output/skipgram_256_reinforce_highway"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    parameterization = HighwayParameterization(x_k=cooccurrence.shape[0],
                                               z_k=z_k,
                                               mlp_p=mlp_p,
                                               mlp_h=mlp_h,
                                               initializer=initializer,
                                               activation=activation,
                                               embedding_units=embedding_units,
                                               units=units,
                                               srng=srng)
    model = ReinforceModel(cooccurrence=cooccurrence,
                           parameterization=parameterization,
                           z_k=z_k,
                           opt=opt)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
