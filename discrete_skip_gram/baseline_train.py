import csv
import os

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from keras.optimizers import Optimizer
from .tensor_util import save_weights, load_latest_weights
from keras.optimizers import Adam
from .util import make_path
from .baseline_model import BaselineModel


def run_baseline_iterations(z_units,
                            iters,
                            output_path,
                            cooccurrence,
                            epochs,
                            batches,
                            regularizer=None):
    nlls = []
    losses = []
    for i in range(iters):
        iter_path = "{}/iter-{}".format(output_path, z_units, iters)
        model = BaselineModel(
            cooccurrence=cooccurrence,
            z_units=z_units,
            opt=Adam(1e-3),
            regularizer=regularizer
        )
        nll, loss = model.train(outputpath=iter_path,
                                epochs=epochs,
                                batches=batches)
        nlls.append(nll)
        losses.append(loss)
    return np.array(nlls), np.array(losses)


def run_baseline(z_ks,
                 iters,
                 output_path,
                 cooccurrence,
                 epochs,
                 batches,
                 regularizer=None):
    nlls = []
    losses = []
    for z_units in z_ks:
        z_path = "{}/z-{}".format(output_path, z_units)
        nll, loss = run_baseline_iterations(z_units=z_units,
                                            iters=iters,
                                            output_path=z_path,
                                            cooccurrence=cooccurrence,
                                            epochs=epochs,
                                            batches=batches,
                                            regularizer=regularizer)
        nlls.append(nll)
        losses.append(loss)
    np.savez("{}.npz".format(output_path),
             z_ks=np.array(z_ks),
             nlls=np.stack(nlls),
             losses=np.stack(losses))
