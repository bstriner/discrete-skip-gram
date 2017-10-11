"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""
import os

import numpy as np
from keras.optimizers import Adam
from tqdm import tqdm

from .flat_nn_model import FlatNNModel
from .flat_validation import run_flat_validation
from .util import write_csv


def train_flat_nn_model(outputpath,
                        epochs,
                        batches,
                        cooccurrence,
                        z_k,
                        opt,
                        initializer,
                        pz_regularizer=None,
                        pz_weight_regularizer=None):
    model = FlatNNModel(cooccurrence=cooccurrence,
                        z_k=z_k,
                        opt=opt,
                        initializer=initializer,
                        pz_regularizer=pz_regularizer,
                        pz_weight_regularizer=pz_weight_regularizer)
    model.train(outputpath, epochs=epochs, batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


def train_flat_nn_battery(
        outputpath,
        epochs,
        batches,
        z_k,
        iters,
        cooccurrence,
        initializer,
        pz_regularizer=None,
        pz_weight_regularizer=None
):
    nlls = []
    utilizations = []
    data = []
    for iter in range(iters):
        nll, utilization = train_flat_nn_model(outputpath="{}/iter-{}".format(outputpath, iter),
                                            cooccurrence=cooccurrence,
                                            z_k=z_k,
                                            opt=Adam(1e-3),
                                            epochs=epochs,
                                            batches=batches,
                                               initializer=initializer,
                                            pz_regularizer=pz_regularizer,
                                            pz_weight_regularizer=pz_weight_regularizer)
        nlls.append(nll)
        utilizations.append(utilization)
        data.append([iter, nll, utilization])
    write_csv("{}.csv".format(outputpath), rows=data, header=["Iteration", "NLL", 'Utilization'])
    nlls = np.array(nlls)  # (iters,)
    utilizations = np.array(utilizations)  # (iters,)
    np.savez("{}.npz".format(outputpath),
             nlls=nlls,
             utilizations=utilizations)
    return nlls, utilizations

