"""
Columnar analysis. Parameters are p(z|x). Each batch is a set of buckets.
"""
from keras.optimizers import Adam
import os

import numpy as np
from tqdm import tqdm

from .flat_model import FlatModel
from .util import write_csv
from .flat_validation import run_flat_validation


def train_flat_model(outputpath,
                     epochs,
                     batches,
                     cooccurrence,
                     z_k,
                     opt,
                     pz_regularizer=None,
                     pz_weight_regularizer=None):
    model = FlatModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      opt=opt,
                      pz_regularizer=pz_regularizer,
                      pz_weight_regularizer=pz_weight_regularizer)
    model.train(outputpath, epochs=epochs, batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


def train_flat_battery(
        outputpath,
        epochs,
        batches,
        z_k,
        iters,
        cooccurrence,
        pz_regularizer=None,
        pz_weight_regularizer=None
):
    nlls = []
    utilizations = []
    data = []
    for iter in range(iters):
        nll, utilization = train_flat_model(outputpath="{}/iter-{}".format(outputpath, iter),
                                            cooccurrence=cooccurrence,
                                            z_k=z_k,
                                            opt=Adam(1e-3),
                                            epochs=epochs,
                                            batches=batches,
                                            pz_regularizer=pz_regularizer,
                                            pz_weight_regularizer=pz_weight_regularizer)
        nlls.append(nll)
        utilizations.append(utilization)
        data.append([iter, nll, utilization])
    write_csv("{}.csv".format(outputpath), rows=data, header=["Iteration", "NLL", 'Utilization'])
    nlls = np.array(nlls)
    utilizations = np.array(utilizations)
    np.savez("{}.npz".format(outputpath),
             nlls=nlls,
             utilizations=utilizations)
    return nlls, utilizations


def train_flat_regularizer_battery(
        outputpath,
        cooccurrence,
        epochs,
        batches,
        iters,
        z_k,
        labels,
        regularizers,
        is_weight_regularizer,
        kwdata
):
    assert len(labels) == len(regularizers)
    nlls = []
    utilizations = []
    for label, reg in tqdm(zip(labels, regularizers)):
        if is_weight_regularizer:
            pz_weight_regularizer = reg
            pz_regularizer = None
        else:
            pz_weight_regularizer = None
            pz_regularizer = reg
        nll, utilization = train_flat_battery(outputpath="{}/{}".format(outputpath, label),
                                              cooccurrence=cooccurrence,
                                              z_k=z_k,
                                              iters=iters,
                                              epochs=epochs,
                                              batches=batches,
                                              pz_regularizer=pz_regularizer,
                                              pz_weight_regularizer=pz_weight_regularizer)
        nlls.append(nll)
        utilizations.append(utilization)
    nlls = np.stack(nlls)
    utilizations = np.stack(utilizations)
    np.savez("{}.npz".format(outputpath),
             nlls=nlls,
             utilizations=utilizations,
             **kwdata)
    return
