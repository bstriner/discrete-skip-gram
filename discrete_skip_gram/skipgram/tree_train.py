import numpy as np
from tqdm import tqdm
from keras.optimizers import Adam
from .util import write_csv
from .validation import run_tree_validation
from .tree_model import TreeModel


def train_model(outputpath,
                epochs,
                batches,
                cooccurrence,
                z_k,
                z_depth,
                schedule,
                opt,
                pz_regularizer=None,
                pz_weight_regularizer=None):
    model = TreeModel(cooccurrence=cooccurrence,
                      z_k=z_k,
                      z_depth=z_depth,
                      schedule=schedule,
                      opt=opt,
                      pz_regularizer=pz_regularizer,
                      pz_weight_regularizer=pz_weight_regularizer)
    model.train(outputpath, epochs=epochs, batches=batches)
    return run_tree_validation(
        output_path=outputpath,
        input_path=outputpath,
        z_k=z_k)


def train_battery(
        betas,
        epochs,
        iters,
        batches,
        z_k,
        z_depth,
        outputpath,
        pz_regularizer=None,
        pz_weight_regularizer=None):
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    all_nlls = []
    all_utilizations = []
    for beta in betas:
        data = []
        beta_nlls = []
        beta_utilizations = []
        for i in tqdm(range(iters), 'Training iterations'):
            schedule = np.power(beta, np.arange(z_depth))
            schedule /= np.sum(schedule)
            nlls, utilizations = train_model(outputpath="{}/beta-{}/iter-{}".format(outputpath, beta, i),
                                             schedule=schedule,
                                             epochs=epochs,
                                             batches=batches,
                                             cooccurrence=cooccurrence,
                                             z_k=z_k,
                                             z_depth=z_depth,
                                             opt=Adam(1e-3),
                                             pz_regularizer=pz_regularizer,
                                             pz_weight_regularizer=pz_weight_regularizer)
            beta_nlls.append(nlls)
            beta_utilizations.append(utilizations)
            data.append([i] +
                        [nlls[j] for j in range(z_depth)] +
                        [utilizations[j] for j in range(z_depth)])
        all_nlls.append(np.stack(beta_nlls))
        all_utilizations.append(np.stack(beta_utilizations))
        header = (['Iter'] +
                  ['Nll {}'.format(i) for i in range(z_depth)] +
                  ['Utilization {}'.format(i) for i in range(z_depth)])
        write_csv("{}/beta-{}.csv".format(outputpath, beta), data, header=header)
    betas = np.array(betas)
    nlls = np.stack(all_nlls)  # (betas, iters, depth)
    utilizations = np.stack(all_utilizations)  # (betas, iters, depth)
    np.savez("{}.npz".format(outputpath),
             betas=betas,
             nlls=nlls,
             utilizations=utilizations)
    return nlls, utilizations


def train_regularizer_battery(
        betas,
        epochs,
        iters,
        batches,
        z_k,
        z_depth,
        outputpath,
        labels,
        regularizers,
        kwdata,
        is_weight_regularizer):
    assert len(labels) == len(regularizers)
    all_nlls = []
    all_utilizations = []
    for reg in regularizers:
        target_path = "{}/{}".format(outputpath, labels)
        if is_weight_regularizer:
            pz_regularizer = None
            pz_weight_regularizer = reg
        else:
            pz_regularizer = reg
            pz_weight_regularizer = None
        nlls, utilizations = train_battery(betas=betas,
                                           epochs=epochs,
                                           iters=iters,
                                           batches=batches,
                                           z_k=z_k,
                                           z_depth=z_depth,
                                           outputpath=target_path,
                                           pz_regularizer=pz_regularizer,
                                           pz_weight_regularizer=pz_weight_regularizer
                                           )
        all_nlls.append(nlls)
        all_utilizations.append(utilizations)
    nlls = np.stack(all_nlls)  # (regularizers, betas, iters, depth)
    utilizations = np.stack(all_utilizations)  # (regularizers, betas, iters, depth)
    np.savez("{}.npz".format(outputpath),
             betas=betas,
             nlls=nlls,
             utilizations=utilizations,
             **kwdata)
    return nlls, utilizations
