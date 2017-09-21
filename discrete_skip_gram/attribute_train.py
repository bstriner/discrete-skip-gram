import numpy as np
from keras.optimizers import Adam

from .attribute_model import AttributeModel


def run_attribute_iterations(zk,
                             ak,
                             iters,
                             z_path,
                             cooccurrence,
                             epochs,
                             batches,
                             pz_regularizer=None):
    nlls = []
    vnlls = []
    for i in range(iters):
        iter_path = "{}/iter-{}".format(z_path, i)
        model = AttributeModel(
            cooccurrence=cooccurrence,
            opt=Adam(1e-3),
            zk=zk,
            ak=ak,
            pz_regularizer=pz_regularizer)
        nll, vnll = model.train(outputpath=iter_path,
                                epochs=epochs,
                                batches=batches)
        nlls.append(nll)
        vnlls.append(vnll)
    nlls = np.stack(nlls, axis=0)  # (n, ak)
    vnlls = np.stack(vnlls, axis=0)  # (n,)
    return nlls, vnlls


