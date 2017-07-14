import os

import numpy as np

import keras.backend as K
from discrete_skip_gram.skipgram.categorical_col import CategoricalColModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from keras.regularizers import L1L2


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def val1():
    batch_size = 8
    opt = Adam(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_col"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    z_k = 1
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                regularizer=regularizer,
                                type_np=type_np, type_t=type_t)
    nll, reg_loss, loss = model.validate(batch_size=batch_size)
    print "1) NLL: {}, Reg loss: {}, loss: {}".format(np.asscalar(nll), np.asscalar(reg_loss), np.asscalar(loss))


#    enc = np.zeros((x_k, ))


def val2(weight):
    batch_size = 8
    opt = Adam(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_col"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    z_k = x_k
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                regularizer=regularizer,
                                type_np=type_np, type_t=type_t)
    enc = np.eye(x_k, z_k) * weight
    K.set_value(model.all_weights[0], enc.astype(type_np))
    nll, reg_loss, loss = model.validate(batch_size=batch_size)
    print "2 (weight={}) NLL: {}, Reg loss: {}, loss: {}".format(weight,
                                                                 np.asscalar(nll),
                                                                 np.asscalar(reg_loss),
                                                                 np.asscalar(loss))


def val3(z_k):
    batch_size = 8
    opt = Adam(3e-4)
    regularizer = L1L2(1e-12, 1e-12)
    outputpath = "output/skipgram_categorical_col"
    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    model = CategoricalColModel(cooccurrence=cooccurrence,
                                z_k=z_k,
                                opt=opt,
                                regularizer=regularizer,
                                type_np=type_np, type_t=type_t)
    nll, reg_loss, loss = model.validate(batch_size=batch_size)
    print "3 (z_k={}) NLL: {}, Reg loss: {}, loss: {}".format(z_k,
                                                              np.asscalar(nll),
                                                              np.asscalar(reg_loss),
                                                              np.asscalar(loss))


#    enc = np.zeros((x_k, ))

def main():
    val1()
    val2(weight=1)
    val2(weight=10)
    val2(weight=100)
    val3(z_k=2)
    val3(z_k=32)
    val3(z_k=1024)


if __name__ == "__main__":
    main()
