import os

import numpy as np

#from discrete_skip_gram.skipgram.categorical_col_acc import CategoricalColAccModel
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
#from discrete_skip_gram.skipgram.optimizers import AdamOptimizer
#from keras.regularizers import L1L2

import itertools
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"


def main():
    #opt = AdamOptimizer(lr=1e-3)
    epochs = 1000
    batches = 64
    batch_size = 8
    z_k = 1024
    z_depth = 10
    #regularizer = L1L2(1e-10, 1e-10)
    inputpath = "output/skipgram_categorical_col_acc"


    outputpath = "output/skipgram_categorical_col_post"

    type_t = 'float32'
    type_np = np.float32
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)


    ar = itertools.permutations(list(range(z_k)))
    for i, a in enumerate(ar):
        #print "{}: {}".format(i, a)
        pass
    print i


if __name__ == "__main__":
    main()
