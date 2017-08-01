import os

import numpy as np

from discrete_skip_gram.skipgram.categorical_col import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from discrete_skip_gram.skipgram.util import write_csv


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    opt = Adam(1e-3)
    epochs = 10
    iters = 10
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for i in range(iters):
        datum = train_model(outputpath="{}/iter-{}".format(output_path, i),
                            epochs,
                            batches,
                            cooccurrence,
                            z_k,
                            Adam(1e-3))
        data.append([i] + datum)
    header = ['Iter', 'Nll', 'Utilization']
    write_csv("output/skipgram_flat.csv", data, header=header)


if __name__ == "__main__":
    main()
