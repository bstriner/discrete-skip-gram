import os

import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.util import write_csv
from keras.optimizers import Adam
from tqdm import tqdm
from discrete_skip_gram.skipgram.flat_model import train_model
from keras.regularizers import l1

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat-l1"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for name, weight in tqdm([
        ("1e-4", 1e-4),
        ("1e-5", 1e-5),
        ("1e-6", 1e-6),
        ("1e-7", 1e-7),
        ("5e-8", 5e-8),
        ("1e-8", 1e-8),
        ("5e-9", 5e-9),
        ("1e-9", 1e-9)
    ]):
        datum = train_model(outputpath="{}/{}".format(outputpath, name),
                            cooccurrence=cooccurrence,
                            z_k=z_k,
                            opt=Adam(1e-3),
                            epochs=epochs,
                            batches=batches,
                            pz_weight_regularizer=l1(weight))
        data.append([weight] + datum)
    write_csv("{}.csv".format(outputpath), rows=data, header=["Weight", "NLL", 'Utilization'])
    np.save("{}.npy".format(outputpath), data)


if __name__ == "__main__":
    main()
