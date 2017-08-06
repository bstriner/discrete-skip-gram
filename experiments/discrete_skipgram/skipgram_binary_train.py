import os

import numpy as np

from discrete_skip_gram.skipgram.flat_model import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from keras.optimizers import Adam
from discrete_skip_gram.skipgram.util import write_csv
from tqdm import tqdm

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    iters = 10
    batches = 4096
    z_k = 2
    outputpath = "output/skipgram_binary"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for i in tqdm(range(iters), 'Training iterations'):
        datum = train_model(outputpath="{}/iter-{}".format(outputpath, i),
                            epochs=epochs,
                            batches=batches,
                            cooccurrence=cooccurrence,
                            z_k=z_k,
                            opt=Adam(1e-3))
        data.append([i] + datum)
    header = ['Iter', 'Nll', 'Utilization']
    write_csv("{}.csv".format(outputpath), data, header=header)
    np.save("{}.npy".format(outputpath), np.array(data))


if __name__ == "__main__":
    main()
