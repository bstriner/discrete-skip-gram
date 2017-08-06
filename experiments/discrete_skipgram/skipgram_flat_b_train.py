import numpy as np

from discrete_skip_gram.skipgram.flat_model import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.regularizers import BalanceRegularizer
from keras.optimizers import Adam
from discrete_skip_gram.skipgram.util import write_csv
from tqdm import tqdm


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat-b"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for name, weight in tqdm([("1e1", 1e1),
                              ("1e-1", 1e-1),
                              ("1e-2", 1e-2),
                              ("1e-3", 1e-3),
                              ("5e-4", 5e-4),
                              ("1e-4", 1e-4),
                              ("5e-5", 5e-5),
                              ("1e-5", 1e-5),
                              ("1e-6", 1e-6),
                              ("1e-7", 1e-7)], desc='Meta-iteration'):
        datum = train_model(
            outputpath="{}/{}".format(outputpath, name),
            pz_regularizer=BalanceRegularizer(weight),
            cooccurrence=cooccurrence,
            z_k=z_k,
            opt=Adam(1e-3),
            epochs=epochs,
            batches=batches)
        data.append([weight] + datum)
    write_csv('{}.csv'.format(outputpath), rows=data, header=['Weight', 'Nll', 'Utilization'])
    np.save("{}.npy".format(outputpath), data)


if __name__ == "__main__":
    main()
