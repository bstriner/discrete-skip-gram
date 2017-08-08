import numpy as np
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.flat_model import train_model
from discrete_skip_gram.skipgram.regularizers import ExclusiveLasso
from keras.optimizers import Adam
from tqdm import tqdm

from discrete_skip_gram.util import write_csv


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat-el"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for name, weight in tqdm([
        ("1e-7", 1e-7),
        ("1e-8", 1e-8),
        ("1e-9", 1e-9),
        ("5e-10", 5e-10),
        ("1e-10", 1e-10),
        ("7.5e-11", 7.5e-11),
        ("5e-11", 5e-11),
        ("2.5e-11", 2.5e-11),
        ("1e-11", 1e-11),
        ("1e-12", 1e-12),
        ("1e-13", 1e-13)], desc="Meta-iteration"):
        datum = train_model(
            outputpath="{}/{}".format(outputpath, name),
            pz_weight_regularizer=ExclusiveLasso(weight),
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
