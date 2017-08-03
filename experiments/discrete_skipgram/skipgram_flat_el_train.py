import numpy as np

from discrete_skip_gram.skipgram.categorical_col import train_model
from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.regularizers import ExclusiveLasso
from discrete_skip_gram.skipgram.util import write_csv
from keras.optimizers import Adam
from tqdm import tqdm

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 10
    batches = 4096
    z_k = 1024
    outputpath = "output/skipgram_flat-el"
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    data = []
    for name, weight in tqdm([
        ("1e7", 1e-7),
        ("1e8", 1e-8),
        ("1e9", 1e-9),
        ("1e10", 1e-10),
        ("1e11", 1e-11)], desc="Meta-iteration"):
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


if __name__ == "__main__":
    main()
