# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import csv
import os

import numpy as np
from keras.callbacks import CSVLogger

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset
from discrete_skip_gram.models.util import makepath
from discrete_skip_gram.models.word_skipgram_unrolled_greedy import WordSkipgramUnrolledGreedy
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_unrolled_greedy"
    min_count = 5
    batch_size = 128
    epochs = 25
    steps_per_epoch = 64
    window = 3
    units = 512
    z_k = 4
    z_depth = 6
    # 4^6 = 4096
    decay = 0.9
    # reg = L1L2(1e-6, 1e-6)
    reg = None
    act_reg = TanhRegularizer(1e-3)
    balance_reg = 1e-3
    certainty_reg = 1e-3
    # balance_reg = 0
    # certainty_reg = 0
    lr = 3e-4
    batch = True

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    x_k = ds.k
    model = WordSkipgramUnrolledGreedy(dataset=ds, z_k=z_k, z_depth=z_depth,
                                       window=window,
                                       reg=reg, act_reg=act_reg,
                                       balance_reg=balance_reg,
                                       certainty_reg=certainty_reg,
                                       units=units, lr=lr)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    if os.path.exists(csvpath):
        os.remove(csvpath)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)

    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                output_path=outputpath)

    modelpath = "{}/model.h5".format(outputpath)
    model.models[-1].save_weights(modelpath)

    epath = "{}/encodings.csv".format(outputpath)
    with open(epath, 'wb') as f:
        w = csv.writer(f)
        w.writerow(["ID", "Word"] + ["Cat {}".format(i) for i in range(z_depth)])
        x = np.arange(x_k).reshape((-1, 1))
        zs = model.encode_model.predict(x)
        for i in range(x_k):
            word = ds.get_word(i)
            enc = [z[i, 0] for z in zs]
            w.writerow([i, word] + enc)


if __name__ == "__main__":
    main()
