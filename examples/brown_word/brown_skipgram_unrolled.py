import os

#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from keras.models import Model
from keras.callbacks import LambdaCallback, CSVLogger
from keras import backend as K
import csv
import theano.tensor as T

import itertools
import numpy as np
from discrete_skip_gram.layers.utils import softmax_nd_layer
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, format_encoding_sequential_continuous
from discrete_skip_gram.datasets.word_dataset import docs_to_arrays, skip_gram_batch, \
    skip_gram_ones_generator, WordDataset
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.models.word_skipgram_unrolled import WordSkipgramUnrolled
from discrete_skip_gram.models.util import makepath
from keras.regularizers import L1L2
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer


#maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_unrolled"
    min_count = 5
    batch_size = 128
    epochs = 1000
    steps_per_epoch = 256
    window = 2
    units = 512
    z_k = 2
    z_depth = 10
    #4^6 = 4096
    decay = 0.9
    reg = L1L2(1e-6, 1e-6)
    act_reg = TanhRegularizer(1e-3)
    lr = 3e-4

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    k = ds.k
    schedule = np.power(decay, np.arange(z_depth))
    model = WordSkipgramUnrolled(dataset=ds, z_k=z_k, z_depth=z_depth, schedule=schedule,
                                 window=window,
                                          reg=reg, act_reg=act_reg,
                                          units=units,lr=lr)
    csvpath = "{}/history.csv".format(outputpath)
    makepath(csvpath)
    csvcb = CSVLogger(csvpath)
    validation_n = 4096
    vd = ds.cbow_batch(n=validation_n, window=window, test=True)

    def on_epoch_end(epoch, logs):
        path = "{}/generated-{:08d}.txt".format(outputpath, epoch)
        n = 128
        samples = 8
        _, x = ds.cbow_batch(n=n, window=window, test=True)
        ys = [model.predict_model.predict(x, verbose=0) for _ in range(samples)]
        with open(path, 'w') as f:
            for i in range(n):
                strs= []
                w = ds.get_word(x[i, 0])
                for y in ys:
                    ctx = [ds.get_word(y[i, j]) for j in range(window * 2)]
                    lctx = " ".join(ctx[:window])
                    rctx = " ".join(ctx[window:])
                    strs.append("{} [{}] {}".format(lctx, w, rctx))
                f.write("{}: {}\n".format(w, " | ".join(strs    )))

        if (epoch + 1) % 10 == 0:
            path = "{}/encoded-{:08d}.csv".format(outputpath, epoch)
            x = np.arange(k).reshape((-1, 1))
            zs = model.encode_model.predict(x, verbose=0)
            with open(path, 'w') as f:
                w = csv.writer(f)
                w.writerow(["Idx", "Word", "Encoding"]+["Cat {}".format(i) for i in range(len(zs))])
                for i in range(k):
                    word = ds.get_word(i)
                    enc = [z[i,0] for z in zs]
                    encf = "".join(chr(ord('a')+e) for e in enc)
                    w.writerow([i, word, encf]+enc)
            path = "{}/weights-{:08d}.h5".format(outputpath, epoch)
            model.model.save_weights(path)
#            path = "{}/encoded-array-{:08d}.txt".format(outputpath, epoch)
#            np.save(path, z)

    gencb = LambdaCallback(on_epoch_end=on_epoch_end)
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                callbacks=[csvcb, gencb])


if __name__ == "__main__":
    main()
