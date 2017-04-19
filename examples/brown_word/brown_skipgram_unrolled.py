# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import csv

import numpy as np
from keras.callbacks import LambdaCallback, CSVLogger
from keras.regularizers import L1L2

from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean
from discrete_skip_gram.datasets.word_dataset import WordDataset
from discrete_skip_gram.models.util import makepath
from discrete_skip_gram.models.word_skipgram_unrolled import WordSkipgramUnrolled
from discrete_skip_gram.models.word_skipgram_unrolled_batch import WordSkipgramUnrolledBatch
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_unrolled"
    min_count = 5
    batch_size = 128
    epochs = 1000
    steps_per_epoch = 256
    window = 3
    units = 512
    z_k = 4
    z_depth = 5
    # 4^6 = 4096
    decay = 0.9
    #reg = L1L2(1e-6, 1e-6)
    reg = None
    act_reg = TanhRegularizer(1e-3)
    balance_reg = 1e-3
    certainty_reg = 1e-3
    #balance_reg = 0
    #certainty_reg = 0
    lr = 3e-4
    batch = False

    docs = clean_docs(brown_docs(), simple_clean)
    docs, tdocs = docs[:-5], docs[-5:]
    ds = WordDataset(docs, min_count, tdocs=tdocs)
    ds.summary()
    k = ds.k
    schedule = np.power(decay, np.arange(z_depth))
    if batch:
        model = WordSkipgramUnrolledBatch(dataset=ds, z_k=z_k, z_depth=z_depth, schedule=schedule,
                                 window=window,
                                 reg=reg, act_reg=act_reg, balance_reg=balance_reg, certainty_reg=certainty_reg,
                                 units=units, lr=lr)
    else:
        model = WordSkipgramUnrolled(dataset=ds, z_k=z_k, z_depth=z_depth, schedule=schedule,
                                 window=window,
                                 reg=reg, act_reg=act_reg, balance_reg=balance_reg, certainty_reg=certainty_reg,
                                 units=units, lr=lr)
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
                strs = []
                w = ds.get_word(x[i, 0])
                for y in ys:
                    ctx = [ds.get_word(y[i, j]) for j in range(window * 2)]
                    lctx = " ".join(ctx[:window])
                    rctx = " ".join(ctx[window:])
                    strs.append("{} [{}] {}".format(lctx, w, rctx))
                f.write("{}: {}\n".format(w, " | ".join(strs)))

        if (epoch + 0) % 5 == 0:
            path = "{}/encoded-{:08d}.csv".format(outputpath, epoch)
            x = np.arange(k).reshape((-1, 1))
            ret = model.encode_model.predict(x, verbose=0)
            pzs, zs = ret[:z_depth], ret[z_depth:]

            #if z_depth == 1:
            #    zs = [zs]
            with open(path, 'wb') as f:
                w = csv.writer(f)
                w.writerow(["Idx", "Word", "Encoding"] +
                           ["Cat {}".format(i) for i in range(len(zs))] +
                           ["Pz {}".format(i) for i in range(len(zs))])
                for i in range(k):
                    word = ds.get_word(i)
                    enc = [z[i, 0] for z in zs]
                    pzfs = [", ".join("{:03f}".format(p) for p in pz[i,:]) for pz in pzs]
                    encf = "".join(chr(ord('a') + e) for e in enc)
                    w.writerow([i, word, encf] + enc + pzfs)
            path = "{}/weights-{:08d}.h5".format(outputpath, epoch)
            model.model.save_weights(path)
        #            path = "{}/encoded-array-{:08d}.txt".format(outputpath, epoch)
        #            np.save(path, z)

    gencb = LambdaCallback(on_epoch_begin=on_epoch_end)
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                callbacks=[csvcb, gencb])


if __name__ == "__main__":
    main()
