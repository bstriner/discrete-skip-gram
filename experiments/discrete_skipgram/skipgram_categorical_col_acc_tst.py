import numpy as np

from discrete_skip_gram.skipgram.cooccurrence import load_cooccurrence
from discrete_skip_gram.skipgram.util import latest_file


# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

def main():
    epochs = 1000
    batches = 64
    batch_size = 8
    z_k = 1024
    inputpath = "output/skipgram_categorical_col_acc"
    type_t = 'float32'
    type_np = np.float32
    cooccurrence = load_cooccurrence('output/cooccurrence.npy').astype(np.float32)
    file, epoch = latest_file(inputpath, "probabilities-(\d+).npy")
    if not epoch:
        raise ValueError("File not found")
    #f1 = "output/skipgram_categorical_col_acc/probabilities-00000214.npy"
    print "Using epoch {}: {}".format(epoch, file)
    p = np.load(file)
    print p.shape
    n = p.shape[0]
    k = p.shape[1]
    m = np.argmax(p, axis=1)
    used = set(m[i] for i in range(n))
    print "used: {}/{}".format(len(used), k)
    #for i in range(k):
    #    if i not in used:
    #        print "unused: {}".format(i)

    ma = np.max(p, axis=0)
    print ma.shape
    print ma[1023]
    print ma[1022]
    print ma[1021]
    print p[:, 1023]
    print np.argmax(p[:, 1023])
    print p[1789, 1023]


if __name__ == "__main__":
    main()
