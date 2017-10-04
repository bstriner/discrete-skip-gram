import os

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
import numpy as np
from keras.optimizers import Adam

from discrete_skip_gram.flat_validation import run_flat_validation
from discrete_skip_gram.initializers import uniform_initializer
from discrete_skip_gram.reinforce_factored_model import ReinforceFactoredModel
from discrete_skip_gram.util import one_hot_np, logit, softmax_np

def main():
    epochs = 1000
    batches = 4096
    z_k = 256
    inputpath = 'output/skipgram_256-b/b-1.0e-04/iter-0/encodings-00000009.npy'
    outputpath = "output/skipgram_256_finetune_reinforce_factored"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)

    # create initial weights
    enc = np.load(inputpath)
    w = one_hot_np(enc, k=z_k)
    smoothing = 0.1
    w = (w*(1.-smoothing))+(smoothing/z_k)
    w = np.log(w)
    w -= np.max(w, axis=1, keepdims=True)
    print "Weights"
    print w[0,:]
    print (softmax_np(w))[0,:]

    opt = Adam(1e-3)
    initializer = uniform_initializer(0.05)
    model = ReinforceFactoredModel(cooccurrence=cooccurrence,
                                   z_k=z_k,
                                   opt=opt,
                                   initial_pz_weight=w,
                                   initializer=initializer)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)
    return run_flat_validation(input_path=outputpath,
                               output_path=os.path.join(outputpath, "validate.txt"),
                               cooccurrence=cooccurrence)


if __name__ == "__main__":
    main()
