
import h5py
import numpy as np

def tst(p):
    with h5py.File(p, 'r') as f:
        ar = []
        for i in f.keys():
            if f[i].ndim > 0:
                ar.append(np.reshape(f[i], (-1,)))
        z = np.concatenate(ar, axis=0)
        print "Min: {}, Max: {}, Avg Abs: {}, Count: {}".format(np.min(z), np.max(z), np.mean(np.abs(z)), z.shape[0])
#        for i, w in enumerate(self.weights):
#            K.set_value(w, f["param_{}".format(i)])

path = r'D:\Projects\discrete-skip-gram\experiments\brown\output\brown\skipgram_sequential_softmax_relu'
import os
tst(os.path.join(path, "model-00000009.h5"))
tst(os.path.join(path, "model-00000019.h5"))
tst(os.path.join(path, "model-00000319.h5"))
tst(os.path.join(path, "model-00000519.h5"))
tst(os.path.join(path, "model-00000759.h5"))
tst(os.path.join(path, "model-00000769.h5"))
tst(os.path.join(path, "model-00000779.h5"))
tst(os.path.join(path, "model-00000789.h5"))
tst(os.path.join(path, "model-00000799.h5"))
