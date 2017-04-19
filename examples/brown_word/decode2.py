import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np
path = 'output/encoded-array-00000999.txt.npy'
import csv


from keras.optimizers import Adam
from discrete_skip_gram.layers.utils import softmax_nd_layer
from discrete_skip_gram.datasets.utils import clean_docs, simple_clean, get_words, count_words, \
    format_encoding_sequential_continuous
from discrete_skip_gram.datasets.word_dataset import docs_to_arrays, skip_gram_generator, skip_gram_batch, \
    skip_gram_ones_generator, WordDataset
from discrete_skip_gram.datasets.corpus import brown_docs
from discrete_skip_gram.models.word_ngram_sequential_continuous import WordNgramSequentialContinuous
from discrete_skip_gram.models.util import makepath
from keras.regularizers import L1L2
from discrete_skip_gram.regularizers.tanh_regularizer import TanhRegularizer

min_count = 5
batch_size = 128
epochs = 1000
steps_per_epoch = 256
window = 3
hidden_dim = 512
z_k = 2
z_depth = 6
#4^6 = 4096
decay = 0.9
#reg = L1L2(1e-6, 1e-6)
reg = None
#act_reg = TanhRegularizer(1e-3)
act_reg = None
lr = 1e-3 #3e-4

docs = clean_docs(brown_docs(), simple_clean)
docs, tdocs = docs[:-5], docs[-5:]
ds = WordDataset(docs, min_count, tdocs=tdocs)
ds.summary()

class X(object):
    def __init__(self, id, val):
        self.id=id
        self.val=val
        self.enc = []

a = np.load(path)
print a.shape
xs = []
for i in range(a.shape[0]):
    xs.append(X(i, a[i:i+1,:,:]))

n = a.shape[0]
depth = a.shape[1]
k = a.shape[2]

splits = [xs]
for i in range(depth):
    for j in range(k):
        newsplits = []
        for split in splits:
            print "Split: {}".format(len(split))
            d = np.concatenate([_x.val for _x in split], axis=0)[:,i,j]
            mu = np.mean(d, axis=0)
            split1 = []
            split2 = []
            for x in split:
                if x.val[0, i, j] > mu:
                    x.enc.append(0)
                    split1.append(x)
                else:
                    x.enc.append(1)
                    split2.append(x)
            if len(split1) > 0:
                newsplits.append(split1)
            if len(split2) > 0:
                newsplits.append(split2)
        splits = newsplits

for x in xs:
    e = x.enc
    e2 = []
    for i in range(len(e)/k):
        b = np.array([e[i] for i in range(i*k, i*k+k)])
        p = np.power(2, np.arange(k))
        e2.append(np.sum(b*p))
    x.enc=e2

with open('test.csv','w') as f:
    w = csv.writer(f)
    w.writerow(["Id","Word","Encoding"])
    for x in xs:
        w.writerow([x.id, ds.get_word(x.id), "".join("{}".format(e) for e in x.enc)])