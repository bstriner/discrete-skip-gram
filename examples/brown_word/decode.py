import numpy as np
path = 'output/encoded-array-00000999.txt.npy'

x = np.load(path)
print x.shape

n = x.shape[0]
depth = x.shape[1]
k = x.shape[2]

cats = np.zeros((n, depth, k), dtype='int32')
splits = [np.nonzero(np.isfinite(x[:,:,0]))]
for i in range(depth):
    for j in range(k):
        for split in splits:

        subset = x
        for i2 in range(i+1):
            for j2 in range(j):
                subset = np.nonzero()
        col = x[:, i, j]
        mu = np.mean(col, keepdims=True)
        cats[:,i,j] = (col-mu)>0
        split1 = x[np.nonzero(np.equal(cats,0))]
        split2 = x[np.nonzero(np.equal(cats,1))]
