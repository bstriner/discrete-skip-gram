import timeit

import keras.backend as K
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.wrappers import TimeDistributed

from discrete_skip_gram.layers.time_distributed_dense import TimeDistributedDense
from discrete_skip_gram.layers.time_distributed_dense2 import TimeDistributedDense2


def create_model(dense_fun):
    x = Input((None, 50))
    d, kernel = dense_fun(40)
    y = d(x)
    m = Model(x, y)
    m.compile('adam', 'mse')
    return m, kernel


def main():
    def td1(_units):
        d = Dense(_units)
        return TimeDistributed(d), lambda: d.kernel
    def td2(_units):
        d = TimeDistributedDense(_units)
        return d, lambda: d.kernel
    def td3(_units):
        d = TimeDistributedDense2(_units)
        return d, lambda: d.kernel
    tds = [td1, td2, td3]
    n = 64
    x = np.random.random((n, 25, 50))
    y = np.random.random((n, 25, 40))
    mks = [create_model(td) for td in tds]
    # set same weights
    k0 = K.get_value(mks[0][1]())
    for mk in mks:
        K.set_value(mk[1](), k0)
    ygold = mks[0][0].predict_on_batch(x)
    # confirm same output
    for mk in mks:
        ypred = mk[0].predict_on_batch(x)
        assert np.allclose(ypred, ygold)
    # check timings
    ms = [mk[0] for mk in mks]
    n = 1000
    for i, (m, td) in enumerate(zip(ms, tds)):
        m.predict_on_batch(x)
        pt = timeit.timeit(lambda: m.predict_on_batch(x), number=n)/n
        m.train_on_batch(x, y)
        tt = timeit.timeit(lambda: m.train_on_batch(x, y), number=n)/n
        print "Model {}: predict {} s, train {} s".format(i, pt, tt)


if __name__ == "__main__":
    main()
