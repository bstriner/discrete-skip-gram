import os

os.environ["THEANO_FLAGS"] = "device=cpu,optimizer=None"
from keras.models import Model
from keras.layers import Dense, Input
from theano import tensor as T
import theano
import numpy as np


def main():
    n = 32
    dim_in = 4
    dim_out = 7
    testvar = theano.shared(np.int32(0))
    x = Input((dim_in,))
    d = Dense(dim_out)
    y = d(x)
    updates = [(testvar, testvar+1)]
    d.add_update(updates=updates)
    m = Model(x, y)
    m.compile('adam','mse')

    _x = np.random.random((n, dim_in))
    _y = np.random.random((n, dim_out))
    for i in range(10):
        m.fit(_x, _y, batch_size=8, epochs=1)
        _test = testvar.get_value()
        print "{}: {}".format(i, _test)


if __name__ == "__main__":
    main()
