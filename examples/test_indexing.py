import theano.tensor as T
import theano
import numpy as np

x = T.ftensor3(name="x")
y = T.ivector(name="y")
t = x[T.arange(x.shape[0]), :, y]
f = theano.function([x,y], t)

_x = np.random.random((3,4,5)).astype(np.float32)
_y = np.random.randint(low=0, high=5, size=(3,))
o=f(_x, _y)
print o.shape
print _x
print _y
print o

