print "Dot numpy"
import os

os.environ["THEANO_FLAGS"] = "optimizer=None,device=cpu"
import numpy as np
import theano.tensor as T
import theano

a = np.random.random((3, 4))
b = np.random.random((4, 5))

print np.dot(a, b)

A = T.constant(a)
B = T.constant(b)
c = T.dot(a, b)
f1 = theano.function([], c)
print f1()

a1 = A.dimshuffle((0, 1, 'x'))
b1 = B.dimshuffle(('x', 0, 1))
c1 = T.sum(a1 * b1, axis=1)
f2 = theano.function([], c1)
print f2()
