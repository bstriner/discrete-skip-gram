import numpy as np


def experiment(p, y):
    print "P: {}".format(p)
    print "Y: {}".format(y)
    klp = np.sum(p * np.log(p / y))
    kly = np.sum(y * np.log(y / p))
    print "KL p: {}".format(klp)
    print "KL y: {}".format(kly)


p1 = np.array([0.75, 0.1, 0.15])
p2 = np.array([0.9, 0.05, 0.05])
p3 = np.array([0.05, 0.05, 0.9])
p4 = np.array(np.ones((3,)) / 3)

for p in [p1, p2, p3, p4]:
    for y in [p1, p2, p3, p4]:
        experiment(p, y)


import theano
import theano.tensor as T
x = theano.shared(np.float32(7))
y1 = T.sum(x**3, axis=None)
g1 = T.grad(y1, x)
x2 = x*2
y2 = theano.clone(y1, replace={x:x2})
g2 = T.grad(y2, x)
g3 = theano.clone(g1, replace={x:x2})

_g1 = theano.function([], g1)
_g2 = theano.function([], g2)
_g3= theano.function([], g3)
print "G1: {}, G2: {}, G3: {}".format(_g1(), _g2(), _g3())


print "Testing sort"
x = np.random.random_integers(low=0, high=100, size=(8,))
print x
_x = theano.shared(x)
_y1 = T.argsort(_x)
_y2 = T.sort(_x)
f1 = theano.function([],_y1)
f2 = theano.function([],_y2)
y1 = f1()
y2 = f2()
print y1
print y2