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
