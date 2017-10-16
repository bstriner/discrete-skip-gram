import timeit

import numpy as np

from ..util import make_path


def run_speedtest(output_path, train, test, iters):
    train_time = timeit.timeit(train, number=iters) / iters
    test_time = timeit.timeit(test, number=iters) / iters
    make_path(output_path)
    np.savez(output_path, train_time=train_time, test_time=test_time)
    print("Train time: {}".format(train_time))
    print("Test time: {}".format(test_time))
