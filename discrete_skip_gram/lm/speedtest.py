import timeit

import numpy as np

from ..util import make_path


def run_speedtest(output_path, train, test, iters, warmup=5):
    print("Timing training")
    for i in range(warmup):
        train()
    train_time = timeit.timeit(train, number=iters) / iters
    print("Timing testing")
    for i in range(warmup):
        test()
    test_time = timeit.timeit(test, number=iters) / iters
    make_path(output_path)
    np.savez(output_path, train_time=train_time, test_time=test_time)
    print("Train time: {}ms".format(train_time * 1000))
    print("Test time: {}ms".format(test_time * 1000))
    return train_time, test_time
