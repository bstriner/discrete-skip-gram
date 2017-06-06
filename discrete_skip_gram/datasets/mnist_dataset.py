import numpy as np
from keras.datasets import mnist


def mnist_clean((x,y)):
    return x, y.reshape((-1,1))

def mnist_data():
    train, test = mnist.load_data()
    return mnist_clean(train), mnist_clean(test)


def mnist_val():
    train, test = mnist.load_data()
    return list(test), np.zeros((test[0].shape[0], 1))


def mnist_generator(batch_size):
    (tx, ty), test = mnist_data()
    while True:
        idx = np.random.randint(0, tx.shape[0], (batch_size,))
        x = tx[idx, :]
        y = ty[idx,:]
        target = np.zeros((batch_size, 1))
        batch = [x, y], target
        yield batch
