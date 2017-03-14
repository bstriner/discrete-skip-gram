import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import pytest
import sys
import theano
import theano.tensor as T
from discrete_skip_gram.sequence_model import hinge_targets, grid_space
import numpy as np


def test_grid_space():
    y = T.fmatrix()
    g = grid_space(y.shape)
    f = theano.function([y], g)
    _y = np.random.random((2, 3)).astype(np.float32)
    _g = f(_y)
    expected1 = [[0, 0, 0], [1, 1, 1]]
    expected2 = [[0, 1, 2], [0, 1, 2]]
    assert np.allclose(expected1, _g[0])
    assert np.allclose(expected2, _g[1])


def test_hinge_targets():
    y = T.imatrix()
    yt = hinge_targets(y, 4)
    f = theano.function([y], yt)
    _y = np.array([[0, 1, 2], [3, 2, 1]]).astype(np.int32)
    _yt = f(_y)
    expected = [[[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1]],
                [[-1, -1, -1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1]]]
    assert np.allclose(expected, _yt)


if __name__ == "__main__":
    pytest.main(sys.argv[1:])
