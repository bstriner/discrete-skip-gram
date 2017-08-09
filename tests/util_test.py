import pytest
import sys
import numpy as np
from discrete_skip_gram.util import stats_string


def test_stats_string():
    x = np.random.random((10,))
    s = stats_string(x)
    assert len(s) > 0


if __name__ == "__main__":
    pytest.main(sys.argv[1:])
