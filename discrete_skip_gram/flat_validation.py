import numpy as np

from .util import latest_file
from .util import calc_utilization


def validate_encoding_flat(cooccurrence, enc, eps=1e-9):
    _co = cooccurrence.astype(np.float32)
    _co = _co / np.sum(_co, axis=None)
    z_k = np.max(enc, axis=None) + 1
    x_k = _co.shape[0]
    m = np.zeros((z_k, x_k))  # zk, xk
    m[enc, np.arange(x_k)] = 1
    p = np.dot(m, _co)  # (z_k, x_k) * (x_k, x_k) = z_k, x_k
    marg = np.sum(p, axis=1, keepdims=True)
    cond = p / (marg + eps)
    loss = np.asscalar(np.sum(p * -np.log(eps + cond), axis=None))  # scalar
    return loss


def run_flat_validation(input_path, output_path, cooccurrence):
    encoding_path, epoch = latest_file(input_path, "encodings-(\d+).npy")
    if not epoch:
        raise ValueError("No file found at {}".format(input_path))
    print("Epoch {}: {}".format(epoch, encoding_path))
    enc = np.load(encoding_path)
    nll = validate_encoding_flat(cooccurrence=cooccurrence,
                                 enc=enc)
    utilization = calc_utilization(enc)
    with open(output_path, 'w') as f:
        f.write("NLL: {}\n".format(nll))
        f.write("Utilization: {}\n".format(utilization))
    return [nll, utilization]
