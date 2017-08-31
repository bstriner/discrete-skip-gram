import theano
import theano.tensor as T


def diff2md_scan(depth, tot):
    def scan(*args):
        # sequences, non-sequences
        pos = 0
        idc = args[pos]
        pos += 1
        idcs = args[pos:(pos + depth)]
        x = args[pos]
        pos += 1
        g = args[pos]
        pos += 1
        idx = args[pos:(pos + tot - depth - 1)]
        pos += tot - depth - 1
        assert pos == len(args)
        if idx:
            res, up = theano.scan(diff2_scan(depth + 1, tot), sequences=idx[0], non_sequences=idcs + [x, g] + idx[1:])
            return res
        else:
            idlist = (idc,) + idcs
            g2 = T.grad(cost=g[idlist], wrt=x)[idlist]
            return g2

    return scan


def diff2md(x, y):
    # diff x wrt y
    assert y.ndim == 0
    idx = [T.arange(y.shape[i]) for i in range(y.ndim)]
    g = T.grad(cost=y, wrt=x)
    theano.scan(diff2md_scan, sequences=idx[0], non_sequences=[x, g] + idx[1:])


def diff2_scan(i, x, g):
    print("{},{},{}".format(i, x, g))
    return T.grad(cost=g[i], wrt=x)[i]


def diff2(cost, wrt):
    assert cost.ndim == 0
    assert wrt.ndim == 1
    idx = T.arange(wrt.shape[0])
    g = T.grad(cost=cost, wrt=wrt)
    r, _ = theano.scan(diff2_scan, sequences=[idx], non_sequences=[wrt, g])
    return r
