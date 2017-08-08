import numpy as np
import theano
import theano.tensor as T

from discrete_skip_gram.tensor_util import softmax_nd


class TreeParameterization(object):
    def __init__(self, x_k, z_depth, z_k):
        self.x_k = x_k
        self.z_depth = z_depth
        self.z_k = z_k
        self.params = []
        self.pzs = None
        self.loss = None

    def calc_encoding(self):
        # Discrete encoding
        e0 = T.zeros((self.x_k,), dtype='int32')  # (x_k,)
        encs = []
        for depth in range(self.z_depth):
            pz = self.pzs[depth]  # (x_k, buckets)
            pzt = T.reshape(pz, (self.x_k, -1, self.z_k))  # (x_k, b0, z_k)
            enc = T.argmax(pzt[T.arange(pzt.shape[0]), e0, :], axis=1)  # (x_k,) [int 0-z_k]
            assert enc.ndim == 1
            e1 = (e0 * self.z_k) + enc  # (x_k,) [int 0-b1] todo: double-check order
            e0 = e1
            encs.append(enc)
        encoding = T.stack(encs, axis=1)  # (x_k, z_depth)
        return encoding


class ParameterizationFull(TreeParameterization):
    def __init__(self, x_k, z_depth, z_k, scale=1e-1, **kwargs):
        super(ParameterizationFull, self).__init__(x_k=x_k, z_depth=z_depth, z_k=z_k, **kwargs)
        for depth in range(z_depth):
            buckets = int(z_k ** depth)
            initial_weight = np.random.uniform(-scale, scale, (x_k, buckets, z_k)).astype(self.type_np)
            pz_weight = theano.shared(initial_weight, name="pz_{}".format(depth))  # (x_k, buckets, z_k)
            self.params.append(pz_weight)

        # calculate p(z|x)
        p0 = T.reshape(softmax_nd(self.params[0]), (x_k, z_k))  # (x_k, z_k)
        pzs = [p0]
        for depth in range(1, z_depth):
            p = softmax_nd(self.params[depth])  # (x_k, b0, z_k)
            h = (p0.dimshuffle((0, 1, 'x'))) * p  # (x_k, b0, z_k)
            p1 = T.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))  # (x_k, b1)
            pzs.append(p1)
            p0 = p1
        self.pzs = pzs
        self.encoding = self.calc_encoding()


class ParameterizationReg(TreeParameterization):
    def __init__(self, x_k, z_depth, z_k, scale=1e-1, weight=1e2, **kwargs):
        super(ParameterizationReg, self).__init__(x_k=x_k, z_depth=z_depth, z_k=z_k, **kwargs)
        for depth in range(z_depth):
            buckets = int(z_k ** (depth + 1))
            initial_weight = np.random.uniform(-scale, scale, (x_k, buckets)).astype(self.type_np)
            pz_weight = theano.shared(initial_weight, name="pz_{}".format(depth))  # (x_k, buckets, z_k)
            self.params.append(pz_weight)

        # calculate p(z|x)
        pzs = []
        for depth in range(0, z_depth):
            p = softmax_nd(self.params[depth])  # (x_k, b0)
            pzs.append(p)

        self.pzs = pzs
        weight = T.constant(weight)
        # loss
        self.loss = 0.
        for i0 in range(0, z_depth - 1):
            p0 = pzs[i0]  # (x_k, b0)
            for i1 in range(i0, z_depth):
                p1 = pzs[i1]
                p1r = T.reshape(p1, (x_k, p0.shape[1], -1))  # (x_k, b0, -1)
                p1s = T.sum(p1r, axis=2)  # (x_k, b0)
                l2 = T.sum(T.square(p1s - p0), axis=None)
                self.loss += l2 * weight
        self.encoding = self.calc_encoding()


class ParameterizationSum(TreeParameterization):
    def __init__(self, x_k, z_depth, z_k, scale=1e-1, weight=1e2, **kwargs):
        super(ParameterizationSum, self).__init__(x_k=x_k, z_depth=z_depth, z_k=z_k, **kwargs)
        buckets = int(z_k ** z_depth)
        initial_weight = np.random.uniform(-scale, scale, (x_k, buckets)).astype(self.type_np)
        pz_weight = theano.shared(initial_weight, name="pz_weight")  # (x_k, z_k)
        self.params.append(pz_weight)
        pz = softmax_nd(pz_weight)  # (x_k, z_k)
        # calculate p(z|x)
        pzs = []
        for depth in range(0, z_depth - 1):
            b0 = int(z_k ** (depth + 1))
            h = T.reshape(pz, (x_k, b0, -1))  # (x_k, b0, -1)
            pzt = T.sum(h, axis=2)
            pzs.append(pzt)
        pzs.append(pz)
        self.pzs = pzs
        self.encoding = self.calc_encoding()


class ParameterizationBU(TreeParameterization):
    """
    Bottom-up parameterization
    """

    def __init__(self, x_k, z_depth, z_k, scale=1e-1, weight=1e2, **kwargs):
        super(ParameterizationBU, self).__init__(x_k=x_k, z_depth=z_depth, z_k=z_k, **kwargs)

        # probability of bottom bucket
        buckets = int(z_k ** z_depth)
        initial_weight = np.random.uniform(-scale, scale, (x_k, buckets)).astype(self.type_np)
        pz_weight = theano.shared(initial_weight, name="pz_weight")  # (x_k, z_k)
        self.params.append(pz_weight)
        pz = softmax_nd(pz_weight)  # (x_k, z_k)

        # probability of combining buckets
        pcs = []
        for depth in range(0, z_depth - 1):
            d0 = z_depth - depth
            d1 = d0 - 1
            b0 = int(z_k ** d0)
            b1 = int(z_k ** d1)
            initial_weight = np.random.uniform(-scale, scale, (b0, b1)).astype(self.type_np)
            pc_weight = theano.shared(initial_weight, name="pc_weight_{}_{}".format(d0, d1))  # (b0, b1)
            self.params.append(pc_weight)
            pc = softmax_nd(pc_weight)  # (b0, b1)
            pcs.append(pc)

        # calculate p(z|x)
        pzs = [pz]
        p0 = pz  # (x, b0)
        for depth in range(0, z_depth - 1):
            pc = pcs[depth]  # (b0, b1)
            p1 = T.dot(p0, pc)  # (x, b1)
            pzs.append(p1)
            p0 = p1
        pzs.reverse()
        self.pzs = pzs
        self.encoding = None

    def calc_encoding(self):
        raise NotImplementedError()
