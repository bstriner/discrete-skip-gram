import theano.tensor as T


class AttributeBarrierRegularizer(object):
    def __init__(self, weight, eps=1e-9):
        self.weight = weight
        self.eps = eps

    def __call__(self, pz, co_m):
        pzw = pz * (co_m.dimshuffle(('x', 0, 1)))  # (a,x,z)
        # pz (a,x,z)
        h = T.dot(T.transpose(pzw, (0, 2, 1)), pz)  # (a, z, a, z)
        assert h.ndim == 4
        reg_loss = -self.weight * T.sum(T.log(self.eps + h), axis=None)
        return reg_loss


class AttributeBarrierUniformRegularizer(object):
    def __init__(self, weight, eps=1e-9):
        self.weight = weight
        self.eps = eps

    def __call__(self, pz, co_m):
        pzw = pz / (pz.shape[1])  # (a,x,z)
        # pz (a,x,z)
        h = T.dot(T.transpose(pzw, (0, 2, 1)), pz)  # (a, z, a, z)
        assert h.ndim == 4
        reg_loss = -self.weight * T.sum(T.log(self.eps + h), axis=None)
        return reg_loss


class AttributeChiSqRegularizer(object):
    def __init__(self, weight, eps=1e-9):
        self.weight = weight
        self.eps = eps

    def __call__(self, pz, co_m):
        # pz: p(z1|x): (a,x,z)
        pzw = pz * (co_m.dimshuffle(('x', 0, 1)))  # p(z1,x): (a,x,z)
        pzj = T.dot(T.transpose(pzw, (0, 2, 1)), pz)  # p(z1, z2) (a, z, a, z)

        pzm = T.sum(pzw, axis=1)  # p(z): (a, z)
        pzmm = (pzm.dimshuffle((0, 1, 'x', 'x'))) * (pzm.dimshuffle(('x', 'x', 0, 1)))  # p(a1z1)p(a2z2): (a,z,a,z)

        d = pzmm - pzj
        reg_loss = self.weight * T.sum(T.square(d))
        return reg_loss
