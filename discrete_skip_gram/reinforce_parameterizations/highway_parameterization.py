import keras.backend as K
import theano
import theano.tensor as T


class HighwayParameterization(object):
    def __init__(self,
                 x_k,
                 z_k,
                 mlp_p,
                 mlp_h,
                 initializer,
                 activation,
                 srng,
                 embedding_units=128,
                 units=512):
        self.z_k = z_k
        self.mlp_p = mlp_p
        self.mlp_h = mlp_h
        self.activation = activation
        x_embeddings = K.variable(initializer(x_k, embedding_units))
        h0 = K.variable(initializer(units, ))
        wph = K.variable(initializer(units, units))
        wpx = K.variable(initializer(embedding_units, units))
        wpb = K.variable(initializer(units, ))
        whh = K.variable(initializer(units, units))
        whx = K.variable(initializer(embedding_units, units))
        whz = K.variable(initializer(embedding_units, units))
        whb = K.variable(initializer(units, ))
        z_embeddings = K.variable(initializer(z_k, embedding_units))

        rnd = srng.uniform(low=0., high=1., dtype='float32', size=(x_k,))
        sequences = [x_embeddings, rnd]
        outputs_info = [h0, None, None]
        non_sequences = ([wph, wpx, wpb, whh, whx, whz, whb, z_embeddings] +
                         self.mlp_p.params +
                         self.mlp_h.params)
        (h, z, logpz), _ = theano.scan(self.scan,
                                       sequences=sequences,
                                       outputs_info=outputs_info,
                                       non_sequences=non_sequences)
        self.outputs = (z, logpz)

    def scan(self, x, rnd, h0, wph, wpx, wpb, whh, whx, whz, whb, z_embeddings, *params):
        assert len(params) == len(self.mlp_p.params) + len(self.mlp_h.params)
        params_p = params[:len(self.mlp_p.params)]
        params_h = params[len(self.mlp_p.params):]

        ctx_p = self.activation(T.dot(h0, wph) + T.dot(x, wpx) + wpb)
        pz = self.mlp_p.call_on_params(ctx_p, params_p)
        assert pz.ndim == 1
        cs = T.cumsum(pz, axis=0)
        sel = T.sum(T.gt(rnd, cs))
        sel = T.clip(sel, 0, self.z_k - 1)
        lpz = T.log(pz[sel])

        ze = z_embeddings[sel, :]

        ctx_h = self.activation(T.dot(h0, whh) + T.dot(x, whx) + T.dot(ze, whz) + whb)
        hd = self.mlp_h.call_on_params(ctx_h, params_h)
        h1 = h0 + hd
        return h1, sel, lpz
