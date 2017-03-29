import theano
import theano.tensor as T
from keras.initializations import glorot_uniform, zero
import keras.backend as K


def W(shape, name):
    return K.variable(glorot_uniform()(shape), name=name)


def b(shape, name):
    return K.variable(zero()(shape), name=name)


def Wb(shape, name):
    return [W(shape, name="{}_W".format(name)), b((shape[1],), name="{}_b".format(name))]


class Decoder(object):
    def __init__(self, k, latent_dim, hidden_dim):
        self.k=k
        self.latent_dim=latent_dim
        self.hidden_dim = hidden_dim
        h_W = W((hidden_dim, hidden_dim), "h_W")
        h_U = W((k + 1, hidden_dim), "h_U")
        h_V = W((latent_dim, hidden_dim), "h_V")
        h_b = b((hidden_dim,), "h_b")

        f_W, f_b = Wb((hidden_dim, hidden_dim), "f")
        i_W, i_b = Wb((hidden_dim, hidden_dim), "i")
        c_W, c_b = Wb((hidden_dim, hidden_dim), "c")
        o_W, o_b = Wb((hidden_dim, hidden_dim), "o")
        t_W, t_b = Wb((hidden_dim, hidden_dim), "t")
        y_W, y_b = Wb((hidden_dim, k), "t")
        self.non_sequences = [h_W, h_U, h_b,
                              f_W, f_b,
                              i_W, i_b,
                              c_W, c_b,
                              o_W, o_b,
                              t_W, t_b,
                              y_W, y_b]

    def p_step(self, x, h0, y0, z,
               h_W, h_U, h_V, h_b,
               f_W, f_b,
               i_W, i_b,
               c_W, c_b,
               o_W, o_b,
               t_W, t_b,
               y_W, y_b
               ):
        h = T.tanh(T.dot(h0, h_W) + h_U[x, :] + T.dot(z, h_V) + h_b)
        f = T.sigmoid(T.dot(h, f_W) + f_b)
        i = T.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(h1 * o, t_W) + t_b)
        y1 = T.softmax(T.dot(t, y_W) + y_b)
        return h1, y1

    def p(self, x, z):
        xr = T.transpose(x, (1, 0))
        n = x.shape[0]
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n, self.k), dtype='float32')]
        (h,yr), _ = theano.scan(self.p_step, sequences=xr, outputs_info=outputs_info,
                                non_sequences=[z] + self.non_sequences)
        y = T.transpose(yr, (1,0, 2))
        return y

    def generate_step(self, rng, h0, y0, z,
               h_W, h_U, h_V, h_b,
               f_W, f_b,
               i_W, i_b,
               c_W, c_b,
               o_W, o_b,
               t_W, t_b,
               y_W, y_b
               ):
        h = T.tanh(T.dot(h0, h_W) + h_U[y0, :] + T.dot(z, h_V) + h_b)
        f = T.sigmoid(T.dot(h, f_W) + f_b)
        i = T.sigmoid(T.dot(h, i_W) + i_b)
        c = T.tanh(T.dot(h, c_W) + c_b)
        o = T.sigmoid(T.dot(h, o_W) + o_b)
        h1 = (h0 * f) + (c * i)
        t = T.tanh(T.dot(h1 * o, t_W) + t_b)
        y1 = T.softmax(T.dot(t, y_W) + y_b)
        cumsum = T.cumsum(y1, axis=1)
        output1 = T.sum(rng.dimshuffle((0,'x')) > cumsum, axis=1) + 1
        output1 = T.cast(output1, 'int32')
        return h1, output1

    def generate(self, rng, z):
        rngr = T.transpose(rng, (1,0))
        n = rng.shape[0]
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n,), dtype='int32')]
        (h, yr), _ = theano.scan(self.generate_step, sequences=rngr, outputs_info=outputs_info,
                                 non_sequences=[z] + self.non_sequences)
        y = T.transpose(yr, (1, 0)) - 1
        return y



class Encoder(object):
    def __init__(self, depth, k, ):


