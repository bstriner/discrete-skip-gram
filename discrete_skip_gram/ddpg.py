import theano
import theano.tensor as T
from keras.initializers import glorot_uniform, zeros
from keras.optimizers import Adam, RMSprop
import numpy as np
from discrete_skip_gram.constraints import ClipConstraint
from theano.tensor.shared_randomstreams import RandomStreams
import keras.backend as K


def make_W(shape):
    return K.variable(glorot_uniform()(shape), dtype='float32')


def make_b(shape):
    return K.variable(zeros()(shape), dtype='float32')


class Generator(object):
    def __init__(self, latent_dim, hidden_dim, x_k):
        self.hidden_dim = hidden_dim
        Wh = make_W((hidden_dim, hidden_dim))
        Uh = make_W((x_k + 1, hidden_dim))
        Vh = make_W((latent_dim, hidden_dim))
        bh = make_b((hidden_dim,))

        def pair(shape):
            return make_W(shape), make_b((shape[1],))

        Wf, bf = pair((hidden_dim, hidden_dim))
        Wi, bi = pair((hidden_dim, hidden_dim))
        Wc, bc = pair((hidden_dim, hidden_dim))
        Wo, bo = pair((hidden_dim, hidden_dim))
        Wt, bt = pair((hidden_dim, hidden_dim))
        Wy, by = pair((hidden_dim, x_k + 1))
        self.params = [Wh, Uh, Vh, bh,
                       Wf, bf,
                       Wi, bi,
                       Wc, bc,
                       Wo, bo,
                       Wt, bt,
                       Wy, by]

    # sequence, prior, non-sequence
    def step(self, hidden_t0, output_t0, z, *params):
        (Wh, Uh, Vh, bh,
         Wf, bf,
         Wi, bi,
         Wc, bc,
         Wo, bo,
         Wt, bt,
         Wy, by) = params
        h = T.tanh(T.dot(hidden_t0, Wh) + Uh[output_t0, :] + T.dot(z, Vh) + bh)
        f = T.nnet.sigmoid(T.dot(h, Wf) + bf)
        i = T.nnet.sigmoid(T.dot(h, Wi) + bi)
        c = T.tanh(T.dot(h, Wc) + bc)
        hidden_t1 = hidden_t0 * f + i * c
        o = T.nnet.sigmoid(T.dot(h, Wo) + bo)
        output = o * hidden_t1
        t = T.tanh(T.dot(output, Wt) + bt)
        output_t1 = T.argmax(T.dot(t, Wy) + by, axis=-1)
        output_t1 = T.cast(output_t1, "int32")
        return hidden_t1, output_t1

    def __call__(self, z, n_steps):
        n = z.shape[0]
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'), T.zeros((n,), dtype='int32')]
        (h, yr), _ = theano.scan(self.step, sequences=[], non_sequences=[z] + self.params, outputs_info=outputs_info,
                                 n_steps=n_steps)
        y = T.transpose(yr, (1, 0))
        return y


class Discriminator(object):
    def __init__(self, hidden_dim, x_k, constraint):
        self.hidden_dim = hidden_dim
        Wh = make_W((hidden_dim, hidden_dim))
        Uh = make_W((x_k + 1, hidden_dim))
        bh = make_b((hidden_dim,))

        def pair(shape):
            return make_W(shape), make_b((shape[1],))

        Wf, bf = pair((hidden_dim, hidden_dim))
        Wi, bi = pair((hidden_dim, hidden_dim))
        Wc, bc = pair((hidden_dim, hidden_dim))
        Wo, bo = pair((hidden_dim, hidden_dim))
        Wt, bt = pair((hidden_dim, hidden_dim))
        Wy, by = pair((hidden_dim, 1))
        self.params = (Wh, Uh, bh,
                       Wf, bf,
                       Wi, bi,
                       Wc, bc,
                       Wo, bo,
                       Wt, bt,
                       Wy, by)
        Ws = [Wh, Uh, Wf, Wi, Wc, Wo, Wt, Wy]
        self.constraints = {w: constraint for w in Ws}

    def step(self, x_t0, hidden_t0, output_t0, *params):
        (Wh, Uh, bh,
         Wf, bf,
         Wi, bi,
         Wc, bc,
         Wo, bo,
         Wt, bt,
         Wy, by) = params
        h = T.tanh(T.dot(hidden_t0, Wh) + Uh[x_t0, :] + bh)
        f = T.nnet.sigmoid(T.dot(h, Wf) + bf)
        i = T.nnet.sigmoid(T.dot(h, Wi) + bi)
        c = T.tanh(T.dot(h, Wc) + bc)
        hidden_t1 = hidden_t0 * f + i * c
        o = T.nnet.sigmoid(T.dot(h, Wo) + bo)
        output = o * hidden_t1
        t = T.tanh(T.dot(output, Wt) + bt)
        # output_t1 = T.nnet.sigmoid(T.dot(t, Wy) + by)
        output_t1 = T.dot(t, Wy) + by
        return hidden_t1, output_t1[:, 0]

    def __call__(self, input):
        n = input.shape[0]
        inputr = T.transpose(input, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'), T.zeros((n,), dtype='float32')]
        (h, yr), _ = theano.scan(self.step, sequences=[inputr], non_sequences=self.params, outputs_info=outputs_info)
        y = T.transpose(yr, (1, 0))
        return y[:, -1]


class PolicyModel(object):
    def __init__(self, target_shapes):
        self.Ls = []
        self.mus = []
        self.Ps = []
        epsilon = np.float32(1e-6)
        for shape in target_shapes:
            dim = np.prod(shape)
            # print "Dim: {}".format(dim)
            # tril = np.tril_indices(dim)
            # tril_n = tril[0].shape[0]
            # print "tril_n: {}".format(tril_n)
            L = theano.shared(np.random.random((dim,)).astype(np.float32))
            mu = theano.shared(np.random.random((dim,)).astype(np.float32))
            self.Ls.append(L)
            self.mus.append(mu)
            # l = T.zeros((dim, dim), dtype=np.float32)
            # l = T.set_subtensor(l[tril], L)
            # P = T.dot(l, T.transpose(l, (1, 0))) + (epsilon * T.eye(dim))
            P = T.exp(L) + epsilon
            self.Ps.append(P)

        self.b = theano.shared(np.random.random((1,)).astype(np.float32))
        self.params = self.Ls + self.mus + [self.b]

    def value(self, inputs):
        v = self.b[0]
        for i, P, mu in zip(inputs, self.Ps, self.mus):
            # x = T.reshape(i - mu, (1, -1))
            x = T.flatten(i) - mu
            y = -0.5 * T.sum(x * P * x, axis=None)
            v += y
        return v


"""
def flatten_params(params):
    shapes = [p.get_value().shape for p in params]
    total = sum(np.prod(s) for s in shapes)
    x = T.concatenate([T.flatten(p) for p in params], axis=0)
    return x, total
"""


def param_shapes(params):
    return [p.get_value().shape for p in params]


def param_count(params):
    return sum(np.prod(s) for s in param_shapes(params))


class PolicyOptimizer(RMSprop):
    def __init__(self, gradients, *args, **kwargs):
        self.gradients = gradients
        RMSprop.__init__(self, *args, **kwargs)

    def get_gradients(self, loss, params):
        return self.gradients


class DDPG(object):
    def __init__(self,
                 x_k,
                 latent_dim,
                 generator_hidden_dim=512,
                 discriminator_hidden_dim=512,
                 constraint=ClipConstraint(1e-1),
                 lr_d=1e-3,
                 lr_p=1e-4,
                 lr_g=1e-3,
                 gradient_clip=1e-1):
        generator = Generator(latent_dim, generator_hidden_dim, x_k)
        gparams = generator.params
        gshapes = param_shapes(gparams)
        print "Generator params: {}".format(param_count(gparams))
        policy = PolicyModel(gshapes)
        pparams = policy.params
        print "Policy params: {}".format(param_count(pparams))
        # genparams, genparamcount = flatten_params(generator.params)
        # print "Param count: {}".format(genparamcount)
        # print "Tril count: {}".format(np.tril_indices(genparamcount)[0].shape[0])

        discriminator = Discriminator(discriminator_hidden_dim, x_k, constraint)
        dparams = discriminator.params
        print "Discriminator params: {}".format(param_count(dparams))

        # train discriminator
        x_real = T.imatrix()
        y_real = discriminator(x_real)
        rng = RandomStreams(seed=123)
        z = rng.normal((x_real.shape[0], latent_dim), avg=0, std=1, dtype='float32')
        x_fake = generator(z, x_real.shape[1])
        y_fake = discriminator(x_fake)
        # dloss = T.mean(-T.log(1 - y_fake), axis=None) + T.mean(- T.log(y_real), axis=None)
        dloss = T.mean(y_fake, axis=None) - T.mean(y_real, axis=None)
        dopt = RMSprop(lr=lr_d)
        dupdates = dopt.get_updates(dparams, discriminator.constraints, dloss)

        # train policy
        # gloss = T.mean(-T.log(y_fake), axis=None) + T.mean(-T.log(1 - y_real), axis=None)
        gloss = T.mean(y_real, axis=None) - T.mean(y_fake, axis=None)
        greward = -gloss
        gpredicted = policy.value(gparams)
        ploss = T.abs_(greward - gpredicted)
        popt = RMSprop(lr=lr_p)
        pparams = policy.params
        pupdates = popt.get_updates(pparams, {}, ploss)

        # train generator
        goptimal = policy.b[0]
        ggradients = [T.clip(p - T.reshape(mu, p.shape), -gradient_clip, gradient_clip)
                      for mu, p in zip(policy.mus, gparams)]
        # ggradients = [p - T.reshape(mu, p.shape) for mu, p in
        #              zip(policy.mus, gparams)]

        gopt = PolicyOptimizer(ggradients, lr=lr_g)
        gupdates = gopt.get_updates(gparams, {}, None)

        loss = dloss + gloss + ploss
        updates = dupdates + gupdates + pupdates
        self.train_f = theano.function([x_real], [loss, dloss, gloss, ploss], updates=updates)

        input_batch_size = T.iscalar()
        input_depth = T.iscalar()
        z_test = rng.normal((input_batch_size, latent_dim), avg=0, std=1, dtype='float32')
        x_test = generator(z_test, input_depth)
        self.predict_f = theano.function([input_batch_size, input_depth], x_test)

    def train(self, x_real):
        return self.train_f(x_real)

    def predict(self, batch_size, depth):
        return self.predict_f(batch_size, depth)
