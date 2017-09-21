import numpy as np
import theano
import theano.tensor as T


class LanguageModel(object):
    def __init__(self, ngrams, embedding, units, opt):
        self.ngrams = ngrams
        self.embedding = embedding
        h_k = embedding.shape[1]
        ng = T.constant(np.int32(ngrams), name='ngrams')
        e = T.constant(np.float32(embedding), name='embedding')  # (x_k, h)
        x_k = embedding.shape[0]
        self.x_k = x_k
        idx = T.ivector(name='idx')

        ngx = ng[idx, :]  # (n, ng)
        z = e[ngx, :]  # (n, ng, h)

        r1w = theano.shared(np.random.normal(loc=0, scale=0.05, size=(units, units)))
        r1u = theano.shared(np.random.normal(loc=0, scale=0.05, size=(h_k, units)))
        r1b = theano.shared(np.random.normal(loc=0, scale=0.05, size=(units,)))
        r2w = theano.shared(np.random.normal(loc=0, scale=0.05, size=(units, units)))
        r2b = theano.shared(np.random.normal(loc=0, scale=0.05, size=(units,)))
        r3w = theano.shared(np.random.normal(loc=0, scale=0.05, size=(units, x_k)))
        r3b = theano.shared(np.random.normal(loc=0, scale=0.05, size=(x_k,)))
        h0 = theano.shared(np.random.normal(loc=0, scale=0.05, size=(1, units)))
        y0 = theano.shared(np.random.normal(loc=0, scale=0.05, size=(1, h_k)))

        zr = T.transpose(z, (1, 0, 2))
        zrs = T.concatenate((y0, zr[:-1, :, :]), axis=1)
        sequences = [zrs]
        outputs_info = [h0, None]
        non_sequences = [r1w, r1u, r1b, r2w, r2b, r3w, r3b]

        h, pr = theano.scan(self.scan_train,
                           sequences=sequences,
                           outputs_info=outputs_info,
                           non_sequences=non_sequences)
        # pr: (ng, n, x_k)
        p = T.transpose(pr, (1,0,2)) # (n, ng, x_k)
        i0 = T.mgrid[0:p.shape[0], 0:p.shape[1]][0]
        i1 = T.mgrid[0:p.shape[0], 0:p.shape[1]][1]
        i2 = ngx
        px = p[i0, i1, i2]
        eps = 1e-9
        loss = -T.log(eps+px)

        params = [r1w, r1u, r1b, r2w, r2b, r3w, r3b, h0, y0]
        updates = opt.get_updates(loss=loss, params=params)
        self.train_function = theano.function(inputs=[idx], outputs=loss, updates=updates)

    def scan_train(self,
                   # sequences
                   z0,
                   # priors
                   h0,
                   r1w, r1u, r1b, r2w, r2b, r3w, r3b):
        h1 = T.nnet.relu(T.dot(h0, r1w) + T.dot(z0, r1u) + r1b)
        h = T.nnet.relu(T.dot(h1, r2w) + r2b)
        p1 = T.nnet.softmax(T.dot(h, r3w) + r3b)
        return h1, p1

    def train(self, epochs, batches, batch_size):
        idx = np.arange(self.x_k)
        losses = np.zeros((batches,), dtype=np.float32)
        for epoch in range(epochs):
            for batch in range(batches):
                ids = np.random.choice(idx, size=(batch_size,), replace=True)
                losses[batch] = self.train_function(ids)
            batch_loss = np.mean(losses)
