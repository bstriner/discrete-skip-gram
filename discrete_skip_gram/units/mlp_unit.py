import itertools

import theano.tensor as T

from .unit_utils import batch_normalization
from ..layers.utils import build_pair, build_kernel, build_bias
from ..layers.utils import layer_norm


class MLPUnit(object):
    def __init__(self,
                 model,
                 input_units,
                 units,
                 output_units,
                 name,
                 hidden_layers=2,
                 inner_activation=T.nnet.relu,
                 layernorm=False,
                 batchnorm=True,
                 output_activation=None):
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.layernorm = layernorm
        self.batchnorm = batchnorm
        input_b = build_bias(model, (units,), "{}_input_b".format(name))
        input_ws = [build_kernel(model, (iu, units), "{}_input_{}_W".format(name, i)) for i, iu in
                    enumerate(input_units)]
        hidden_params = [build_pair(model, (units, units), "{}_hidden_{}".format(name, i)) for i in
                         range(self.hidden_layers)]
        output_params = build_pair(model, (units, output_units), "{}_output".format(name))
        self.non_sequences = ([input_b] + input_ws +
                              list(itertools.chain.from_iterable(hidden_params)) +
                              output_params)

        self.count = len(self.non_sequences)

    def call(self, xs, params):
        assert len(xs) == len(self.input_units)
        idx = 0
        h = params[idx]
        idx += 1
        for x in xs:
            weight = params[idx]
            idx += 1
            h += T.dot(x, weight)
        if self.layernorm:
            h = layer_norm(h)
        if self.batchnorm:
            h = batch_normalization(h)
        h = self.inner_activation(h)
        for i in range(self.hidden_layers):
            weight = params[idx]
            idx += 1
            bias = params[idx]
            idx += 1
            h = T.dot(h, weight) + bias
            if self.layernorm:
                h = layer_norm(h)
            if self.batchnorm:
                h = batch_normalization(h)
            h = self.inner_activation(h)
        weight = params[idx]
        idx += 1
        bias = params[idx]
        idx += 1
        assert idx == len(params)
        y = T.dot(h, weight) + bias
        if self.output_activation:
            y = self.output_activation(y)
        return y
