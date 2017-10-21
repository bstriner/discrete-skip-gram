import theano.tensor as T

def clip_constraint(scale):
    return lambda x: T.clip(x, -scale, scale)