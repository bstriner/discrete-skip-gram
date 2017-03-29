def W(model, shape, name):
    return model.add_weight(shape,
                           initializer=model.kernel_initializer,
                           name=name,
                           regularizer=model.kernel_regularizer)


def b(model, shape, name):
    return model.add_weight(shape,
                           initializer=model.bias_initializer,
                           name=name,
                           regularizer=model.bias_regularizer)


def pair(model, shape, name):
    return W(model, shape, "{}_W".format(name)), b(model, (shape[1],), "{}_b".format(name))
