import numpy as np
# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from keras.optimizers import Adam
from discrete_skip_gram.attribute_regularizers import AttributeBarrierRegularizer
from discrete_skip_gram.attribute_model import AttributeModel


def main():
    epochs = 10
    batches = 50000
    outputpath = "output/skipgram_attribute_b"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-2
    opt = Adam(1e-3)
    reg = AttributeBarrierRegularizer(1e-2)
    model = AttributeModel(cooccurrence=cooccurrence,
                           zk=4,
                           ak=5,
                           opt=opt,
                           pz_regularizer=reg,
                           scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)


if __name__ == "__main__":
    main()
