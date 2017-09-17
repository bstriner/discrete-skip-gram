import numpy as np
from keras.optimizers import Adam

# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"
from discrete_skip_gram.attribute_model import AttributeModel


def main():
    epochs = 1000
    batches = 2048
    zks = [4] * 5
    outputpath = "output/skipgram_attribute"
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    scale = 1e-2
    opt = Adam(1e-3)
    model = AttributeModel(cooccurrence=cooccurrence,
                           zks=zks,
                           opt=opt,
                           reg_weight=1e-8,
                           scale=scale)
    model.train(outputpath,
                epochs=epochs,
                batches=batches)


if __name__ == "__main__":
    main()
