# import os
# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

from dataset_util import load_dataset
from discrete_skip_gram.layers.utils import leaky_relu
from discrete_skip_gram.skipgram_models.skipgram_discrete_binary_model import SkipgramDiscreteBinaryModel
from keras.optimizers import Adam


def main():
    outputpath = "output/skipgram_discrete_binary"
    ds = load_dataset()
    batch_size = 256
    epochs = 5000
    steps_per_epoch = 2048
    window = 2
    frequency = 10
    units = 512
    embedding_units = 128
    z_depth = 10
    # kernel_regularizer = L1L2(1e-9, 1e-9)
    # embeddings_regularizer = L1L2(1e-9, 1e-9)
    # embeddings_regularizer = None
    loss_weight = 1e-2
    opt = Adam(3e-4)
    layernorm = False
    batchnorm = True
    balancer = False
    dense_batch = False
    do_validate = False
    do_prior = False
    model = SkipgramDiscreteBinaryModel(dataset=ds,
                                        z_depth=z_depth,
                                        window=window,
                                        do_validate=do_validate,
                                        do_prior=do_prior,
                                        balancer=balancer,
                                        embedding_units=embedding_units,
                                        loss_weight=loss_weight,
                                        opt=opt,
                                        layernorm=layernorm,
                                        batchnorm=batchnorm,
                                        inner_activation=leaky_relu,
                                        dense_batch=dense_batch,
                                        units=units)
    model.summary()
    model.train(batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                output_path=outputpath,
                frequency=frequency)


if __name__ == "__main__":
    main()
