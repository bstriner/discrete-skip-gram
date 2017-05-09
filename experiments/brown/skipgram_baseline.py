# os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu"

import numpy as np

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_baseline import WordSkipgramBaseline
from sample_validation import validation_load
from keras.regularizers import L1L2

# 44sec on laptop w 512 units
# w refactor into units takes 54 secs


# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_baseline"
    dataset = load_dataset()
    vd = validation_load()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 20
    window = 2
    units = 512
    kernel_regularizer = L1L2(1e-9, 1e-9)
    embedding_units = 128
    lr = 1e-3

    model = WordSkipgramBaseline(dataset=dataset,
                                 window=window,
                                 units=units,
                                 kernel_regularizer=kernel_regularizer,
                                 embedding_units=embedding_units,
                                 lr=lr)
    model.summary()
    vn = 4096
    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[0][:vn], vd[1][:vn]], np.ones((vn, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
