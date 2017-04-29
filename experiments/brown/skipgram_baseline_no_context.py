import numpy as np

from dataset import load_dataset
from discrete_skip_gram.models.word_skipgram_baseline_no_context import WordSkipgramBaselineNoContext

# maybe try movie_reviews or reuters
def main():
    outputpath = "output/brown/skipgram_baseline_no_context"
    dataset = load_dataset()
    batch_size = 128
    epochs = 5000
    steps_per_epoch = 512
    frequency = 10
    window = 2
    units = 128
    lr = 3e-4

    model = WordSkipgramBaselineNoContext(dataset=dataset,
                                 window=window,
                                 units=units, lr=lr)
    validation_n = 4096
    vd = dataset.cbow_batch(n=validation_n, window=window, test=True)

    model.train(batch_size=batch_size,
                epochs=epochs,
                frequency=frequency,
                steps_per_epoch=steps_per_epoch,
                validation_data=([vd[1], vd[0]], np.ones((validation_n, 1), dtype=np.float32)),
                output_path=outputpath)


if __name__ == "__main__":
    main()
