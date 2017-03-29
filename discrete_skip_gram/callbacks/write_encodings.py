
from keras.callbacks import Callback
import os
from ..datasets.character_dataset import skip_gram_batch, array_to_word
from ..datasets.utils import format_encoding
class WriteEncodings(Callback):
    def __init__(self, encoder_model, adocs, charset, path_format):
        self.encoder_model = encoder_model
        self.adocs = adocs
        self.charset = charset
        self.path_format = path_format

    def on_epoch_end(self, epoch, logs):
        path = self.path_format.format(epoch)
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        n = 64
        m = 16
        x, y = skip_gram_batch(self.adocs, 0, n)
        zs = [self.encoder_model.predict(x) for _ in range(m)]
        with open(path, 'w') as f:
            for i in range(n):
                word = array_to_word(x[i,:], self.charset)
                encs = ", ".join(format_encoding(z[i,:]) for z in zs)
                f.write("{}: {}\n".format(word, encs))



