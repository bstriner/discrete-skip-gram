import pickle

import numpy as np

from discrete_skip_gram.visualization.tree import build_js


def main():
    enc_path = 'output/encodings-00000009.npy'
    output_path = 'web/vocab.js'
    vocab_path = 'output/corpus/vocab.pkl'
    tpl_path = 'web/vocab.js.mako'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    enc = np.load(enc_path)
    z_k = 2
    js = build_js(enc=enc, vocab=vocab, tpl_path=tpl_path, z_k=z_k)
    with open(output_path, 'w') as f:
        f.write(js)

if __name__=="__main__":
    main()