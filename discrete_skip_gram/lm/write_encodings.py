import csv
from ..util import make_path

def encoding_to_string(enc):
    return "".join(chr(ord('a') + e) for e in enc)


def write_encodings(output_path, vocab, enc):
    make_path(output_path)
    with open(output_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Word', 'Encoding'])
        for i, v in enumerate(vocab):
            w.writerow([i, v, encoding_to_string(enc[i, :])])
