import numpy as np
from discrete_skip_gram.tree_validation import validate_encoding_tree
from tqdm import tqdm

from discrete_skip_gram.util import write_csv


def random_tree_baseline(iters, z_k, z_depth, x_k, cooccurrence):
    nlls = []
    utilizations = []
    for _ in tqdm(range(iters), desc="Random tree baseline"):
        encoding = np.random.random_integers(0, z_k - 1, (x_k, z_depth))
        nll, utilization = validate_encoding_tree(cooccurrence=cooccurrence, encoding=encoding, z_k=z_k)
        nlls.append(nll)
        utilizations.append(utilization)
    return np.stack(nlls), np.stack(utilizations)


def main():
    iters = 50
    z_k = 2
    z_depth = 10
    cooccurrence = np.load('output/cooccurrence.npy').astype(np.float32)
    x_k = cooccurrence.shape[0]
    output_path = "output/skipgram_tree_random"
    nlls, utilizations = random_tree_baseline(iters=iters,
                                              z_k=z_k,
                                              z_depth=z_depth,
                                              x_k=x_k,
                                              cooccurrence=cooccurrence)
    header = (['Iteration'] +
              ['NLL {}'.format(i) for i in range(iters)] +
              ['Utilization {}'.format(i) for i in range(iters)])
    data = []
    for i in range(iters):
        data.append([i] +
                    [nlls[i, j] for j in range(z_depth)] +
                    [utilizations[i, j] for j in range(z_depth)])
    write_csv("{}.csv".format(output_path), rows=data, header=header)
    np.savez("{}.npz".format(output_path),
             nlls=nlls,
             utilizations=utilizations)


if __name__ == "__main__":
    main()
