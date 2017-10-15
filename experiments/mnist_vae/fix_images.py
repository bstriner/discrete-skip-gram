import os

import numpy as np

from discrete_skip_gram.plot_util import write_image


def fix_images(path):
    for f in os.listdir(path):
        if f.lower().endswith(".png.npy"):
            outputpath = os.path.join(path, f[:-4])
            img = np.load(os.path.join(path, f))
            write_image(img=img, outputpath=outputpath)


def main():
    fix_images('../output/mnist/gumbel_vae')
    fix_images('../output/mnist/gumbel_vae1')


if __name__ == '__main__':
    main()
