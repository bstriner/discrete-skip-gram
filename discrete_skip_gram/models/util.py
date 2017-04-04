from os.path import dirname, exists
from os import makedirs


def makepath(path):
    if not exists(dirname(path)):
        makedirs(dirname(path))
