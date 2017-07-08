import os
import re
from os import listdir
from os.path import join

def make_path(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def latest_file(path, fmt):
    prog = re.compile(fmt)
    latest_epoch = -1
    latest_m = None
    for f in listdir(path):
        m = prog.match(f)
        if m:
            epoch = int(m.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_m = f
    if latest_m:
        return join(path, latest_m), latest_epoch
    else:
        return None