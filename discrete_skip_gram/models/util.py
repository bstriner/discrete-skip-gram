import re
from os import makedirs, listdir
from os.path import dirname, exists, join


def makepath(path):
    if not exists(dirname(path)):
        makedirs(dirname(path))


def latest_model(path, fmt, fail=False):
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
        if fail:
            raise ValueError("File not found: {}".format(path))
        else:
            return None


#if __name__ == "__main__":
#    print latest_model("D:\\Projects\\discrete-skip-gram\\experiments\\brown\\output\\brown\\skipgram_baseline",
#                 "model-(\\d+).csv")
