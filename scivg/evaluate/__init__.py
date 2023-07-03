from evaluate import load as _load
from os.path import dirname, isdir, join

def load(path, *args, **kwargs):
    if isdir(local := join(dirname(__file__), path)):
        return _load(local, *args, **kwargs)
    return _load(path, *args, **kwargs)
