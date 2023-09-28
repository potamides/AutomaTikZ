from os.path import dirname, isdir, join

from ..util import optional_dependencies

with optional_dependencies():
    from evaluate import load as _load

def load(path, *args, **kwargs):
    try:
        if isdir(local := join(dirname(__file__), path)):
            return _load(local, *args, **kwargs)
        return _load(path, *args, **kwargs)
    except (NameError, ImportError):
        raise ValueError(
            "Missing dependencies: "
            "Install this project with the [evaluate] feature name!"
        )
