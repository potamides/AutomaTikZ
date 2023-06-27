from contextlib import contextmanager

import datasets
import transformers

@contextmanager
def set_verbosity(verbosity: int | str):
    """
    Temporally change log level for huggingface libraries by using context
    managers.
    """

    old_verbosities = dict()
    for logging in [transformers.logging, datasets.logging]:
        old_verbosities[logging] = logging.get_verbosity()
        logging.set_verbosity(verbosity if isinstance(verbosity, int) else logging.log_levels[verbosity])

    try:
        yield
    finally:
        for logging, old_verbosity in old_verbosities.items():
            logging.set_verbosity(old_verbosity)
