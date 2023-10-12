from contextlib import contextmanager

@contextmanager
def optional_dependencies(error: str = "ignore"):
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        if error == "raise":
            raise e
        elif error == "warn":
            msg = f'Missing optional dependency "{e.name}". Use pip or conda to install.'
            print(f'Warning: {msg}')

@contextmanager
def temporary_change_attributes(something, **kwargs):
    previous_values = {k: getattr(something, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(something, k, v)
    try:
        yield
    finally:
        for k, v in previous_values.items():
            setattr(something, k, v)
