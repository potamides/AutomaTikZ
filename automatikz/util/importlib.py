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
