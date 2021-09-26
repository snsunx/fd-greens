"""Helper functions for Green's function calculations."""
import functools
import warnings

def load_parameters(circ, fname) -> dict:
    """Converts keys in dictionaries from original forms to string forms."""
    d_new = {}
    for key, val in d.items():
        d_new.update({str(key): val})
    return d_new

def dict_keys_from_strings(d: dict) -> dict:
    """Convert keys in dictionaries from string forms to original forms."""
    d_new = {}
    for key, val in d.items():
        d_new.update({eval(key): val})
    return d_new

def deprecate_function(msg: str, stacklevel: int = 2):
    """Emits a warning prior to calling decorated function.
    
    Args:
        msg: Warning message to emit.
        stacklevel: The warning stackevel to use, defaults to 2.
    
    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # warn only once
            if not wrapper._warned:
                warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
                wrapper._warned = True
            return func(*args, **kwargs)

        wrapper._warned = False
        return wrapper

    return decorator