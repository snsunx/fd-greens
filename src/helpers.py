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

def deprecate_arguments(kwarg_map):
    """Decorator to automatically alias deprecated argument names and warn 
    upon use.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map)
            return func(*args, **kwargs)

        return wrapper

    return decorator


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

def _rename_kwargs(func_name, kwargs, kwarg_map):
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")

            warnings.warn(
                "{} keyword argument {} is deprecated and "
                "replaced with {}.".format(func_name, old_arg, new_arg),
                DeprecationWarning,
                stacklevel=3,
            )

            kwargs[new_arg] = kwargs.pop(old_arg)