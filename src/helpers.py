"""Helper functions for Green's function calculations."""

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