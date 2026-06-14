import importlib.util


def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available() -> bool:
    if not is_bnb_available():
        return False
    try:
        import bitsandbytes as bnb
    except Exception:
        return False
    return hasattr(bnb, "nn") and hasattr(bnb.nn, "Linear4bit")
