def require(ok, message):
    if not ok:
        raise RuntimeError(message)
