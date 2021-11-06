from datetime import datetime


def vprint(*msg, verbose: bool, timestamp: bool = False, **kwargs):
    if verbose:
        if timestamp:
            ts = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
            msg = (ts,) + msg
        print(*msg, **kwargs)
