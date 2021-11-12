from datetime import datetime


def vprint(*msg, verbose: bool, timestamp: bool = False, **kwargs):
    if verbose:
        if timestamp:
            ts = "[" + get_timestamp("%Y-%m-%d %H:%M:%S") + "]"
            msg = (ts,) + msg
        print(*msg, **kwargs)


def get_timestamp(fmt: str) -> str:
    return datetime.now().strftime(fmt)
