from typing import Sized, Iterable


def chunked(iterable:Sized, size) -> Iterable:
    """ returns a generator of chunks of a (large?) list """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]