"""This module provides miscellaneous utilities.
"""

import functools
import math


def dot(v1, v2):
    """Compute the dot product of two vectors of equal length.

    >>> dot([1, 2], [3, 4])
    11
    """
    return sum([e1 * e2 for e1, e2 in zip(v1, v2)])


def argsort(seq, reverse=False):
    """Compute the indices that would sort a sequence.

    >>> argsort([3, 1, 2])
    [1, 2, 0]
    """
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def lcm(v):
    """Compute the lcm of a vector.
    """
    return functools.reduce(lambda x, y: (x * y) // math.gcd(x, y), v)
