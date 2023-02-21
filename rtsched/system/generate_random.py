
import math

import numpy as np
from rtsched.system.task import Task
from rtsched.util.drs import drs

# from rtsched.util.randfixedsum import randfixedsum

DEADLINE_TYPES = ['implicit', 'constrained', 'arbitrary']


def generate_system(rng: np.random.Generator, n: int, max_wcet: int, sum_util,
                    deadline_type: str, sum_dens, max_jitter: int):
    """Generate a random system of n hard real-time tasks.

    Args:
        rng: a random number generator from numpy like np.random.default_rng()

        n: number of tasks in the system

        max_wcet: maximum wcet of any task in the system. wcets are picked from
                  a discrete uniform distribution from 1 to max_wcet.

        sum_util: total utilization of the system. we use Dirichlet-Rescale
                  (DRS) algorithm to generate n utilizations that sum to
                  sum_util. Since we require all parameters to be integral, the
                  periods are rounded up; thus the total utilization may be
                  slightly less than sum_util.

        deadline_type: if the type is 'implicit', then each deadline is equal
                       to its respective period; if the type is 'constrained',
                       then each deadline is at most its respective period; if
                       the type is 'arbitrary', then no relations need to hold
                       between deadlines and periods.

        sum_dens: total density of the system, where density of a task is equal
                  to the ratio of its wcet to its deadline. We define density
                  in an ad-hoc way so that we can generate deadlines from the
                  wcets similar to the way that we generate periods from wcets.
                  The caller must take care to ensure that sum_dens is
                  consistent with deadline_type; inconsistency is handled by
                  drs by throwing a ValueError (see
                  https://pypi.org/project/drs/).

        max_jitter: maximum jitter of any task in the system. jitters are
                  picked from a discrete uniform distribution from 0 to
                  min(max_jitter, deadline - wcet); the second argument to min
                  is due to our insistence on producing well-formed tasks that
                  satisfy wcet + jitter <= deadline.

    Returns:
        A random system of hard real-time tasks.

    >>> generate_system(np.random.default_rng(seed=42), n=4, max_wcet=100, sum_util=0.75, deadline_type='constrained', sum_dens=0.9, max_jitter=2)
    [Task(21, 99, 78, 1), Task(10, 121, 69, 1), Task(53, 189, 185, 0), Task(98, 571, 500, 2)]

    """

    # alternative to drs
    # us = randfixedsum(n, sum_util, 1, rng)[0]

    us = drs(rng, n, sum_util)
    wcets = rng.integers(low=1, high=max_wcet+1, size=n)
    ones = [1.0] * n
    periods = [math.ceil(wcet / u) for u, wcet in zip(us, wcets)]
    assert deadline_type in DEADLINE_TYPES
    if deadline_type == 'implicit':
        assert math.isclose(sum_util, sum_dens)
        # TODO: investigate why drs(rng, n, sum_dens, upper_bounds=us,
        # lower_bounds=us) does not work
        deadlines = periods
    elif deadline_type == 'constrained':
        assert math.isclose(sum_util, sum_dens) or sum_util < sum_dens
        ds = drs(rng, n, sum_dens, lower_bounds=us, upper_bounds=ones)
        deadlines = [math.ceil(wcet / d) for d, wcet in zip(ds, wcets)]
        assert all(d <= p for d,p in zip(deadlines, periods))
    else:
        ds = drs(rng, n, sum_dens, upper_bounds=ones)
        deadlines = [math.ceil(wcet / d) for d, wcet in zip(ds, wcets)]
    jitters = rng.integers(low=0, high=[min(max_jitter, d - w) + 1 for w,d in
                                        zip(wcets, deadlines)], size=n)
    return [Task(wcet=w.item(), period=p, deadline=d, jitter=j.item())
            for w,p,d,j in zip(wcets, periods, deadlines, jitters)]
