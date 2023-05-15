import math

import numpy as np
from rtsched.system.task import Task
from rtsched.util.drs import drs

# from rtsched.util.randfixedsum import randfixedsum

DEADLINE_TYPES = ['implicit', 'constrained', 'arbitrary']


def generate_system_from_wcets(rng: np.random.Generator, n: int, min_wcet: int,
                               max_wcet: int, sum_util, deadline_type: str,
                               sum_dens, max_jitter: int):
    """Generate a random system of hard real-time tasks by sampling wcets from
    log-uniform distribution.

    Args:
        rng: a random number generator from numpy like np.random.default_rng()

        n: number of tasks in the system

        min_wcet (resp, max_wcet): minimum (resp., maximum) wcet of any task in
                  the system. wcets are picked from a log-uniform distribution
                  over [min_wcet, max_wcet).

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

    >>> generate_system_from_wcets(np.random.default_rng(seed=42), n=4, min_wcet=1, max_wcet=100, sum_util=0.75, deadline_type='constrained', sum_dens=0.9, max_jitter=2)
    [Task(2, 10, 9, 2), Task(90, 1084, 826, 1), Task(34, 121, 112, 1), Task(38, 222, 133, 2)]

    """

    # alternative to drs
    # us = randfixedsum(n, sum_util, 1, rng)[0]

    us = drs(rng, n, sum_util)
    # wcets = rng.integers(low=min_wcet, high=max_wcet, size=n)
    wcets = np.exp(
        rng.uniform(low=np.log(min_wcet), high=np.log(max_wcet),
                    size=n)).tolist()
    wcets = [math.ceil(p) for p in wcets]
    ones = [1.0] * n
    periods = [math.ceil(wcet / u) for u, wcet in zip(us, wcets)]
    us = [wcet / period for wcet, period in zip(wcets, periods)]
    sum_util = sum(us)
    assert deadline_type in DEADLINE_TYPES
    if deadline_type == 'implicit':
        deadlines = periods
    elif deadline_type == 'constrained':
        assert math.isclose(sum_util, sum_dens) or sum_util < sum_dens
        ds = drs(rng, n, sum_dens, lower_bounds=us, upper_bounds=ones)
        deadlines = [math.floor(wcet / d) for d, wcet in zip(ds, wcets)]
        assert all(d <= p for d, p in zip(deadlines, periods))
    else:
        ds = drs(rng, n, sum_dens, upper_bounds=ones)
        deadlines = [math.floor(wcet / d) for d, wcet in zip(ds, wcets)]
    jitters = rng.integers(
        low=0,
        high=[min(max_jitter, d - w) + 1 for w, d in zip(wcets, deadlines)],
        size=n)
    return [
        Task(wcet=w, period=p, deadline=d, jitter=j.item())
        for w, p, d, j in zip(wcets, periods, deadlines, jitters)
    ]


def generate_system_from_periods(rng: np.random.Generator, n: int,
                                 min_period: int, max_period: int, sum_util,
                                 deadline_type: str, sum_dens, max_jitter: int,
                                 abs_tol: float):
    """Generate a random system of hard real-time tasks by sampling periods
    from a log-uniform distribution.

    Args:
        rng: a random number generator from numpy like np.random.default_rng()

        n: number of tasks in the system

        min_period (resp., max_period): minimum (resp., maximum) period of any
                  task in the system. periods are picked from a log-uniform
                  distribution over [min_period, max_period).

        sum_util: total utilization of the system. we use Dirichlet-Rescale
                  (DRS) algorithm to generate n utilizations that sum to
                  sum_util. Since we require all parameters to be integral, we
                  try to find a good rational approximation of the utilization
                  in a neighborhood of the corresponding period; the wcet and
                  period are set to equal the numerator and denominator of this
                  fraction respectively.

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
                  consistent with deadline_type.

        max_jitter: maximum jitter of any task in the system. jitters are
                  picked from a discrete uniform distribution from 0 to
                  min(max_jitter, deadline - wcet); the second argument to min
                  is due to our insistence on producing well-formed tasks that
                  satisfy wcet + jitter <= deadline.

        abs_tol: absolute tolerance of divergence from sum_util.

    Returns:
        A random system of hard real-time tasks.

    >>> generate_system_from_periods(np.random.default_rng(seed=42), n=4, min_period=10, max_period=10000, sum_util=0.75, deadline_type='constrained', sum_dens=0.9, max_jitter=2, abs_tol=0.05)
    [Task(22, 2099, 2099, 1), Task(85, 208, 208, 0), Task(595, 3766, 3766, 2), Task(210, 1237, 1237, 1)]

    """
    ones = [1.0] * n
    while True:
        periods = np.exp(
            rng.uniform(low=np.log(min_period),
                        high=np.log(max_period),
                        size=n)).tolist()
        periods = [math.ceil(p) for p in periods]
        us = drs(rng, n, sum_util)
        wcets = [max(1, math.floor(p * u)) for (p, u) in zip(periods, us)]
        us = [w / p for (w, p) in zip(wcets, periods)]
        if not math.isclose(sum_util, sum(us), abs_tol=abs_tol):
            continue
        if deadline_type == 'implicit':
            deadlines = periods
        elif deadline_type == 'constrained':
            if sum(us) < sum_dens:
                # assume implicit deadlines because total utilization must be
                # less than total density for a constrained-deadline system.
                deadlines = periods
            else:
                ds = drs(rng, n, sum_dens, lower_bounds=us, upper_bounds=ones)
                deadlines = [
                    math.floor(wcet / d) for d, wcet in zip(ds, wcets)
                ]
                assert all(d <= p for d, p in zip(deadlines, periods))
        else:
            ds = drs(rng, n, sum_dens, upper_bounds=ones)
            deadlines = [math.floor(wcet / d) for d, wcet in zip(ds, wcets)]
        jitters = rng.integers(low=0,
                               high=[
                                   min(max_jitter, d - w) + 1
                                   for w, d in zip(wcets, deadlines)
                               ],
                               size=n)
        return [
            Task(wcet=w, period=p, deadline=d, jitter=j.item())
            for w, p, d, j in zip(wcets, periods, deadlines, jitters)
        ]
