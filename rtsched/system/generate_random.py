import math
from fractions import Fraction
from typing import Optional

import numpy as np
from rtsched.system.task import Task
from rtsched.util.drs import drs

# from rtsched.util.randfixedsum import randfixedsum

DEADLINE_TYPES = ['implicit', 'constrained', 'arbitrary']


def generate_system_from_wcets(rng: np.random.Generator, n: int, min_wcet: int,
                               max_wcet: int, sum_util, deadline_type: str,
                               sum_dens, max_jitter: int):
    """Generate a random system of hard real-time tasks by sampling wcets from
    uniform distribution.

    Args:
        rng: a random number generator from numpy like np.random.default_rng()

        n: number of tasks in the system

        min_wcet (resp, max_wcet): minimum (resp., maximum) wcet of any task in
                  the system. wcets are picked from a discrete uniform
                  distribution over [min_wcet, max_wcet).

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
    [Task(20, 94, 74, 1), Task(10, 121, 69, 1), Task(53, 189, 186, 0), Task(97, 565, 495, 2)]

    """

    # alternative to drs
    # us = randfixedsum(n, sum_util, 1, rng)[0]

    us = drs(rng, n, sum_util)
    wcets = rng.integers(low=min_wcet, high=max_wcet, size=n)
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
        deadlines = [math.ceil(wcet / d) for d, wcet in zip(ds, wcets)]
        assert all(d <= p for d, p in zip(deadlines, periods))
    else:
        ds = drs(rng, n, sum_dens, upper_bounds=ones)
        deadlines = [math.ceil(wcet / d) for d, wcet in zip(ds, wcets)]
    jitters = rng.integers(
        low=0,
        high=[min(max_jitter, d - w) + 1 for w, d in zip(wcets, deadlines)],
        size=n)
    return [
        Task(wcet=w.item(), period=p, deadline=d, jitter=j.item())
        for w, p, d, j in zip(wcets, periods, deadlines, jitters)
    ]


def generate_system_from_periods(rng: np.random.Generator, n: int,
                                 min_period: int, max_period: int, sum_util,
                                 deadline_type: str, sum_dens,
                                 max_jitter: int):
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

    Returns:
        A random system of hard real-time tasks.

    >>> generate_system_from_periods(np.random.default_rng(seed=42), n=4, min_period=10, max_period=10000, sum_util=0.75, deadline_type='constrained', sum_dens=0.9, max_jitter=2)
    [Task(13, 1095, 752, 2), Task(124, 303, 286, 1), Task(711, 4489, 4025, 1), Task(126, 739, 460, 2)]
    """
    ones = [1.0] * n
    lbs = [0.001] * n
    while True:
        periods = np.exp(
            rng.uniform(low=np.log(min_period),
                        high=np.log(max_period),
                        size=n)).tolist()
        us = drs(rng, n, sum_util, lower_bounds=lbs)
        wcets = []
        for i in range(n):
            period = math.ceil(periods[i])
            x = Fraction(us[i]).limit_denominator(period * 2)
            s = 1
            if x.denominator < period:
                s = math.floor(Fraction(period, x.denominator))
            periods[i] = x.denominator * s
            wcets.append(x.numerator * s)
            us[i] = float(x)
        if sum(us) >= 1.0 or any(wcet < 1 for wcet in wcets):
            continue
        if deadline_type == 'implicit':
            deadlines = periods
        elif deadline_type == 'constrained':
            if sum(us) > sum_dens:
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


def generate_system(rng: np.random.Generator, n: int, min_wcet: Optional[int],
                    max_wcet: Optional[int], min_period: Optional[int],
                    max_period: Optional[int], sum_util, deadline_type: str,
                    sum_dens, max_jitter: int, method: str):
    """Generate a random system of hard real-time tasks by sampling periods
    from a log-uniform distribution.

    This function calls the appropriate system generation function based on the
    method parameter. If the method is 'periods' then min_wcet and max_wcet are
    ignored; and if the method is 'wcets' then min_period and max_period are
    ignored.

    """
    if method == 'periods':
        assert min_period is not None and max_period is not None
        return generate_system_from_periods(rng, n, min_period, max_period,
                                            sum_util, deadline_type, sum_dens,
                                            max_jitter)
    elif method == 'wcets':
        assert min_wcet is not None and max_wcet is not None
        return generate_system_from_wcets(rng, n, min_wcet, max_wcet, sum_util,
                                          deadline_type, sum_dens, max_jitter)
    assert False, "invalid system generation method"
