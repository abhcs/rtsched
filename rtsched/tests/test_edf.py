"""Test correctness of solve and schedulable from edf module.

"""

import math
from fractions import Fraction
from typing import List

import numpy as np
import pytest
from rtsched.sched_test.edf import L, schedulable, solve
from rtsched.system.generate_random import generate_system_from_wcets
from rtsched.system.task import Task


def test_bad_inputs():
    with pytest.raises(AssertionError):
        solve([], method='cp')  # empty task list

    with pytest.raises(AssertionError):
        solve([Task(1, 2), Task(2, 4)], method='xp')  # incorrect method

    with pytest.raises(AssertionError):
        solve([Task(2, 1)], method='cp')  # utilization > 1

    with pytest.raises(AssertionError):
        solve([Task(1.1, 2)], method='cp')  # fractional data


def test_1():
    wcets = [6000, 2000, 1000, 90, 8, 2, 10, 26]
    periods = [31000, 9800, 17000, 4200, 96, 12, 280, 660]
    deadlines = [18000, 9000, 12000, 3000, 78, 16, 120, 160]
    tsks = [
        Task(wcet=w, period=p, deadline=d)
        for w, p, d in zip(wcets, periods, deadlines)
    ]
    assert schedulable(tsks)
    tsks.append(Task(10, 100, 15))
    assert not schedulable(tsks)


def test_2():
    wcets = [331, 3654]
    periods = [15000, 77000]
    deadlines = [2688, 3849]
    tsks = [
        Task(wcet=w, period=p, deadline=d)
        for w, p, d in zip(wcets, periods, deadlines)
    ]
    assert not schedulable(tsks)


def test_3():
    wcets = [331, 3654, 413, 349, 1113]
    periods = [15000, 77000, 34000, 70000, 83000]
    deadlines = [2688, 3849, 1061, 20189, 10683]
    tsks = [
        Task(wcet=w, period=p, deadline=d)
        for w, p, d in zip(wcets, periods, deadlines)
    ]
    assert not schedulable(tsks)


def dbf(tsks: List[Task], t: int) -> int:
    """Evaluate the demand bound function for a list of tasks at time t."""
    return sum([
        max(
            0,
            math.floor(
                Fraction(t + tsk.period + tsk.jitter - tsk.deadline,
                         tsk.period)) * tsk.wcet) for tsk in tsks
    ])


def qpa(tsks: List[Task]):
    """A traditional implementation of quick processor-demand analysis (QPA).

    We compare the outputs of this implementation with the outputs of the three
    methods implemented in the kernel to test consistency/correctness.

    """
    assert tsks

    us = [tsk.utilization for tsk in tsks]
    assert sum(us) <= 1

    def max_dj(t):
        """compute max{dᵢ < t | dᵢ = kTᵢ + Dᵢ - Jᵢ, k ∈ ℕ}.

        """
        max = 0
        for tsk in tsks:
            if tsk.dj < t:
                d = math.floor(Fraction(t - tsk.dj,
                                        tsk.period)) * tsk.period + tsk.dj
                if d == t:
                    d -= tsk.period
                if d > max:
                    max = d
        return max

    t = L(tsks)
    min_dj = min(tsk.dj for tsk in tsks)
    while t >= min_dj:
        v = dbf(tsks, t)
        if v > t:
            return t
        if v < t:
            t = v
        else:
            t = max_dj(t)
    return None


@pytest.fixture
def seed():
    return 42


@pytest.mark.parametrize(
    "num_systems, n, min_wcet, max_wcet, sum_util, deadline_type, sum_dens,"
    "max_jitter", [(100, 2, 1, 100, 0.75, 'constrained', 0.9, 10),
                   (1000, 5, 1, 100, 0.80, 'arbitrary', 0.9, 10)])
def test_4(seed: int, num_systems: int, n: int, min_wcet: int, max_wcet: int,
           sum_util, deadline_type: str, sum_dens, max_jitter: int):
    rng = np.random.default_rng(seed)
    for _ in range(num_systems):
        tsks = generate_system_from_wcets(rng, n, min_wcet, max_wcet, sum_util,
                                          deadline_type, sum_dens, max_jitter)
        expected = qpa(tsks)

        got = solve(tsks, method='fp')
        assert (got == expected or (got is not None and expected is not None
                                    and dbf(tsks, got) == dbf(tsks, expected))),\
                                    str(tsks)

        got = solve(tsks, method='ip')
        assert (got == expected or (got is not None and expected is not None
                                    and dbf(tsks, got) == dbf(tsks, expected))),\
                                    str(tsks)

        got = solve(tsks, method='cp')
        assert (got == expected or (got is not None and expected is not None
                                    and dbf(tsks, got) == dbf(tsks, expected))),\
                                    str(tsks)
