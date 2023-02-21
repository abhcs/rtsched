"""Test correctness of solve and schedulable from fp module.

"""

import math
from fractions import Fraction
from typing import List

import numpy as np
import pytest
from rtsched.sched_test.fp import schedulable, solve
from rtsched.system.generate_random import generate_system
from rtsched.system.task import Task


@pytest.mark.parametrize("wcets, periods, expected", [
    ([20, 10, 33], [40, 50, 150], 143),
    ([1], [4], 1),
    ([1, 1], [4, 5], 2),
    ([1, 1, 3], [4, 5, 9], 7),
    ([1, 1, 3, 3], [4, 5, 9, 18], 18),
    ([1], [2], 1),
    ([1, 5], [2, 20], 10),
    ([1, 5, 1], [2, 20, 20], 12)
])
def test_1(wcets: List[int], periods: List[int], expected: int):
    tsks = [Task(wcet=w, period=p) for w, p in zip(wcets, periods)]
    assert solve(tsks, method='fp') == expected
    assert solve(tsks, method='ip') == expected
    assert solve(tsks, method='cp') == expected


def rta(tsks: List[Task]):
    """A traditional implementation of response time analysis (RTA).

    We compare the outputs of this implementation with the outputs of the three
    methods implemented in the kernel to test consistency/correctness.

    """
    assert tsks
    us = [tsk.utilization for tsk in tsks]
    assert sum(us) <= 1
    def rbf(t: int):
        return sum(math.ceil(Fraction(t + tsk.jitter, tsk.period)) * tsk.wcet
                   for tsk in tsks)
    t_ = 1
    while t_ <= tsks[-1].deadline - tsks[-1].jitter:
        v = rbf(t_)
        if v == t_:
            return t_
        t_ = v
    return None


def test_2():
    wcets = [3, 15, 15, 40, 30, 200]
    periods = [10, 100, 200, 400, 1000, 1000]
    deadlines = [10, 50, 200, 400, 500, 1000]
    jitters = [2, 5, 5, 50, 50, 100]
    tsks = [Task(wcet=w, period=p, deadline=d, jitter=j) for w, p, d, j in
            zip(wcets, periods, deadlines, jitters)]
    expected = [3, 24, 45, 124, 166, 682]
    for i in range(len(tsks)):
        assert rta(tsks[:i+1]) == expected[i]
        assert solve(tsks[:i+1], method='fp') == expected[i]
        assert solve(tsks[:i+1], method='ip') == expected[i]
        assert solve(tsks[:i+1], method='cp') == expected[i]


@pytest.mark.parametrize("tsks, expected", [
    ([Task(wcet=i, period=j) for i,j in zip([20, 10, 33], [40, 50, 150])], True),
    ([Task(wcet=i, period=j) for i,j in zip([1, 1, 3, 3], [4, 5, 9, 18])], True),
    ([Task(wcet=i, period=j) for i,j in zip([1, 5, 1], [2, 20, 20])], True)
])
def test_3(tsks, expected):
    assert schedulable(tsks) == expected


def test_bad_inputs():
    with pytest.raises(AssertionError):
        solve([], method='cp') # empty task list

    with pytest.raises(AssertionError):
        solve([Task(1,2)], method='xp') # incorrect method

    with pytest.raises(AssertionError):
        solve([Task(2,1)], method='cp') # utilization > 1

    with pytest.raises(AssertionError):
        solve([Task(1.1, 2)], method='cp') # fractional data


@pytest.fixture
def seed():
    return 42


@pytest.mark.parametrize("num_systems, n, max_wcet, sum_util, deadline_type, sum_dens,"
                         "max_jitter",[
                             (1000, 2, 100, 0.75, 'constrained', 0.9, 10),
                             (1000, 4, 100, 0.9, 'implicit', 0.9, 10),
                             (20, 16, 100, 0.9, 'implicit', 0.9, 10)
                         ])
def test_4(seed: int, num_systems: int, n: int, max_wcet: int, sum_util,
           deadline_type: str, sum_dens, max_jitter: int):
    rng = np.random.default_rng(seed)
    for _ in range(num_systems):
        tsks = generate_system(rng, n, max_wcet, sum_util, deadline_type,
                               sum_dens, max_jitter)
        expected = rta(tsks)
        assert solve(tsks, method='fp') == expected
        assert solve(tsks, method='ip') == expected
        assert solve(tsks, method='cp') == expected
