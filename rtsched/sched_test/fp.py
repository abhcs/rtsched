"""Solve FP schedulability.

FP schedulability corresponds to the following optimization problem:

    minimize t
    s.t. t ∈ (0, tsks[-1].deadline - tsks[-1].jitter] ∩ ℤ
         sum(ceil((t + tsk.jitter) / tsk.period) * tsk.wcet
             for tsk in tsks[:-1]) + tsks[-1].wcet <= t

tsks is a constant list of tasks with total utilization at most 1; the tasks
are listed in decreasing order of priority.

FP schedulability can be reduced to the kernel; thus, it can be solved using
any of the three methods available for solving the kernel.

"""

import math
from fractions import Fraction
from typing import List

from rtsched.sched_test import kernel
from rtsched.system.task import Task
from rtsched.util.math import dot


def solve(tsks: List[Task], method, perf=None):
    """Solve an FP schedulability instance.

    Args:
        tsks: nonempty list of tasks

        method: if method is 'fp' (resp, 'ip', 'cp') then the FP sched instance
                is solved using fixed-point iteration (resp., integer
                programming, specialized cutting plane)

        perf: if perf is not None, then the number of iterations used by the
              method is stored in perf.num_iterations

    Returns: the optimal objective value if the instance is feasible; None,
        otherwise.

    >>> solve([Task(20, 40), Task(10, 50), Task(33, 150)], method='cp')
    143

    >>> solve([Task(27, 442, 166, 8), Task(78, 114, 106, 2)], method='ip')
    

    """

    assert tsks
    us = [tsk.utilization for tsk in tsks]
    assert sum(us) <= 1

    # compute left endpoint a of interval [a,b] in the reduction to the kernel.
    # traditionally, this is also known as the 'initial value' for 'response
    # time analysis'.
    js = [tsk.jitter for tsk in tsks[:-1]]
    us = [tsk.utilization for tsk in tsks[:-1]]
    num = tsks[-1].wcet + dot(us, js)
    den = 1 - sum(us)
    a = math.ceil(Fraction(num, den))

    # call the appropriate kernel function
    return kernel.solve(tsks=tsks[:-1],
                        alphas=js,
                        beta=tsks[-1].wcet,
                        a=a,
                        b=tsks[-1].deadline - tsks[-1].jitter,
                        perf=perf,
                        method=method)


def schedulable(tsks: List[Task]) -> bool:
    """Decides whether the tasks are schedulable.

    Args:
        tsks: list of tasks

    Returns:
        True if system described by tsks is schedulable; False otherwise.

    >>> schedulable([Task(20, 40), Task(10, 50), Task(33, 150)])
    True

    >>> schedulable([Task(2, 4), Task(3, 6)])
    False
    """
    us = [tsk.utilization for tsk in tsks]
    return sum(us) <= 1 and all(
        solve(tsks[:i], 'cp') is not None for i in range(2,
                                                         len(tsks) + 1))
