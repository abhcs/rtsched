"""Solve EDF schedulability.

EDF schedulability corresponds to the following n optimization problems where
k ∈ range(1, n+1):

    minimize t
    s.t. t ∈ (-b[k], -a[k]] ∩ ℤ
         sum(ceil((t + tsk.deadline - tsk.period - tsk.jitter) / tsk.period) * tsk.wcet
             for tsk in tsks[:k]) + 1 <= t

tsks is a constant list of tasks with total utilization at most 1; the tasks
are listed in nondecreasing order using the key deadline - period - jitter.

EDF schedulability can be reduced to the kernel; thus, it can be solved using
any of the three methods available for solving the kernel.

"""

import math
from typing import List

from rtsched.sched_test import kernel
from rtsched.system.task import Task
from rtsched.util.math import dot, lcm


def L(tsks: List[Task]) -> int:
    """Computes the initial value for EDF schedulability.
    """
    us = [tsk.utilization for tsk in tsks]
    sum_us = sum(us)
    vs = [tsk.dj - tsk.period for tsk in tsks]
    if sum_us < 1:
        return max(math.floor(dot(us, vs) / (sum_us - 1) - 1), max(vs))
    js = [tsk.jitter for tsk in tsks]
    ws = [tsk.wcet for tsk in tsks]
    ps = [tsk.period for tsk in tsks]
    s = kernel.cutting_plane(tsks=tsks,
                             alphas=js,
                             beta=0,
                             a=sum(ws),
                             b=lcm(ps))
    assert s is not None
    return math.floor(s)


def solve(tsks: List[Task], method, perf=None):
    """Solve an EDF schedulability instance.

    Args:
        tsks: nonempty list of tasks

        method: if method is 'fp' (resp, 'ip', 'cp') then the EDF sched
                instance is solved using fixed-point iteration (resp., integer
                programming, specialized cutting plane)

        perf: if perf is not None, then the number of iterations used by the
              method is stored in perf.num_iterations

    Returns: the optimal objective value if the instance is feasible; None,
        otherwise.

    >>> solve([Task(5, 13, 10), Task(6, 17, 10), Task(1, 20, 31)], method='cp')
    10

    """
    assert tsks
    us = [tsk.utilization for tsk in tsks]
    assert sum(us) <= 1

    tsks = sorted(tsks, key=lambda tsk: tsk.dj - tsk.period)
    n = len(tsks)

    min_dj = min(tsk.dj for tsk in tsks)
    l = L(tsks)

    if min_dj >= l:
        return None

    vs = [tsk.dj - tsk.period for tsk in tsks]
    p = 0
    while p < n - 1:
        if vs[p + 1] > min_dj:
            break
        p += 1
    q = n - 1
    while q >= 0:
        if vs[q] < l:
            break
        q -= 1
    k = q

    while k >= p:
        a = max(min_dj, vs[k])
        b = l if k == n - 1 else vs[k + 1]
        s = kernel.solve(tsks=tsks[:k + 1],
                         alphas=vs[:k + 1],
                         beta=1,
                         a=-b,
                         b=-a,
                         perf=perf,
                         method=method)
        if s is not None:
            return -s
        k -= 1
    return None


def schedulable(tsks: List[Task]) -> bool:
    """Decides whether the tasks are schedulable.

    Args:
        tsks: list of tasks

    Returns:
        True if system described by tsks is schedulable; False otherwise.

    >>> schedulable([Task(2, 4), Task(3, 6)])
    True
    """
    us = [tsk.utilization for tsk in tsks]
    return sum(us) <= 1 and solve(tsks, method='cp') is None
