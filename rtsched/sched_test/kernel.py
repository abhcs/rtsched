"""Solve the kernel.

The kernel is an optimization problem:

    minimize t
    s.t. t ∈ [a, b] ∩ ℤ
         sum(ceil((t + alpha) / tsk.period) * tsk.wcet
             for tsk, alpha in zip(tsks, alphas)) + beta <= t

a, b, alphas, beta are integral constants, and tsks is a constant list of tasks
with total utilization at most 1. Both FP and EDF schedulability reduce to the
kernel in polynomial time.

The kernel can be solved by any of the following three methods:

    • fixed-point iteration
    • integer programming
    • a specialized cutting-plane algorithm

All three methods are implemented in this module.

More details about the kernel may be found in https://arxiv.org/abs/2210.11185.

"""

import ctypes
import math
import os
from fractions import Fraction
from typing import List, Optional, Tuple

from docplex.mp.model import Model
from rtsched.system.task import Task
from rtsched.util.math import argsort, dot


def fixed_point(tsks: List[Task],
                alphas: List[int],
                beta: int,
                a: int,
                b: int,
                perf=None) -> Optional[int]:
    """Solve a kernel instance using fixed-point iteration.

    Args:
        tsks: list of tasks

        alphas: list of integers of same length as tsks

        beta: an integer

        a, b: integer endpoints of interval [a, b]. tsks, alphas, beta, a, b
              are the constants used to specify the kernel instance.

        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    >>> fixed_point(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    """
    def phi(t):
        return sum([
            math.ceil(Fraction(t + alpha, tsk.period)) * tsk.wcet
            for tsk, alpha in zip(tsks, alphas)
        ]) + beta

    # check trivial cases
    if not tsks and a <= b:
        return a

    if a > b:
        return None

    if phi(a) <= a:
        return a

    # fixed point iteration
    t_ = a
    while t_ <= b:
        v = phi(t_)

        if perf is not None:
            perf.num_iterations += 1

        if v == t_:  # fixed point is found
            return t_
        t_ = v

    # problem is infeasible
    return None


def integer_program(tsks: List[Task],
                    alphas: List[int],
                    beta: int,
                    a: int,
                    b: int,
                    perf=None) -> Optional[int]:
    """Solve a kernel instance using integer programming.

    Args:
        tsks: list of tasks
        alphas: list of integers of same length as tsks
        beta: an integer
        a, b: integer endpoints of interval [a, b]
              tsks, alphas, beta, a, b are the constants used to specify the
              kernel instance.
        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns: the optimal objective value if the instance is feasible; None,
    otherwise.

    Note that cplex uses double-precision (64-bit) arithmetic in its
    computations but the arguments provided to the function are python integers
    that are potentially unbounded.The integers in cplex are not perfect but
    are subject to an integrality tolerance; the constraints in cplex are also
    imperfect and subject to a feasibility tolerance. We tune these tolerances
    using the numbers in the problem instance but for instances with very large
    numbers the cutting-plane implementation is more suitable.

    Remember to add the path to your cplex installation to the environment
    variable PYTHONPATH, and to use a python version that is compatible with
    the installed cplex. I use python 3.8.

    Some tips for building cplex models efficiently are provided here:
    https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/efficient.ipynb

    >>> integer_program(tsks=[Task(20, 40), Task(10, 50), Task(33, 150)], alphas=[0, 0, 0], beta=0, a=20, b=150)
    143

    >>> integer_program(tsks=[Task(6, 17, 10), Task(5, 13, 10)], alphas=[-7, -3], beta=1, a=-12, b=-10)
    -10

    TODO: Experiment with cut callbacks as described here:
    https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/callbacks/cut_callback.py

    """
    if not tsks and a <= b:
        return a

    if a > b:
        return None

    # create model
    # change log_output to True to see cplex output on stdout
    mdl = Model(ignore_names=True, log_output=False, checker='off')

    # create variables
    t = mdl.integer_var(lb=a, ub=b)
    n = len(tsks)
    xs = mdl.integer_var_list(n, lb=-mdl.infinity, ub=mdl.infinity)

    # add constraints
    wcets = [tsk.wcet for tsk in tsks]
    periods = [tsk.period for tsk in tsks]
    mdl.add_constraint(t - mdl.scal_prod(terms=xs, coefs=wcets) >= beta)
    mdl.add_constraints(periods[i] * xs[i] - t >= alphas[i] for i in range(n))

    # set objective
    mdl.minimize(t)

    # specify cplex tolerances for integrality and feasibility
    tol = min(1.0 / (max(periods) + 3.0), 1.0 / (sum(wcets) + 3.0))
    tol = min(tol, 0.1)
    mdl.parameters.mip.tolerances.integrality = tol
    mdl.parameters.simplex.tolerances.feasibility = tol

    # turn off bound strengthening during presolve.if bound strengthening is
    # on, the problem often gets solved during presolve but we wish to measure
    # the number of iterations used by the branch& cut algorithm.if you are not
    # interested in this measurement, you may want to toggle this switch and
    # see if it reduces solve times.
    mdl.parameters.preprocessing.boundstrength = 0

    #solve integer program
    s = mdl.solve()

    if perf is not None:
        perf.num_iterations += mdl.solve_details.nb_iterations

    if s is None:
        return None

    return round(s.objective_value)


def cutting_plane(tsks: List[Task],
                  alphas: List[int],
                  beta: int,
                  a: int,
                  b: int,
                  perf=None) -> Optional[int]:
    """Solve a kernel instance using a specialized cutting-plane algorithm.

    Args:
        tsks: list of tasks
        alphas: list of integers of same length as tsks
        beta: an integer
        a, b: integer endpoints of interval [a, b]
              tsks, alphas, beta, a, b are the constants used to specify the
              kernel instance.
        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    Unlike integer_program(...), this function does not have double-precision
    arithmetic issues because we use python integers and fractions.

    >>> cutting_plane(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    cutting_plane(tsks=[Task(25, 1181, 377, 8), Task(83, 1261, 893, 10), Task(6, 44, 15, 6), Task(4, 9, 13, 9)], alphas= [-812, -378, -35, -5], beta=1, a=-6000, b=-4)
    -13

    """
    if a > b:
        return None

    n = len(tsks)
    if n == 0:
        return a

    p, q, r, xs, ys, pi = beta, 1, beta, [], [], []
    for i in range(n):
        period = tsks[i].period
        wcet = tsks[i].wcet
        util = tsks[i].utilization
        alpha = alphas[i]
        xs.append(math.ceil(Fraction(a + alpha, period)))
        ys.append(period * xs[-1] - alpha)
        pi.append(i)
        p, q, r = p + util * alpha, q - util, r + wcet * xs[-1]

    assert q >= 0, "utilization greater than one is not allowed"
    if p > 0 and q == 0:
        return None
    if r <= a:
        return a
    i0 = 1 if q == 0 else 0
    pi.sort(key=ys.__getitem__, reverse=True)

    while True:
        p, q, i = r, 1, n - 1
        t = r
        while i > i0:
            k = pi[i]
            if p <= q * ys[k]:
                t = math.ceil(Fraction(p, q))
                break
            wcet = tsks[k].wcet
            util = tsks[k].utilization
            alpha = alphas[k]
            p, q, i = p - wcet * xs[k] + util * alpha, q - util, i - 1

        if i == i0:
            t = Fraction(p, q)

        if perf is not None:
            perf.num_iterations += 1

        # instance is infeasible because relaxation is infeasible
        if t > b:
            return None

        # solution is integral and feasible, and hence optimal
        if i == n - 1:
            return math.ceil(t)

        for j in range(i + 1, n):
            k = pi[j]
            period = tsks[k].period
            wcet = tsks[k].wcet
            alpha = alphas[k]
            d = math.ceil(Fraction(t + alpha, period)) - xs[k]
            xs[k], ys[k], r = xs[k] + d, ys[k] + period * d, r + wcet * d

        i += 1
        l1 = pi[:i].copy()
        l2 = sorted(pi[i:n], key=ys.__getitem__, reverse=True)
        i1, i2, k = 0, 0, 0
        while i1 < i and i2 < n - i:
            if ys[l1[i1]] >= ys[l2[i2]]:
                pi[k] = l1[i1]
                i1 += 1
            else:
                pi[k] = l2[i2]
                i2 += 1
            k += 1
        if i1 == i:
            l1 = l2
            i1 = i2
            i = n - i
        while i1 < i:
            pi[k] = l1[i1]
            i1 += 1
            k += 1


def fixed_point_cpp(tsks: List[Task],
                    alphas: List[int],
                    beta: int,
                    a: int,
                    b: int,
                    perf=None) -> Optional[int]:
    """Solve a kernel instance using fixed-point iteration.

    Args:
        tsks: list of tasks

        alphas: list of integers of same length as tsks

        beta: an integer

        a, b: integer endpoints of interval [a, b]. tsks, alphas, beta, a, b
              are the constants used to specify the kernel instance.

        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    There may be overflow issues when converting from python integers to C++
    integers! The C++ implementation itself may contain integer overflow
    errors! DO NOT USE THIS FUNCTION UNLESS YOU KNOW WHAT YOU'RE DOING!

    >>> fixed_point_cpp(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    """
    dir = os.path.dirname(__file__)
    lib = ctypes.CDLL(os.path.join(dir, "build/libkernel.dylib"))
    lib.fixed_point.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.fixed_point.restype = ctypes.c_int64

    wcets = [tsk.wcet for tsk in tsks]
    periods = [tsk.period for tsk in tsks]
    n = len(tsks)
    cs = (ctypes.c_uint64 * n)(*wcets)
    ps = (ctypes.c_uint64 * n)(*periods)
    als = (ctypes.c_int64 * n)(*alphas)
    feasible = ctypes.c_bool(True)
    time = ctypes.c_double(0)
    res = lib.fixed_point(cs, ps, als, ctypes.c_uint64(n),
                          ctypes.c_int64(beta), ctypes.c_int64(a),
                          ctypes.c_int64(b), ctypes.byref(feasible),
                          ctypes.byref(time))
    if perf is not None:
        perf.time += time.value
    if feasible:
        return res
    return None


def cutting_plane_cpp(tsks: List[Task],
                      alphas: List[int],
                      beta: int,
                      a: int,
                      b: int,
                      perf=None) -> Optional[int]:
    """Solve a kernel instance using the c++ implementation of the specialized
    cutting-plane algorithm.

    Args:

        tsks: list of tasks (the length of the list must not exceed 100, and
        the utilization must be strictly less than one)

        alphas: list of integers of same length as tsks

        beta: an integer

        a, b: integer endpoints of interval [a, b]. tsks, alphas, beta, a, b
              are the constants used to specify the kernel instance.

        perf: if perf is not None, then the time used by the method is added to
              perf.time

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    There may be overflow issues when converting from python integers to C++
    integers! The C++ implementation itself may contain integer overflow and
    double-precision errors! The function only works for bounded-utilization
    systems! DO NOT USE THIS FUNCTION UNLESS YOU KNOW WHAT YOU'RE DOING!

    >>> cutting_plane_cpp(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    >>> cutting_plane_cpp(tsks=[Task(25, 1181, 377, 8), Task(83, 1261, 893, 10), Task(6, 44, 15, 6), Task(4, 9, 13, 9)], alphas= [-812, -378, -35, -5], beta=1, a=-6000, b=-4)
    -13

    """
    dir = os.path.dirname(__file__)
    lib = ctypes.CDLL(os.path.join(dir, "build/libkernel.dylib"))
    lib.cutting_plane.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_bool),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.cutting_plane.restype = ctypes.c_int64

    wcets = [tsk.wcet for tsk in tsks]
    periods = [tsk.period for tsk in tsks]
    utils = [tsk.utilization for tsk in tsks]
    assert sum(utils) < 1
    utils = [float(u) for u in utils]
    n = len(tsks)
    cs = (ctypes.c_uint64 * n)(*wcets)
    ps = (ctypes.c_uint64 * n)(*periods)
    us = (ctypes.c_double * n)(*utils)
    als = (ctypes.c_int64 * n)(*alphas)
    feasible = ctypes.c_bool(True)
    time = ctypes.c_double(0)
    res = lib.cutting_plane(cs, ps, us, als, ctypes.c_uint64(n),
                            ctypes.c_int64(beta), ctypes.c_int64(a),
                            ctypes.c_int64(b), ctypes.byref(feasible),
                            ctypes.byref(time))
    if perf is not None:
        perf.time += time.value
    if feasible:
        return res
    return None


METHODS = {
    'fp': fixed_point,  # fixed-point iteration
    'ip': integer_program,  # integer programming
    'cp': cutting_plane,  # specialized cutting-plane algorithm
    'fp_cpp': fixed_point_cpp,  # fixed-point iteration (C++ wrapper)
    'cp_cpp':
    cutting_plane_cpp  # specialized cutting-plane algorithm (C++ wrapper)
}


def solve(tsks: List[Task],
          alphas: List[int],
          beta: int,
          a: int,
          b: int,
          perf=None,
          method: str = 'cp') -> Optional[int]:
    """Solve a kernel instance.

    Args:
        tsks: list of tasks

        alphas: list of integers of same length as tsks

        beta: an integer

        a, b: integer endpoints of interval [a, b]. tsks, alphas, beta, a, b
              are the constants used to specify the kernel instance.

        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

        method: if method is 'fp' (resp, 'ip', 'cp') then the kernel instance
                is solved using fixed-point iteration (resp., integer
                programming, specialized cutting plane)

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    >>> solve(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150, method='cp')
    143

    """
    us = [tsk.utilization for tsk in tsks]
    assert sum(us) <= 1, 'the utilization of the system must be at most 1'
    assert len(tsks) == len(alphas)
    assert method in METHODS
    f = METHODS[method]
    return f(tsks, alphas, beta, a, b, perf)
