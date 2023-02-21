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


import math
from fractions import Fraction
from typing import List, Optional, Tuple

from docplex.mp.model import Model
from rtsched.system.task import Task
from rtsched.util.math import argsort, dot


def fixed_point(tsks : List[Task], alphas: List[int], beta: int, a: int,
                b: int, perf=None) -> Optional[int]:
    """Solve a kernel instance using fixed-point iteration.

    Args:
        tsks: nonempty list of tasks
        alphas: list of integers of same length as tsks
        beta: an integer
        a, b: integer endpoints of interval [a, b]
              tsks, alphas, beta, a, b are the constants used to specify the
              kernel instance.
        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    See Sec. 5.1 in https://arxiv.org/abs/2210.11185.

    >>> fixed_point(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    """
    def phi(t):
        return sum([math.ceil(Fraction(t + alpha, tsk.period)) * tsk.wcet
                    for tsk, alpha in zip(tsks, alphas)]) + beta


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

        if v == t_: # fixed point is found
            return t_
        t_ = v

    # problem is infeasible
    return None


def integer_program(tsks : List[Task], alphas: List[int], beta: int, a: int,
                    b: int, perf=None) -> Optional[int]:
    """Solve a kernel instance using integer programming.

    Args:
        tsks: nonempty list of tasks
        alphas: list of integers of same length as tsks
        beta: an integer
        a, b: integer endpoints of interval [a, b]
              tsks, alphas, beta, a, b are the constants used to specify the
              kernel instance.
        perf: if perf is not None, then the number of iterations used by the
              method is added to perf.num_iterations

    Returns:
        the optimal objective value if the instance is feasible; None, otherwise.

    See Sec. 5.2 in https://arxiv.org/abs/2210.11185 for more details on the
    integer program.

    Note that cplex uses double-precision (64-bit) arithmetic in its
    computations but the arguments provided to the function are python integers
    that are potentially unbounded. The integers in cplex are not perfect but
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
    t = mdl.integer_var(lb = a, ub = b)
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

    # turn off bound strengthening during presolve. if bound strengthening is
    # on, the problem often gets solved during presolve but we wish to measure
    # the number of iterations used by the branch & cut algorithm. if you are
    # not interested in this measurement, you may want to toggle this switch
    # and see if it reduces solve times.
    mdl.parameters.preprocessing.boundstrength = 0

    # solve integer program
    s = mdl.solve()

    if perf is not None:
        perf.num_iterations += mdl.solve_details.nb_iterations

    if s is None:
        return None

    return round(s.objective_value)


def cutting_plane(tsks : List[Task], alphas: List[int], beta: int, a: int,
                    b: int, perf=None) -> Optional[int]:
    """Solve a kernel instance using a specialized cutting-plane algorithm.

    Args:
        tsks: nonempty list of tasks
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

    See Sec. 5.3-5.6 in https://arxiv.org/abs/2210.11185 for more details on
    the algorithm.

    >>> cutting_plane(tsks = [Task(20, 40), Task(10, 50), Task(33, 150)], alphas = [0, 0, 0], beta= 0, a = 20, b = 150)
    143

    """
    if not tsks and a <= b:
        return a

    if a > b:
        return None

    xs = [math.ceil(Fraction(a + alpha, tsk.period))
         for tsk, alpha in zip(tsks, alphas)]

    def solve_linear_relaxation() -> Tuple[Optional[Fraction], bool]:
        """Solve linear relaxation of the integer program as described in Sec.
        5.5, https://arxiv.org/abs/2210.11185; returns the optimal objective
        value and a boolean indicating whether the solution is integral.

        """
        utils = [tsk.utilization for tsk in tsks]
        sum_util = sum(utils)
        assert sum_util <= 1
        num = beta + dot(alphas, utils)
        den = 1 - sum_util
        if num > 0 and den == 0: # the relaxation is infeasible
            return None, False

        # sort the problem instance in nonincreasing order using the key
        # x * tsk.period - alpha
        ys = [alpha - x * tsk.period for alpha, x, tsk in zip(alphas, xs, tsks)]
        pi = argsort(ys)

        # find the local, and hence global, maximum of f, as described in
        # Sec. 5.6
        local_max = None
        for idx in pi:
            num += xs[idx] * tsks[idx].wcet - alphas[idx] * utils[idx]
            den += utils[idx]
            v = Fraction(num, den)
            if local_max is not None and v < local_max:
                return local_max, False
            local_max = v
        return local_max, True

    while True:
        s, integral = solve_linear_relaxation()

        if perf is not None:
            perf.num_iterations += 1

        if s:
            # the trivial case
            if s <= a:
                return a

            # instance is infeasible because relaxation is infeasible
            if s > b:
                return None

            # solution is integral and feasible, and hence optimal
            if integral:
                break

            # solution is not integral; generate cutting planes
            xs = [math.ceil((s + alpha) / tsk.period)
                  for tsk, alpha in zip(tsks, alphas)]
        else:
            return None

    return math.ceil(s)


METHODS = {
    'fp': fixed_point, # fixed-point iteration
    'ip': integer_program, # integer programming
    'cp': cutting_plane  # specialized cutting-plane algorithm
}


def solve(tsks : List[Task], alphas: List[int], beta: int, a: int,
          b: int, perf=None, method: str = 'cp') -> Optional[int]:
    """Solve a kernel instance.

    Args:
        tsks: nonempty list of tasks
        alphas: list of integers of same length as tsks
        beta: an integer
        a, b: integer endpoints of interval [a, b]
              tsks, alphas, beta, a, b are the constants used to specify the
              kernel instance.
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
