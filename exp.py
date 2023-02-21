"""Run the experiments described in Sec. 6, https://arxiv.org/abs/2210.11185.

"""

import pathlib
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from attr import dataclass

import rtsched.sched_test.edf as edf
import rtsched.sched_test.fp as fp
from rtsched.system.generate_random import generate_system
from rtsched.system.task import Task


@dataclass
class Perf:
    num_iterations: int = 0

HUGE_INT = int(pow(10,10))
LABEL = {'fp': 'fixed point',
         'ip': 'integer program',
         'cp': 'cutting plane'}


def run_fp(seed, num_systems: int, n: int, max_wcet: int, sum_util, max_jitter:
           int, method1: str, method2: str, data_fn: str):
    """Compare performance of two methods for solving FP schedulability.

    Args:

        seed: a seed to initialize the random number generator. see
              https://numpy.org/doc/stable/reference/random/generator.html.

        num_systems: number of randomly generated systems

        n: number of tasks in a system

        max_wcet: maximum wcet of any task in a system

        sum_util: total utilization of a system

        max_jitter: maximum jitter of any task in the system.

        method1, method2: the two methods used to solve the FP schedulability
                          problem for the randomly generated systems. each
                          method must be in ['fp', 'ip', 'cp'].

        data_fn: filename of npz file that will store the performance
                 measurements. if the name is None, then the data is not
                 stored; providing a filename is recommended, especially for
                 long experiments.

    Returns: (data1, data2): two numpy arrays containing the number of
        iterations used by each method.

    This experiment is described in greater detail in Section 6.1,
    https://arxiv.org/abs/2210.11185.

    """
    rng = np.random.default_rng(seed)
    data1 = np.zeros(num_systems)
    data2 = np.zeros(num_systems)
    for i in range(num_systems):
        tsks = generate_system(rng, n-1, max_wcet, sum_util,
                               deadline_type='implicit', sum_dens=sum_util,
                               max_jitter=max_jitter)
        # this lowest priority task puts the methods being evaluated under
        # maximum stress
        tsks.append(Task(wcet=rng.integers(low=1, high=max_wcet+1).item(),
                         period=HUGE_INT))
        perf1 = Perf()
        perf2 = Perf()

        assert fp.solve(tsks, method1, perf1) == fp.solve(tsks, method2, perf2)

        data1[i] = perf1.num_iterations
        data2[i] = perf2.num_iterations
    if data_fn:
        np.savez(data_fn, data1=data1, data2=data2, method1=method1,
                 method2=method2)
    return data1, data2


def run_edf(seed: int, num_systems: int, n: int, max_wcet: int, sum_util,
        deadline_type: str, sum_dens, max_jitter: int, method1: str,
        method2: str, data_fn: str):
    """Compare performance of two methods for solving EDF schedulability.

    Args:

        seed: a seed to initialize the random number generator. see
              https://numpy.org/doc/stable/reference/random/generator.html.

        num_systems: number of randomly generated systems

        n: number of tasks in a system

        max_wcet: maximum wcet of any task in a system

        sum_util: total utilization of a system

        deadline_type: the type of deadlines in a system. the type must be in
                       ['implicit', 'constrained', 'arbitrary'].

        sum_dens: total density of the system, where density of a task is equal
                  to the ratio of its wcet to its deadline.

        max_jitter: maximum jitter of any task in the system.

        method1, method2: the two methods used to solve the FP schedulability
                          problem for the randomly generated systems. each
                          method must be in ['fp', 'ip', 'cp'].

        data_fn: filename of npz file that will store the performance
                 measurements. if the name is None, then the data is not
                 stored; providing a filename is recommended, especially for
                 long experiments.

    Returns: (data1, data2): two numpy arrays containing the number of
        iterations used by each method.

    This experiment is described in greater detail in Section 6.2,
    https://arxiv.org/abs/2210.11185.

    """
    rng = np.random.default_rng(seed)
    data1 = np.zeros(num_systems)
    data2 = np.zeros(num_systems)
    for i in range(num_systems):
        tsks = generate_system(rng, n, max_wcet, sum_util, deadline_type,
                               sum_dens, max_jitter)
        perf1 = Perf()
        perf2 = Perf()

        assert edf.solve(tsks, method1, perf1) == edf.solve(tsks, method2,
                                                            perf2), str(tsks)

        data1[i] = perf1.num_iterations
        data2[i] = perf2.num_iterations
    if data_fn:
        np.savez(data_fn, data1=data1, data2=data2, method1=method1,
                 method2=method2)
    return data1, data2


def load(data_fn):
    """Load data from npz file.

    Args:
        data_fn: the filename

    Returns: (method1, method2, data1, data2)
    """
    npzfile = np.load(data_fn)
    method1 = str(npzfile['method1'])
    method2 = str(npzfile['method2'])
    data1 = npzfile['data1']
    data2 = npzfile['data2']
    return method1, method2, data1, data2


# use options such as these to adjust various aspects of the image containing
# the histograms such as the label size for the ticks and colors for the bars.
# Options can also be set inside the function two_histograms(...). consult the
# matplotlib 3.7.0 documentation to find more ways to adjust the final image to
# your liking.
plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.style.use('tableau-colorblind10')


def two_histograms(method1: str, method2: str, data1: np.ndarray, data2:
                   np.ndarray, image_fn: Optional[str]):
    """Draw histograms of performance data for the two methods.

    Args:

        method1, method2: the two methods being evaluated. each method must be
                          in ['fp', 'ip', 'cp'].

        data1, data2: the numpy arrays containing the performance data for the
                      two methods. these arrays are either generated directly
                      from run_fp() or run_edf(), or loaded from an npz file
                      containing data from a run in the past using load().

        image_fn: filename of pdf file that stores the histograms. if name is
                  None, then the image is simply displayed.

    """
    m1 = np.amax(data1)
    m2 = np.amax(data2)
    m = max(m1, m2)+1
    bins = np.arange(start=0, stop=m, step=1)
    plt.hist(data1, bins, label=LABEL[method1], alpha=0.5, density=True)
    plt.hist(data2, bins, label=LABEL[method2], alpha=0.5, density=True)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Normalized Frequencies", fontsize=20)
    plt.legend(loc='upper right', prop={'size': 12})
    if image_fn:
        plt.savefig(f"{image_fn}.pdf", format="pdf")
    else:
        plt.show()
    plt.clf()


def analyze(method1: str, method2: str, data1: np.ndarray, data2: np.ndarray,
            image_fn: Optional[str], show_stats: bool = False):
    """Analyze the performance data.

    Args:

        method1, method2: the two methods being evaluated. each method must be
                          in ['fp', 'ip', 'cp'].

        data1, data2: the numpy arrays containing the performance data for the
                      two methods. these arrays are either generated directly
                      from run_fp() or run_edf(), or loaded from an npz file
                      containing data from a run in the past using load().

        image_fn: filename of pdf file that stores the histograms. if name is
                  None, then the image is simply displayed.

        show_stats: show descriptive statistics of performance data; False by
                    default.

    """
    if show_stats:
        print(f'stats for {LABEL[method1]}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data1))
        print()
        print(f'stats for {LABEL[method2]}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data2))
        print()
    two_histograms(method1, method2, data1, data2, image_fn)

if __name__ == "__main__":
    # choose the methods you want to compare. In
    # https://arxiv.org/abs/2210.11185, we choose 'fp' and 'cp' but 'ip' is
    # also available if you have cplex installed on your system.
    method1 = 'fp'
    method2 = 'cp'

    # the FP schedulability experiment in Sec 6.1, Fig. 2. The following
    # vectors can be edited to produce any configuration mentined in Sec. 6, or
    # any other configuration you want.
    ns = [25]
    us = [0.7, 0.8, 0.9, 0.99]

    # # name the directories in which you want to store the data and images.
    pathlib.Path('fp_exp1_data').mkdir(exist_ok=True)
    pathlib.Path('fp_exp1_images').mkdir(exist_ok=True)

    for u, n in product(us, ns):
        data_fn = f'fp_exp1_data/n_{n}_u_{u:.2f}.npz'
        image_fn = f'fp_exp1_images/n_{n}_u_{u:.2f}'

        print(f'FP schedulability, total utilization = {u:.2f}, {n = }')

      # we used num_systems = 10000 in the actual experiment but the code
      # executed for several hours. use a smaller value here to see if things
      # are working properly first.
        data1, data2 = run_fp(seed=1234, num_systems=100, n=n, max_wcet=1000,
                              sum_util=u, max_jitter=0, method1=method1,
                              method2=method2, data_fn=data_fn)

        analyze(method1, method2, data1, data2, image_fn=image_fn, show_stats=True)


    # the EDF schedulability experiment in Sec 6.2, Fig. 4. The following
    # vectors can be edited to produce any configuration mentined in Sec. 6, or
    # any other configuration you want.
    ns = [50]
    us = [0.7, 0.8, 0.9, 0.99]
    ds = [1.75]

    # name the directories in which you want to store the data and images.
    pathlib.Path('edf_exp1_data').mkdir(exist_ok=True)
    pathlib.Path('edf_exp1_images').mkdir(exist_ok=True)

    for u, n, d in product(us, ns, ds):
        data_fn = f'edf_exp1_data/n_{n}_u_{u:.2f}_d_{d:.2f}.npz'
        image_fn = f'edf_exp1_images/n_{n}_u_{u:.2f}_d_{d:.2f}'

        print(f'EDF schedulability, {n = }, total utilization = {u:.2f},'
              f' total density = {d:.2f}')

        # we used num_systems = 10000 in the actual experiment but the code
        # executed for several hours. use a smaller value here to see if things
        # are working properly first.
        data1, data2 = run_edf(seed=1234, num_systems=100, n=n, max_wcet=1000,
                               sum_util=u, deadline_type='constrained',
                               sum_dens=d, max_jitter=0, method1=method1,
                               method2=method2, data_fn=data_fn)

        analyze(method1, method2, data1, data2, image_fn=image_fn, show_stats=True)
