"""Run the experiments for comparing the performance of fixed-point iteration
and cutting-plane approaches.

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
    time: float = 0


HUGE_INT = int(pow(10, 10))
SOLVE_LABEL = {
    'fp': 'fixed point',
    'ip': 'integer program',
    'cp': 'cutting plane',
    'fp_cpp': 'fixed point (C++)',
    'cp_cpp': 'cutting plane (C++)'
}
PERF_LABEL = {
    'iterations': 'number of iterations',
    'time': 'CPU time (microseconds)'
}


def count_iterations(generate_system, num_systems: int, sched: str,
                     method1: str, method2: str, data_fn: str):
    """Count the number of iterations of two methods for solving FP or EDF
    schedulability.

    Args:

        generate_system: a function that returns a random system

        num_systems: number of randomly generated systems

        sched: the type of scheduler. must be 'FP' or 'EDF'.

        method1, method2: the two methods used to solve the FP schedulability
                          problem for the randomly generated systems. each
                          method must be in ['fp', 'ip', 'cp'].

        data_fn: filename of npz file that will store the performance
                 measurements. if the name is None, then the data is not
                 stored; providing a filename is recommended, especially for
                 long experiments.

    Returns: (data1, data2): two numpy arrays containing the number of
        iterations used by each method.

    """
    assert method1 in ['fp', 'ip', 'cp']
    assert method2 in ['fp', 'ip', 'cp']
    solve = fp.solve
    if sched == 'EDF':
        solve = edf.solve
    data1 = np.zeros(num_systems)
    data2 = np.zeros(num_systems)
    for i in range(num_systems):
        tsks = generate_system()
        perf1 = Perf()
        perf2 = Perf()

        assert solve(tsks, method1, perf1) == solve(tsks, method2, perf2)

        data1[i] = perf1.num_iterations
        data2[i] = perf2.num_iterations
    if data_fn:
        np.savez(data_fn,
                 data1=data1,
                 data2=data2,
                 method1=method1,
                 method2=method2)
    return data1, data2


def measure_time(generate_system, num_systems: int, sched: str, method1: str,
                 method2: str, data_fn: str):
    """Measure running time of two methods for solving FP schedulability.

    Args:

        generate_system: a function that returns a random system

        num_systems: number of randomly generated systems

        sched: the type of scheduler. must be 'FP' or 'EDF'.

        method1, method2: the two methods used to solve the FP schedulability
                          problem for the randomly generated systems. each
                          method must be in ['fp_cpp', 'cp_cpp'].

        data_fn: filename of npz file that will store the performance
                 measurements. if the name is None, then the data is not
                 stored; providing a filename is recommended, especially for
                 long experiments.

    Returns: (data1, data2): two numpy arrays containing the running times of
    each method.

    """
    assert method1 in ['fp_cpp', 'cp_cpp']
    assert method2 in ['fp_cpp', 'cp_cpp']
    solve = fp.solve
    if sched == 'EDF':
        solve = edf.solve
    data1 = np.zeros(num_systems)
    data2 = np.zeros(num_systems)
    for i in range(num_systems):
        tsks = generate_system()
        perf1 = Perf()
        perf2 = Perf()

        r1 = solve(tsks, method1, perf1)
        r2 = solve(tsks, method2, perf2)
        assert (r1 == None) == (r2 == None)

        data1[i] = perf1.time
        data2[i] = perf2.time
    if data_fn:
        np.savez(data_fn,
                 data1=data1,
                 data2=data2,
                 method1=method1,
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


#use options such as these to adjust various aspects of the image containing
#the histograms such as the label size for the ticks and colors for the bars.
#Options can also be set inside the function two_histograms(...).consult the
#matplotlib 3.7.0 documentation to find more ways to adjust the final image to
#your liking.
plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.style.use('tableau-colorblind10')


def two_histograms(method1: str, method2: str, data1: np.ndarray,
                   data2: np.ndarray, xlabel: str, image_fn: Optional[str]):
    """Draw histograms of performance data for the two methods.

    Args:

        method1, method2: the two methods being evaluated. each method must be
                          in ['fp', 'ip', 'cp', 'fp_cpp', 'cp_cpp'].

        data1, data2: the numpy arrays containing the performance data for the
                      two methods. these arrays are either generated directly
                      from run_fp() or run_edf(), or loaded from an npz file
                      containing data from a run in the past using load().

        xlabel: label on x-axis.

        image_fn: filename of pdf file that stores the histograms. if name is
                  None, then the image is simply displayed.

    """
    m1 = np.amax(data1)
    m2 = np.amax(data2)
    m = max(m1, m2) + 1
    bins = np.arange(start=0, stop=m, step=1)
    plt.hist(data1, bins, label=SOLVE_LABEL[method1], alpha=0.5, density=True)
    plt.hist(data2, bins, label=SOLVE_LABEL[method2], alpha=0.5, density=True)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("normalized frequencies", fontsize=20)
    plt.legend(loc='upper right', prop={'size': 12})
    if image_fn:
        plt.savefig(f"{image_fn}.pdf", format="pdf")
    else:
        plt.show()
    plt.clf()


def histogram(xlabel: str, data: np.ndarray, image_fn: Optional[str]):
    """Draw histogram of data.

    Args:
        xlabel: label on x-axis.

        data: the numpy array containing the data

        image_fn: filename of pdf file that stores the histogram. if name is
                  None, then the image is simply displayed.

    """
    #m = np.amax(data) + 1
    #bins = np.arange(start = 0, stop = m, step = 1)
    plt.hist(data, alpha=0.5, density=True)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("normalized frequencies", fontsize=20)
    if image_fn:
        plt.savefig(f"{image_fn}.pdf", format="pdf")
    else:
        plt.show()
    plt.clf()


def analyze(method1: str,
            method2: str,
            perf: str,
            data1: np.ndarray,
            data2: np.ndarray,
            dir: Optional[str],
            show_stats: bool = False):
    """Analyze the performance data.

    Args:

        method1, method2: the two methods being evaluated. each method must be
                          in ['fp', 'ip', 'cp'].

        perf: the performance metric used in the evaluation. perf must be in
              ['iterations', 'time'].

        data1, data2: the numpy arrays containing the performance data for the
                      two methods. these arrays are either generated directly
                      from run_fp() or run_edf(), or loaded from an npz file
                      containing data from a run in the past using load().

        show_stats: show descriptive statistics of performance data; False by
                    default.

    """
    if show_stats:
        print(f'statistics for {PERF_LABEL[perf]} of {SOLVE_LABEL[method1]}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data1))
        print()
        print(f'statistics for {PERF_LABEL[perf]} of {SOLVE_LABEL[method2]}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data2))
        print()
    two_histograms(method1,
                   method2,
                   data1,
                   data2,
                   xlabel=f'{PERF_LABEL[perf]}',
                   image_fn=dir / 'two')
    ratio = data1 / data2
    if show_stats:
        msg = f'statistics for ratio of {PERF_LABEL[perf]} of '\
            f'{SOLVE_LABEL[method2]} to {SOLVE_LABEL[method1]}'
        print(msg)
        print('-' * 80)
        print()
        print(scipy.stats.describe(ratio))
        print()
    histogram(data=ratio,
              xlabel=f'ratio of {PERF_LABEL[perf]} of {SOLVE_LABEL[method1]}'\
              'to {SOLVE_LABEL[method2]}', image_fn=dir / 'ratio')


def fp_sched_exp_1():
    """Compare the number of iterations of fixed-point iteration and cutting
    planes for FP schedulability.

    """
    # name the directories in which you want to store the data and images.
    dir = pathlib.Path('fp_sched_exp1')
    dir.mkdir(exist_ok=True)

    n = 50
    rng = np.random.default_rng(seed=1234)

    def gen():
        tsks = generate_system(rng,
                               n - 1,
                               min_wcet=1,
                               max_wcet=1000,
                               min_period=None,
                               max_period=None,
                               sum_util=0.95,
                               deadline_type='implicit',
                               sum_dens=0.95,
                               max_jitter=0,
                               method='wcets')
        # this lowest priority task puts the methods being evaluated under
        # maximum stress
        tsks.append(
            Task(wcet=rng.integers(low=1, high=1000).item(), period=HUGE_INT))
        return tsks

    # we used num_systems = 10000 in the actual experiment but the code
    # executed for several hours.use a smaller value here to see if things are
    # working properly first.
    data1, data2 = count_iterations(gen,
                                    num_systems=100,
                                    sched='FP',
                                    method1='fp',
                                    method2='cp',
                                    data_fn=(dir / 'data.npz'))

    analyze('fp', 'cp', 'iterations', data1, data2, dir, show_stats=True)


def fp_sched_exp_2():
    """Compare the running times of fixed-point iteration and cutting
    planes for FP schedulability.

    """
    # name the directories in which you want to store the data and images.
    dir = pathlib.Path('fp_sched_exp2')
    dir.mkdir(exist_ok=True)

    n = 50
    rng = np.random.default_rng(seed=1234)

    def gen():
        tsks = generate_system(rng,
                               n - 1,
                               min_wcet=None,
                               max_wcet=None,
                               min_period=10,
                               max_period=pow(10, 6),
                               sum_util=0.80,
                               deadline_type='implicit',
                               sum_dens=0.80,
                               max_jitter=0,
                               method='periods')
        # this lowest priority task puts the methods being evaluated under
        # maximum stress
        tsks.append(Task(wcet=1000, period=HUGE_INT))
        return tsks

    # we used num_systems = 10000 in the actual experiment but the code
    # executed for several hours.use a smaller value here to see if things are
    # working properly first.
    data1, data2 = measure_time(gen,
                                num_systems=10000,
                                sched='FP',
                                method1='fp_cpp',
                                method2='cp_cpp',
                                data_fn=(dir / 'data.npz'))

    analyze('fp', 'cp', 'time', data1, data2, dir, show_stats=True)


if __name__ == "__main__":
    fp_sched_exp_2()
