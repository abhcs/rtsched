"""Run the experiments for comparing the performance of fixed-point iteration
and cutting-plane approaches.

"""

import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from attr import dataclass

import rtsched.sched_test.edf as edf
import rtsched.sched_test.fp as fp
from rtsched.system.generate_random import (generate_system_from_periods,
                                            generate_system_from_wcets)
from rtsched.system.task import Task


@dataclass
class Perf:
    num_iterations: int = 0
    time: float = 0


HUGE_INT = int(pow(10, 10))


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
    i = 0
    while i < num_systems:
        tsks = generate_system()
        perf1 = Perf()
        perf2 = Perf()

        r1 = solve(tsks, method1, perf1)
        r2 = solve(tsks, method2, perf2)
        if r1 != r2:
            assert r1 is not None and r2 is not None, f'r1 = {r1}, r2 = {r2}'
            continue
        data1[i] = perf1.time
        data2[i] = perf2.time
        i += 1
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


def two_histograms(data1: np.ndarray, label1: str, data2: np.ndarray,
                   label2: str, xlabel: str, ylabel: str, bins: str,
                   image_fn: Optional[str]):
    """Draw histograms of performance data for the two methods.

    Args:

        data1, label1: data and label for 1st histogram.

        data2, label2: data and label for 2nd histogram.

        xlabel: label for x-axis.

        ylabel: label for y-axis

        bins: either 'max' or one of the bin options mentioned here:
    https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

        image_fn: filename of pdf file that stores the histograms. if name is
                  None, then the image is simply displayed.

    """
    if bins == 'max':
        a = min(0, np.amin(data1), np.amin(data2))
        b = max(np.amax(data1), np.amax(data2)) + 1
        bins = np.arange(start=a, stop=b, step=1)
    plt.hist(data1, bins, label=label1, alpha=0.5, density=True)
    plt.hist(data2, bins, label=label2, alpha=0.5, density=True)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(loc='upper right', prop={'size': 12})
    plt.tight_layout()
    if image_fn:
        plt.savefig(f"{image_fn}.pdf", format="pdf")
    else:
        plt.show()
    plt.clf()


def histogram(data: np.ndarray, xlabel: str, ylabel: str, bins: str,
              image_fn: Optional[str]):
    """Draw histogram of data.

    Args:
        data: 1-D numpy array containing the data

        xlabel: label for x-axis.

        ylabel: label for y-axis

        bins: either 'max' or one of the bin options mentioned here:
    https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

        image_fn: filename of pdf file that stores the histogram. if name is
                  None, then the image is simply displayed.

    """
    if bins == 'max':
        a = min(0, np.amin(data))
        b = np.amax(data) + 1
        bins = np.arange(start=a, stop=b, step=1)
    plt.hist(data, bins, alpha=0.5, density=True)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.tight_layout()
    if image_fn:
        plt.savefig(f"{image_fn}.pdf", format="pdf")
    else:
        plt.show()
    plt.clf()


def analyze(data1: np.ndarray,
            label1: str,
            data2: np.ndarray,
            label2: str,
            perf_label: str,
            dir_name: pathlib.Path,
            show_stats: bool = False):
    """Analyze the performance data.

    Args:

        data1, label1: data and label for 1st algorithm.

        data2, label2: data and label for 2nd algorithm.

        perf_label: label for performance metric

        dir_name: name of directory where the histogram files are stored.

        show_stats: show descriptive statistics of performance data; False by
                    default.

    """
    if show_stats:
        print(f'statistics for {perf_label} of {label1}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data1))
        print()
        print(f'statistics for {perf_label} of {label2}')
        print('-' * 80)
        print()
        print(scipy.stats.describe(data2))
        print()
    two_histograms(data1,
                   label1,
                   data2,
                   label2,
                   xlabel=f'{perf_label}',
                   ylabel='normalized frequencies',
                   bins='max',
                   image_fn=dir_name / 'two')
    ratio = data1 / data2
    if show_stats:
        msg = f'statistics for ratio of {perf_label} of '\
            f'{label1} to {label2}'
        print(msg)
        print('-' * 80)
        print()
        print(scipy.stats.describe(ratio))
        print()
    histogram(data=ratio,
              xlabel=f'ratio of {perf_label} of\n {label1} '\
              f'to {label2}', ylabel='normalized frequencies', bins='max',
              image_fn=dir_name / 'ratio')


def gen_fp_system_from_periods():
    n = 25
    rng = np.random.default_rng(seed=1234)
    tsks = generate_system_from_periods(rng,
                                        n - 1,
                                        min_period=10,
                                        max_period=pow(10, 6),
                                        sum_util=0.8,
                                        deadline_type='implicit',
                                        sum_dens=0.8,
                                        max_jitter=0,
                                        abs_tol=n * 0.001)
    # this lowest priority task puts the methods being evaluated under
    # maximum stress
    tsks.append(Task(wcet=100, period=HUGE_INT))
    return tsks


def gen_fp_system_from_wcets():
    n = 25
    rng = np.random.default_rng(seed=1234)
    tsks = generate_system_from_wcets(rng,
                                      n - 1,
                                      min_wcet=1,
                                      max_wcet=1000,
                                      sum_util=0.9,
                                      deadline_type='implicit',
                                      sum_dens=0.9,
                                      max_jitter=0)
    # this lowest priority task puts the methods being evaluated under
    # maximum stress
    tsks.append(Task(wcet=100, period=HUGE_INT))
    return tsks


def exp_1():
    """Compare the number of iterations of fixed-point iteration and cutting
    planes for FP schedulability.

    """
    # name the directories in which you want to store the data and images.
    dir = pathlib.Path('exp1')
    dir.mkdir(exist_ok=True)

    # we used num_systems = 10000 in the actual experiment. use a smaller value
    # here to see if things are working properly first.
    data1, data2 = count_iterations(gen_fp_system_from_wcets,
                                    num_systems=10000,
                                    sched='FP',
                                    method1='fp',
                                    method2='cp',
                                    data_fn=(dir / 'data.npz'))

    analyze(data1=data1,
            label1='fixed point (RTA)',
            data2=data2,
            label2='cutting plane (CP-KERN)',
            perf_label='number of iterations',
            dir_name=dir,
            show_stats=True)


def exp_2():
    """Compare the running times of fixed-point iteration and cutting
    planes for FP schedulability.

    """
    # name the directories in which you want to store the data and images.
    dir = pathlib.Path('exp2')
    dir.mkdir(exist_ok=True)

    # we used num_systems = 10000 in the actual experiment. use a smaller value
    # here to see if things are working properly first.
    data1, data2 = measure_time(gen_fp_system_from_wcets,
                                num_systems=10000,
                                sched='FP',
                                method1='fp_cpp',
                                method2='cp_cpp',
                                data_fn=(dir / 'data.npz'))

    analyze(data1=data1,
            label1='fixed point (RTA)',
            data2=data2,
            label2='cutting plane (CP-KERN)',
            perf_label=r'CPU time ($\mu s$)',
            dir_name=dir,
            show_stats=True)


def gen_edf_system_from_wcets():
    rng = np.random.default_rng(seed=1234)
    tsks = generate_system_from_wcets(rng,
                                      n=25,
                                      min_wcet=1,
                                      max_wcet=1000,
                                      sum_util=0.9,
                                      deadline_type='constrained',
                                      sum_dens=2,
                                      max_jitter=0)
    return tsks


def exp_3():
    """Compare the number of iterations of fixed-point iteration and cutting
    planes for EDF schedulability.

    """
    # name the directories in which you want to store the data and images.
    dir = pathlib.Path('exp3')
    dir.mkdir(exist_ok=True)

    # we used num_systems = 10000 in the actual experiment. use a smaller value
    # here to see if things are working properly first.
    data1, data2 = count_iterations(gen_edf_system_from_wcets,
                                    num_systems=10000,
                                    sched='EDF',
                                    method1='fp',
                                    method2='cp',
                                    data_fn=(dir / 'data.npz'))

    analyze(data1=data1,
            label1='fixed point (QPA)',
            data2=data2,
            label2='cutting plane (CP-KERN)',
            perf_label='number of iterations',
            dir_name=dir,
            show_stats=True)


if __name__ == "__main__":
    exp_3()
