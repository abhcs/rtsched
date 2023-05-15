# CUTTING-PLANE ALGORITHMS FOR PREEMPTIVE UNIPROCESSOR REAL-TIME SCHEDULING PROBLEMS

This repository is the official implementation of [Cutting-plane algorithms for
  preemptive uniprocessor real-time scheduling
  problems](https://arxiv.org/abs/2210.11185).

In this paper, we show that fixed-point iteration algorithms like RTA and QPA
are suboptimal cutting-plane algorithms for specific integer programming
formulations of fixed-priority and earliest-deadline-first schedulability
problems; the optimal algorithm, discovered in this paper, converges more
quickly to the solution:

![compare fixed-point iteration to cutting-plane method](docs/images/fp_cp_compare.png)

This repository is primarily intended to be a companion to the arxiv paper. It
contains
- a simple implementation of a hard real-time task (see
  [task.py](rtsched/system/task.py));
- methods for generating random systems of hard real-time tasks (see
  [generate_random.py](rtsched/system/generate_random.py));
- instrumented implementations of FP-KERN, IP-KERN, and CP-KERN that count the
  number of iterations used by the algorithm (see
  [kernel.py](rtsched/sched_test/kernel.py));
- instrumented implementations of FP-KERN and CP-KERN that measure the CPU time
  used by the algorithms (see[kernel.cpp](rtsched/sched_test/cpp/kernel.cpp));
- the reduction from FP schedulability to the kernel described in
  Appendix A (see [fp.py](rtsched/sched_test/fp.py));
- the reduction from EDF schedulability to the kernel described in Appendix B
  (see [edf.py](rtsched/sched_test/edf.py));
- tests for finding inconsistencies between CP-KERN and traditional
  schedulability tests (RTA and QPA) (see [tests](rtsched/tests)); and
- a script for running the experiments and generating the associated data and
  images (see [exp.py](./exp.py)).

We depend on several Python packages including docplex, drs, numpy, matplotlib,
pytest, and scipy. Our C++ implementation of CP-KERN can produce incorrect
answers due to numerical issues stemming from the use of floating-point
arithmetic and integer overflow errors. In contrast, our Python implementation
of CP-KERN uses rational arithmetic using the fractions module and
unlimited-precision integers. Thus, we think that the Python implementation is
more trustworthy. In all experiments, we check that the results produced by
FP-KERN and CP-KERN are consistent; thus, we have confidence in the validity of
the results derived from the C++ implementations. The repository should serve as
a useful starting point for someone interested in developing schedulability
tests that run even more quickly (maybe use a language with good support for
parallel programming).

## Requirements

1. Install [hatch](https://hatch.pypa.io/latest/install/).

2. Clone this repository.

3. Make a virtual Python 3.8 environment. I use
   [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) but feel free
   to use your favorite method.

4. Activate the virtual environment.

5. Install the rtsched package:

```
pip install .
```

6. Build the cpp library using
```
cd rtsched/sched_test && mkdir build && cd build && cmake ../cpp && make
```

Pass the argument `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` to `cmake` to generate
`compile_commands.json`.

## Testing

Test the correctness of the implementation:

```
pytest --doctest-modules
```

## Evaluation

Run the experiments:

```
python exp.py
```

Basic Python literacy is needed to understand and modify the file to get the
behavior you want. By default, the script runs the experiment described in
Figures 2 in the paper.

## Results

The results are described in Section 6 of the paper, and they should match the
results produced by this code.

## History

Version 0.1.2 (2023-05-15)

## Credits

Abhishek Singh

## DOI

[![DOI](https://zenodo.org/badge/604422991.svg)](https://zenodo.org/badge/latestdoi/604422991)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Contributing

Feel free to open an issue.

## Other useful real-time systems projects

The following projects have broader scopes than ours, and may be useful to
people who have stumbled onto this page.

- [rtsim](http://rtsim.sssup.it/)
- [schedcat](https://github.com/brandenburg/schedcat)
- [simso](https://github.com/MaximeCheramy/simso)
