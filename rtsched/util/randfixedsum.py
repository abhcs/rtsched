"""A Python implementation of Roger Stafford's randfixedsum algorithm copied
from https://github.com/MaximeCheramy/simso. The only change we have made to
the function is that now we pass an rng as an argument to the function and
calls to np.random.* are replaced by rng.*. This change is made so that we
adhere to numpy's random number generation policy
(https://numpy.org/neps/nep-0019-rng-policy.html), which states: "The preferred
best practice for getting reproducible pseudorandom numbers is to instantiate a
generator object with a seed and pass it around. The implicit global
RandomState behind the numpy.random.* convenience functions can cause problems,
especially when threads or other forms of concurrency are involved. Global
state is always problematic. We categorically recommend avoiding using the
convenience functions when reproducibility is involved."

Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.
    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value
    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.

"""

import numpy as np


def randfixedsum(rng: np.random.Generator, n: int, sum_util, num_systems: int):
    """
    Args:
        rng: a random number generator from numpy like np.random.default_rng()

        n: number of tasks in the system

        sum_util: total utilization of the system

        num_systems: number of systems
    """
    assert n >= sum_util

    #deal with n=1 case
    if n == 1:
        return np.tile(np.array([sum_util]), [num_systems, 1])

    k = min(int(sum_util), n - 1)
    s = sum_util
    s1 = s - np.arange(k, k - n, -1.)
    s2 = np.arange(k + n, k, -1.) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    x = np.zeros((n, num_systems))
    rt = rng.uniform(size=(n - 1, num_systems))  # rand simplex type
    rs = rng.uniform(size=(n - 1, num_systems))  # rand position in simplex
    s = np.repeat(s, num_systems)
    j = np.repeat(k + 1, num_systems)
    sm = np.repeat(0, num_systems)
    pr = np.repeat(1, num_systems)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
        sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
        sm = sm + (1.0 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    #iterated in fixed dimension order but needs to be randomised
    #permute x row order within each column
    for i in range(0, num_systems):
        x[..., i] = x[rng.permutation(n), i]

    return x.T.tolist()
