#include <_types/_uint64_t.h>
#include <math.h>
#include <sys/_types/_int64_t.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>

uint64_t fixed_point_inner(uint64_t* cs, uint64_t* ps, int64_t* alphas,
                           uint64_t n, int64_t beta, int64_t a, int64_t b,
                           bool* feasible);

uint64_t cutting_plane_inner(uint64_t* cs, uint64_t* ps, double* us,
                             int64_t* alphas, uint64_t n, int64_t beta,
                             int64_t a, int64_t b, bool* feasible);

extern "C" uint64_t fixed_point(uint64_t* cs, uint64_t* ps, int64_t* alphas,
                                uint64_t n, int64_t beta, int64_t a, int64_t b,
                                bool* feasible, double* time) {
  std::clock_t c_start = std::clock();
  uint64_t r = a;
  const int reps = 3;
  for (int i = 0; i < reps; i++) {
    r = fixed_point_inner(cs, ps, alphas, n, beta, a, b, feasible);
  }
  std::clock_t c_end = std::clock();
  *time = 1000000.0 * (c_end - c_start) / (reps * CLOCKS_PER_SEC);
  return r;
}

extern "C" uint64_t cutting_plane(uint64_t* cs, uint64_t* ps, double* us,
                                  int64_t* alphas, uint64_t n, int64_t beta,
                                  int64_t a, int64_t b, bool* feasible,
                                  double* time) {
  std::clock_t c_start = std::clock();
  uint64_t r = a;
  const int reps = 1;
  for (int i = 0; i < reps; i++) {
    r = cutting_plane_inner(cs, ps, us, alphas, n, beta, a, b, feasible);
  }
  std::clock_t c_end = std::clock();
  *time = 1000000.0 * (c_end - c_start) / (reps * CLOCKS_PER_SEC);
  return r;
}

// compute ceil(n / d) assuming that d > 0
int64_t ceil(int64_t n, int64_t d) {
  if (d == 0)
    throw std::invalid_argument("invalid divisor");

  int64_t q = n / d;
  if (n % d == 0 || n <= 0)
    return q;
  return q + 1;
}

uint64_t fixed_point_inner(uint64_t* cs, uint64_t* ps, int64_t* alphas,
                           uint64_t n, int64_t beta, int64_t a, int64_t b,
                           bool* feasible) {
  // check trivial cases
  if (a > b) {
    *feasible = false;
    return a;
  }

  if (n == 0) {
    *feasible = true;
    return a;
  }

  int64_t t = a;

  // fixed point iteration
  do {
    int64_t v = beta;
    for (uint64_t i = 0; i < n; i++) {
      v += ceil(t + alphas[i], ps[i]) * cs[i];
    }
    if (v <= t) {
      *feasible = true;
      return t;
    }
    t = v;
  } while (t <= b);

  // problem is infeasible
  *feasible = false;
  return a;
}

const int N = 100;
int64_t xs[N];
int64_t ys[N];
uint64_t u[N];
uint64_t v[N];

uint64_t cutting_plane_inner(uint64_t* cs, uint64_t* ps, double* us,
                             int64_t* alphas, uint64_t n, int64_t beta,
                             int64_t a, int64_t b, bool* feasible) {
  if (a > b) {
    *feasible = false;
    return a;
  }
  if (n == 0) {
    *feasible = true;
    return a;
  }
  int64_t r = beta;
  uint64_t* pi = u;
  for (uint64_t i = 0; i < n; i++) {
    xs[i] = ceil(a + alphas[i], ps[i]);
    ys[i] = xs[i] * ps[i] - alphas[i];
    pi[i] = i;
    r += cs[i] * xs[i];
  }
  // we assume bounded utilization!!
  if (r <= a) {
    *feasible = true;
    return a;
  }
  const uint64_t i0 = 0;
  std::sort(pi, pi + n, [](uint64_t a, uint64_t b) { return ys[a] > ys[b]; });
  int64_t t = r;
  int64_t prev_t = r;

  while (true) {
    double p = r;
    double q = 1.0;
    uint64_t i = n - 1;
    while (i > i0) {
      auto k = pi[i];
      if (p <= q * ys[k]) {
        t = std::ceil(p / q);
        break;
      }
      p -= (double)cs[k] * xs[k] - us[k] * alphas[k];
      q -= us[k];
      i -= 1;
    }
    if (i == i0) {
      t = std::ceil(p / q);
    }
    if (t > b) {
      *feasible = false;
      return a;
    }
    if (i == n - 1 or t == prev_t) {
      *feasible = true;
      return t;
    }

    for (uint64_t j = i + 1; j < n; j++) {
      auto k = pi[j];
      uint64_t d = ceil(t + alphas[k], ps[k]) - xs[k];
      xs[k] += d;
      ys[k] += ps[k] * d;
      r += cs[k] * d;
    }

    i += 1;
    std::sort(pi + i, pi + n,
              [](uint64_t a, uint64_t b) { return ys[a] > ys[b]; });
    uint64_t* other = v;
    if (pi == v) {
      other = u;
    }

    uint64_t* l1 = pi;
    uint64_t* l2 = pi + i;
    uint64_t i1 = 0;
    uint64_t i2 = 0;
    uint64_t k = 0;
    while (i1 < i and i2 < n - i) {
      if (ys[l1[i1]] >= ys[l2[i2]]) {
        other[k] = l1[i1];
        i1 += 1;
      } else {
        other[k] = l2[i2];
        i2 += 1;
      }
      k += 1;
    }
    if (i1 == i) {
      while (i2 < n - i) {
        other[k] = l2[i2];
        i2 += 1;
        k += 1;
      }
    } else {
      while (i1 < i) {
        other[k] = l1[i1];
        i1 += 1;
        k += 1;
      }
    }
    pi = other;
    prev_t = t;
  }

  // problem is infeasible
  *feasible = false;
  return a;
}
