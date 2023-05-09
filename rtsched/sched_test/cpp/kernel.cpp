#include <_types/_uint64_t.h>
#include <math.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>

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
  const int reps = 3;
  for (int i = 0; i < reps; i++) {
    r = cutting_plane_inner(cs, ps, us, alphas, n, beta, a, b, feasible);
  }
  std::clock_t c_end = std::clock();
  *time = 1000000.0 * (c_end - c_start) / (reps * CLOCKS_PER_SEC);
  return r;
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
int64_t ys[N];
uint64_t u[N];
uint64_t v[N];

uint64_t cutting_plane_inner(uint64_t* cs, uint64_t* ps, double* us,
                             int64_t* alphas, uint64_t n, int64_t beta,
                             int64_t a, int64_t b, bool* feasible) {
  double p0 = beta;
  double q0 = 1;
  uint64_t* pi = u;
  for (uint64_t i = 0; i < n; i++) {
    ys[i] = ceil(a + alphas[i], ps[i]) * ps[i] - alphas[i];
    pi[i] = i;
    p0 += alphas[i] * us[i];
    q0 -= us[i];
  }
  // we assume bounded utilization!!
  // check trivial cases
  if (a > b) {
    *feasible = false;
    return a;
  }

  if (n == 0) {
    *feasible = true;
    return a;
  }

  std::sort(pi, pi + n, [](uint64_t a, uint64_t b) { return ys[a] > ys[b]; });
  if (p0 > q0 * ys[pi[0]]) {
    *feasible = false;
    return a;
  }

  int64_t t = a;
  int64_t prev_t = a;

  // cutting plane iteration
  while (true) {
    uint64_t k = pi[0];
    double p = p0 + us[k] * ys[k];
    double q = q0 + us[k];
    uint64_t i = 1;
    while (i < n) {
      k = pi[i];
      if (p > q * ys[k]) {
        t = std::ceil(p / q);
        break;
      }
      p += us[k] * ys[k];
      q += us[k];
      i += 1;
    }
    if (i == n) {
      t = std::ceil(p / q);
    }

    if (t <= a) {
      *feasible = true;
      return a;
    }

    if (t > b) {
      *feasible = false;
      return a;
    }

    if (i == n or t == prev_t) {
      *feasible = true;
      return t;
    }

    for (uint64_t j = i; j < n; j++) {
      auto k = pi[j];
      auto alpha = alphas[k];
      auto p = ps[k];
      ys[k] = ceil(t + alpha, p) * p - alpha;
    }

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
    k = 0;
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
