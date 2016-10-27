#include <stdint.h>
#include <stdlib.h>

void genrl_volatile_add_f32(size_t n, const float *x, volatile float *y) {
  for (size_t p = 0; p < n; p++) {
    y[p] += x[p];
  }
}

void genrl_volatile_average_f32(size_t n, float alpha, const float *x, volatile float *y) {
  for (size_t p = 0; p < n; p++) {
    y[p] += alpha * (x[p] - y[p]);
  }
}
