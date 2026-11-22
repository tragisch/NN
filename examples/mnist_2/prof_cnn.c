/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Profile the MNIST CNN conv layers directly against mMatrix backends. */
#include "st.h"
#include "st_conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void run_case(const char *label, size_t n, size_t c_in, size_t h, size_t w,
                     size_t c_out, size_t k, size_t stride, size_t pad,
                     size_t out_h, size_t out_w) {
  FloatTensor *in = st_create(4, (size_t[]){n, c_in, h, w});
  FloatTensor *wt = st_create(4, (size_t[]){c_out, c_in, k, k});
  FloatTensor *b = st_create(1, (size_t[]){c_out});
  FloatTensor *out = st_create(4, (size_t[]){n, c_out, out_h, out_w});
  for (size_t i = 0; i < n * c_in * h * w; i++) in->values[i] = (float)((i * 1103515245u) >> 16) / 32768.0f - 1.0f;
  for (size_t i = 0; i < c_out * c_in * k * k; i++) wt->values[i] = (float)((i * 1103515245u) >> 16) / 32768.0f - 1.0f;
  for (size_t i = 0; i < c_out; i++) b->values[i] = 0.1f;

  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = stride;
  p.pad_h = p.pad_w = pad;

  /* warmup */
  st_conv2d_nchw(in, wt, b, &p, out);
  double best = 1e9, total = 0;
  int reps = 30;
  for (int r = 0; r < reps; r++) {
    double t0 = now_ms();
    st_conv2d_nchw(in, wt, b, &p, out);
    double dt = now_ms() - t0;
    if (dt < best) best = dt;
    total += dt;
  }
  printf("%-28s backend=%-10s best=%7.3f ms  mean=%7.3f ms\n", label,
         st_conv2d_last_backend(), best, total / reps);
  st_destroy(in); st_destroy(wt); st_destroy(b); st_destroy(out);
}

int main(void) {
  /* MNIST CNN: conv1 [100,1,28,28] -> 8 filters 3x3 s1 p1 -> [100,8,28,28]
   *            conv2 [100,8,14,14] -> 16 filters 3x3 s1 p1 -> [100,16,14,14] */
  run_case("conv1 100x1x28x28->8 3x3", 100, 1, 28, 28, 8, 3, 1, 1, 28, 28);
  run_case("conv2 100x8x14x14->16 3x3", 100, 8, 14, 14, 16, 3, 1, 1, 14, 14);
  return 0;
}
