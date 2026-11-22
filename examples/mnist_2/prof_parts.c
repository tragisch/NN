/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Micro-benchmark: im2col vs batched-GEMM vs single-big-GEMM vs scatter. */
#include "sm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void bench(const char *label, size_t n, size_t c_in, size_t h, size_t w,
                  size_t c_out, size_t k, size_t out_h, size_t out_w) {
  const size_t patch = c_in * k * k;
  const size_t osp = out_h * out_w;
  float *in = malloc(n * c_in * h * w * sizeof(float));
  float *wt = malloc(c_out * patch * sizeof(float));
  float *col = malloc(n * osp * patch * sizeof(float));
  float *y = malloc(n * osp * c_out * sizeof(float));
  float *out = malloc(n * c_out * osp * sizeof(float));
  for (size_t i = 0; i < n * c_in * h * w; i++) in[i] = 1.0f;
  for (size_t i = 0; i < c_out * patch; i++) wt[i] = 0.001f;

  FloatMatrix w_mat = {.rows = patch, .cols = c_out, .capacity = patch * c_out, .values = wt};

  /* (a) im2col over whole batch, parallel over all rows */
  double t0 = now_ms();
#pragma omp parallel for
  for (size_t row = 0; row < n * osp; row++) {
    const size_t ni = row / osp, orow = row % osp;
    const size_t oh = orow / out_w, ow = orow % out_w;
    for (size_t ci = 0; ci < c_in; ci++)
      for (size_t kh = 0; kh < k; kh++)
        for (size_t kw = 0; kw < k; kw++) {
          const ptrdiff_t ih = (ptrdiff_t)(oh + kh) - 1, iw = (ptrdiff_t)(ow + kw) - 1;
          float v = 0.0f;
          if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w)
            v = in[((ni * c_in + ci) * h + ih) * w + iw];
          col[row * patch + (ci * k + kh) * k + kw] = v;
        }
  }
  double t_im2col = now_ms() - t0;

  /* (b) one big GEMM: [n*osp, patch] @ [patch, c_out] */
  FloatMatrix col_all = {.rows = n * osp, .cols = patch, .capacity = n * osp * patch, .values = col};
  FloatMatrix y_all = {.rows = n * osp, .cols = c_out, .capacity = n * osp * c_out, .values = y};
  t0 = now_ms();
  sm_gemm(&y_all, 1.0f, &col_all, SM_NO_TRANSPOSE, &w_mat, SM_NO_TRANSPOSE, 0.0f);
  double t_gemm_big = now_ms() - t0;

  /* (c) n separate GEMMs (current implementation) */
  t0 = now_ms();
  for (size_t ni = 0; ni < n; ni++) {
    FloatMatrix c1 = {.rows = osp, .cols = patch, .capacity = osp * patch, .values = col + ni * osp * patch};
    FloatMatrix y1 = {.rows = osp, .cols = c_out, .capacity = osp * c_out, .values = y + ni * osp * c_out};
    sm_gemm(&y1, 1.0f, &c1, SM_NO_TRANSPOSE, &w_mat, SM_NO_TRANSPOSE, 0.0f);
  }
  double t_gemm_loop = now_ms() - t0;

  /* (d) scatter y -> NCHW output */
  t0 = now_ms();
#pragma omp parallel for
  for (size_t row = 0; row < n * osp; row++) {
    const size_t ni = row / osp, orow = row % osp;
    for (size_t co = 0; co < c_out; co++)
      out[((ni * c_out + co) * osp) + orow] = y[row * c_out + co];
  }
  double t_scatter = now_ms() - t0;

  printf("%-24s im2col=%6.3f  gemm_big=%6.3f  gemm_loop=%6.3f  scatter=%6.3f ms\n",
         label, t_im2col, t_gemm_big, t_gemm_loop, t_scatter);

  free(in); free(wt); free(col); free(y); free(out);
}

int main(void) {
  bench("conv1 100x1x28x28->8", 100, 1, 28, 28, 8, 3, 28, 28);
  bench("conv2 100x8x14x14->16", 100, 8, 14, 14, 16, 3, 14, 14);
  return 0;
}
