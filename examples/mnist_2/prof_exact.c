/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Replicate st_conv2d_gemm_nchw exactly (per-image loop, same omp guards,
 * serial scatter) and time phases; checksum prevents dead-code elimination. */
#include "sm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float g_sink;

static void bench(const char *label, size_t n, size_t c_in, size_t h, size_t w,
                  size_t c_out, size_t k, size_t out_h, size_t out_w) {
  const size_t patch = c_in * k * k;
  const size_t osp = out_h * out_w;
  float *in = malloc(n * c_in * h * w * sizeof(float));
  float *wt = malloc(c_out * patch * sizeof(float));
  float *col = malloc(osp * patch * sizeof(float));
  float *y = malloc(osp * c_out * sizeof(float));
  float *out = malloc(n * c_out * osp * sizeof(float));
  for (size_t i = 0; i < n * c_in * h * w; i++) in[i] = 1.0f;
  for (size_t i = 0; i < c_out * patch; i++) wt[i] = 0.001f;

  FloatMatrix w_mat = {.rows = patch, .cols = c_out, .capacity = patch * c_out, .values = wt};

  double t_im2col = 0, t_gemm = 0, t_scatter = 0;
  double t0 = now_ms();
  for (size_t ni = 0; ni < n; ni++) {
    double a = now_ms();
#pragma omp parallel for if (osp >= 256)
    for (size_t out_row = 0; out_row < osp; out_row++) {
      const size_t oh = out_row / out_w, ow = out_row % out_w;
      for (size_t ci = 0; ci < c_in; ci++)
        for (size_t kh = 0; kh < k; kh++)
          for (size_t kw = 0; kw < k; kw++) {
            const ptrdiff_t ih = (ptrdiff_t)(oh + kh) - 1, iw = (ptrdiff_t)(ow + kw) - 1;
            float v = 0.0f;
            if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w)
              v = in[((ni * c_in + ci) * h + ih) * w + iw];
            col[out_row * patch + (ci * k + kh) * k + kw] = v;
          }
    }
    double b = now_ms();
    FloatMatrix c1 = {.rows = osp, .cols = patch, .capacity = osp * patch, .values = col};
    FloatMatrix y1 = {.rows = osp, .cols = c_out, .capacity = osp * c_out, .values = y};
    sm_gemm(&y1, 1.0f, &c1, SM_NO_TRANSPOSE, &w_mat, SM_NO_TRANSPOSE, 0.0f);
    double c = now_ms();
    for (size_t oh = 0; oh < out_h; oh++)
      for (size_t ow = 0; ow < out_w; ow++) {
        const size_t orow = oh * out_w + ow;
        for (size_t co = 0; co < c_out; co++)
          out[((ni * c_out + co) * out_h + oh) * out_w + ow] = y[orow * c_out + co];
      }
    double d = now_ms();
    t_im2col += b - a; t_gemm += c - b; t_scatter += d - c;
  }
  double total = now_ms() - t0;
  for (size_t i = 0; i < n * c_out * osp; i++) g_sink += out[i];
  printf("%-24s im2col=%6.3f  gemm=%6.3f  scatter=%6.3f  total=%6.3f ms\n",
         label, t_im2col, t_gemm, t_scatter, total);
  free(in); free(wt); free(col); free(y); free(out);
}

int main(void) {
  bench("conv1 100x1x28x28->8", 100, 1, 28, 28, 8, 3, 28, 28);
  bench("conv2 100x8x14x14->16", 100, 8, 14, 14, 16, 3, 14, 14);
  printf("sink=%f\n", g_sink);
  return 0;
}
