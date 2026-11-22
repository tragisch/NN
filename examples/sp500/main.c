/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

// S&P 500 (^GSPC) next-day log-return prediction, after the example of
// https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data
//
// The reference trains an LSTM on sequences of 60 days of OHLCV plus
// technical indicators (MACD histogram, Stochastics %K, ATR) with a
// RobustScaler, to predict the next day's log return of the adjusted close.
// nm_lib has no recurrent layers yet, so this example feeds the same 60-day
// window, flattened, into a dense MLP - the classic "MLP on a rolling window"
// stand-in for the LSTM.
//
// Data: gspc.csv (^GSPC daily bars 1990-2018, from the reference repo).
// The last 5% of the sequences are held out for testing, as in the reference.

#include "nm_api.h"
#include <log.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define SEQ_LEN 60     // days per input window (as in the reference)
#define N_FEATURES 5   // lrets, MACD hist, Stochastics %K, ATR, Volume
#define TRAIN_FRAC 0.95f

typedef struct {
  float open, high, low, close, adj_close, volume;
} Bar;

// ---- CSV loading ----------------------------------------------------------

static FILE *open_data_file(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file)
    return file;

  // Bazel runfiles locations, as in the other examples
  char path[PATH_MAX];
  const char *runfiles_dir = getenv("RUNFILES_DIR");
  if (runfiles_dir &&
      snprintf(path, sizeof(path), "%s/_main/examples/sp500/%s", runfiles_dir,
               filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, "r");
    if (file)
      return file;
  }
  const char *test_srcdir = getenv("TEST_SRCDIR");
  const char *test_workspace = getenv("TEST_WORKSPACE");
  if (test_srcdir && test_workspace &&
      snprintf(path, sizeof(path), "%s/%s/examples/sp500/%s", test_srcdir,
               test_workspace, filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, "r");
    if (file)
      return file;
  }
  // `bazel run` (manifest-only runfiles) exports the workspace directory
  const char *workspace = getenv("BUILD_WORKSPACE_DIRECTORY");
  if (workspace &&
      snprintf(path, sizeof(path), "%s/examples/sp500/%s", workspace,
               filename) > 0 &&
      strlen(path) < sizeof(path)) {
    file = fopen(path, "r");
    if (file)
      return file;
  }
  return NULL;
}

static size_t load_bars(const char *filename, Bar **out) {
  FILE *file = open_data_file(filename);
  if (!file) {
    log_error("Failed to open data file: %s", filename);
    return 0;
  }
  size_t cap = 8192, n = 0;
  Bar *bars = malloc(cap * sizeof(Bar));
  char line[512];
  if (!fgets(line, sizeof(line), file)) { // header
    log_error("Empty data file");
    free(bars);
    fclose(file);
    return 0;
  }
  while (fgets(line, sizeof(line), file)) {
    char date[32];
    Bar b;
    double vol;
    if (sscanf(line, "%31[^,],%f,%f,%f,%f,%f,%lf", date, &b.open, &b.high,
               &b.low, &b.close, &b.adj_close, &vol) != 7)
      continue;
    b.volume = (float)vol;
    if (n == cap) {
      cap *= 2;
      bars = realloc(bars, cap * sizeof(Bar));
    }
    bars[n++] = b;
  }
  fclose(file);
  *out = bars;
  return n;
}

// ---- Technical indicators (as in notebook 2 of the reference) -------------

// features[t][0..4]: log return, MACD histogram, Stochastics %K, ATR(14), volume
static void compute_features(const Bar *bars, size_t n, float *feat) {
  const float a12 = 2.0f / 13.0f, a26 = 2.0f / 27.0f, a9 = 2.0f / 10.0f;
  float ema12 = bars[0].adj_close, ema26 = bars[0].adj_close;
  float macd = 0.0f, signal = 0.0f;
  float atr = 0.0f;
  float tr_sum = 0.0f;

  for (size_t t = 0; t < n; t++) {
    float *f = feat + t * N_FEATURES;
    float close = bars[t].adj_close;

    // daily log return of the adjusted close
    f[0] = (t == 0) ? 0.0f : logf(close / bars[t - 1].adj_close);

    // MACD histogram: EMA12 - EMA26, signal = EMA9 of MACD, hist = MACD - signal
    ema12 = a12 * close + (1.0f - a12) * ema12;
    ema26 = a26 * close + (1.0f - a26) * ema26;
    macd = ema12 - ema26;
    signal = (t == 0) ? macd : a9 * macd + (1.0f - a9) * signal;
    f[1] = macd - signal;

    // Stochastics %K(14) on the adjusted close
    if (t >= 13) {
      float lo = bars[t].adj_close, hi = bars[t].adj_close;
      for (size_t k = t - 13; k <= t; k++) {
        if (bars[k].adj_close < lo) lo = bars[k].adj_close;
        if (bars[k].adj_close > hi) hi = bars[k].adj_close;
      }
      f[2] = (hi > lo) ? 100.0f * (close - lo) / (hi - lo) : 50.0f;
    } else {
      f[2] = 50.0f;
    }

    // ATR(14): mean true range over the last 14 days
    float tr = bars[t].high - bars[t].low;
    if (t > 0) {
      float hc = fabsf(bars[t].high - bars[t - 1].close);
      float lc = fabsf(bars[t].low - bars[t - 1].close);
      if (hc > tr) tr = hc;
      if (lc > tr) tr = lc;
    }
    tr_sum += tr;
    if (t >= 14) {
      float tr_old = bars[t - 14].high - bars[t - 14].low;
      if (t - 14 > 0) {
        float hc = fabsf(bars[t - 14].high - bars[t - 15].close);
        float lc = fabsf(bars[t - 14].low - bars[t - 15].close);
        if (hc > tr_old) tr_old = hc;
        if (lc > tr_old) tr_old = lc;
      }
      tr_sum -= tr_old;
    }
    atr = (t >= 13) ? tr_sum / 14.0f : tr_sum / (float)(t + 1);
    f[3] = atr;

    f[4] = bars[t].volume;
  }
}

// ---- RobustScaler (median / IQR), fitted on the training days --------------

static int cmp_float(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x > y) - (x < y);
}

static void fit_robust_scaler(const float *feat, size_t n_days, size_t fit_upto,
                              float *median, float *iqr) {
  float *col = malloc(fit_upto * sizeof(float));
  for (size_t j = 0; j < N_FEATURES; j++) {
    for (size_t t = 0; t < fit_upto; t++)
      col[t] = feat[t * N_FEATURES + j];
    qsort(col, fit_upto, sizeof(float), cmp_float);
    median[j] = col[fit_upto / 2];
    float q1 = col[fit_upto / 4], q3 = col[(3 * fit_upto) / 4];
    iqr[j] = (q3 - q1 > 1e-12f) ? (q3 - q1) : 1.0f;
  }
  free(col);
}

static void apply_scaler(float *feat, size_t n_days, const float *median,
                         const float *iqr) {
  for (size_t t = 0; t < n_days; t++)
    for (size_t j = 0; j < N_FEATURES; j++)
      feat[t * N_FEATURES + j] =
          (feat[t * N_FEATURES + j] - median[j]) / iqr[j];
}

// ---- Sequence building ------------------------------------------------------

// Window of SEQ_LEN days ending at day t predicts the log return of day t+1
// (in the reference: data['Y'] = data.lrets.shift(-1)).
static void build_sequences(const float *feat, const Bar *bars, size_t n_days,
                            float *X, float *y, size_t *out_n) {
  size_t n_seq = n_days - SEQ_LEN - 1;
  for (size_t s = 0; s < n_seq; s++) {
    size_t t = s + SEQ_LEN - 1; // last day of the window
    for (size_t d = 0; d < SEQ_LEN; d++)
      for (size_t j = 0; j < N_FEATURES; j++)
        X[(s * SEQ_LEN + d) * N_FEATURES + j] =
            feat[(s + d) * N_FEATURES + j];
    y[s] = (t + 1 < n_days) ? logf(bars[t + 1].adj_close / bars[t].adj_close)
                            : 0.0f;
  }
  *out_n = n_seq;
}

static int on_epoch(const NmProgress *p, void *user) {
  (void)user;
  if (p->epoch % 10 == 0 || p->epoch == p->epochs)
    log_info("epoch %zu/%zu  loss=%.6f  val_loss=%.6f", p->epoch, p->epochs,
             p->loss, p->val_loss);
  return 0;
}

int main(void) {
  log_set_level(LOG_INFO);

  Bar *bars = NULL;
  size_t n_days = load_bars("gspc.csv", &bars);
  if (n_days < SEQ_LEN + 100) {
    log_error("Not enough data rows: %zu", n_days);
    free(bars);
    return EXIT_FAILURE;
  }
  log_info("Loaded %zu daily bars from gspc.csv", n_days);

  // per-day feature matrix
  float *feat = malloc(n_days * N_FEATURES * sizeof(float));
  compute_features(bars, n_days, feat);

  // robust scaling, fitted on the training days only (as the reference fits
  // its scaler on the training split)
  size_t fit_upto = (size_t)(n_days * TRAIN_FRAC);
  float median[N_FEATURES], iqr[N_FEATURES];
  fit_robust_scaler(feat, n_days, fit_upto, median, iqr);
  apply_scaler(feat, n_days, median, iqr);

  // sequences: window -> next-day log return
  size_t n_seq = n_days - SEQ_LEN - 1;
  size_t in_dim = SEQ_LEN * N_FEATURES;
  float *X = malloc(n_seq * in_dim * sizeof(float));
  float *y = malloc(n_seq * sizeof(float));
  build_sequences(feat, bars, n_days, X, y, &n_seq);
  log_info("Built %zu sequences of %d features (window %d x %d)", n_seq,
           (int)in_dim, (int)SEQ_LEN, (int)N_FEATURES);

  // hold out the last 5% of sequences for testing (as the reference)
  size_t n_test = n_seq / 20;
  size_t n_train = n_seq - n_test;

  // MLP: flattened 60-day window -> 128 -> 64 -> 32 -> 1 (linear regression)
  NmSpec *spec = nm_spec_new((NmShape){.ndim = 1, .dims = {in_dim}});
  NmStatus s = NM_STATUS_OK;
  if (s == NM_STATUS_OK) s = nm_spec_dense(spec, 128, NM_ACT_RELU);
  if (s == NM_STATUS_OK) s = nm_spec_dense(spec, 64, NM_ACT_RELU);
  if (s == NM_STATUS_OK) s = nm_spec_dense(spec, 32, NM_ACT_RELU);
  if (s == NM_STATUS_OK) s = nm_spec_dense(spec, 1, NM_ACT_NONE);
  if (s != NM_STATUS_OK) {
    log_error("Failed to describe model: %s", nm_last_error());
    nm_spec_free(spec);
    return EXIT_FAILURE;
  }

  NmBuildOptions build = nm_build_options_default();
  build.seed = 42;
  build.deterministic = true;
  NmNetwork *net = nm_network_build(spec, &build);
  nm_spec_free(spec);
  if (!net) {
    log_error("Failed to build model: %s", nm_last_error());
    return EXIT_FAILURE;
  }
  log_info("Model parameters: %zu", nm_network_parameter_count(net));

  NmTensor x_train = nm_tensor_2d(X, n_train, in_dim);
  NmTensor y_train = nm_tensor_2d(y, n_train, 1);
  NmFitOptions fit = nm_fit_options_default();
  fit.loss = NM_LOSS_MSE;
  fit.optimizer = NM_OPT_ADAM;
  fit.learning_rate = 0.001f;
  fit.batch_size = 64;
  fit.epochs = 60;
  fit.shuffle = true;
  fit.validation_split = 0.05f;

  log_info("Training on %zu sequences (test holdout: %zu)", n_train, n_test);
  s = nm_network_fit(net, x_train, &y_train, &fit, on_epoch, NULL);
  if (s != NM_STATUS_OK) {
    log_error("Training failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }

  // evaluate on the held-out last 5%
  NmTensor x_test = nm_tensor_2d(X + n_train * in_dim, n_test, in_dim);
  NmTensor y_test = nm_tensor_2d(y + n_train, n_test, 1);
  NmMetrics m;
  s = nm_network_evaluate(net, x_test, y_test, NM_LOSS_MSE, &m);
  if (s != NM_STATUS_OK) {
    log_error("Evaluation failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }
  float rmse = sqrtf(m.loss);
  log_info("Test MSE: %.8f   Test RMSE: %.6f (%.3f%% per day)", m.loss, rmse,
           rmse * 100.0f);

  // baseline: always predict 0 (the mean log return)
  float base_mse = 0.0f;
  for (size_t i = 0; i < n_test; i++)
    base_mse += y[n_train + i] * y[n_train + i];
  base_mse /= n_test;
  log_info("Baseline (predict 0) Test RMSE: %.6f (%.3f%% per day)", sqrtf(base_mse),
           sqrtf(base_mse) * 100.0f);

  // MAE, correlation and directional accuracy on the test set
  NmTensor pred;
  s = nm_network_predict(net, x_test, &pred);
  if (s != NM_STATUS_OK) {
    log_error("Prediction failed: %s", nm_last_error());
    nm_network_free(net);
    return EXIT_FAILURE;
  }
  float mae = 0.0f, sp = 0.0f, st = 0.0f, spp = 0.0f, stt = 0.0f, spt = 0.0f;
  size_t dir_ok = 0;
  for (size_t i = 0; i < n_test; i++) {
    float p = pred.data[i], t = y[n_train + i];
    mae += fabsf(p - t);
    spp += p * p;
    stt += t * t;
    spt += p * t;
    sp += p;
    st += t;
    if ((p > 0) == (t > 0)) dir_ok++;
  }
  mae /= n_test;
  float corr = (spt - sp * st / n_test) /
               sqrtf((spp - sp * sp / n_test) * (stt - st * st / n_test) + 1e-20f);
  log_info("Test MAE: %.6f (%.3f%% per day)", mae, mae * 100.0f);
  log_info("Correlation(pred, true): %.4f", corr);
  log_info("Directional accuracy: %zu/%zu (%.1f%%)", dir_ok, n_test,
           100.0f * dir_ok / n_test);

  log_info("Sample predictions (first 5 of test set):");
  for (size_t i = 0; i < 5 && i < n_test; i++)
    log_info("  pred %+.5f  true %+.5f", pred.data[i], y[n_train + i]);

  nm_tensor_free(&pred);
  nm_network_free(net);
  free(X);
  free(y);
  free(feat);
  free(bars);
  return EXIT_SUCCESS;
}
