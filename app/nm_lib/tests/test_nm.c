/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "nm.h"
#include "sm.h"
#include <stdio.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

#define EPS 1e-5f

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void fill_matrix(FloatMatrix *m, float scale) {
  for (size_t i = 0; i < m->rows * m->cols; ++i) {
    m->values[i] = (float)(i % 17) * scale;
  }
}

static void assert_matrix_close(const FloatMatrix *a, const FloatMatrix *b,
                                float eps) {
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);
  TEST_ASSERT_EQUAL_size_t(a->rows, b->rows);
  TEST_ASSERT_EQUAL_size_t(a->cols, b->cols);
  for (size_t i = 0; i < a->rows * a->cols; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(eps, a->values[i], b->values[i]);
  }
}

static uint64_t read_u64_native(const unsigned char *src) {
  uint64_t value = 0;
  memcpy(&value, src, sizeof(value));
  return value;
}

static void write_u64_native(unsigned char *dst, uint64_t value) {
  memcpy(dst, &value, sizeof(value));
}

static unsigned char *read_file_bytes(const char *path, size_t *out_size) {
  TEST_ASSERT_NOT_NULL(path);
  TEST_ASSERT_NOT_NULL(out_size);

  FILE *f = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL(f);
  TEST_ASSERT_EQUAL_INT(0, fseek(f, 0, SEEK_END));
  long size = ftell(f);
  TEST_ASSERT_TRUE(size >= 0);
  TEST_ASSERT_EQUAL_INT(0, fseek(f, 0, SEEK_SET));

  unsigned char *data = (unsigned char *)malloc((size_t)size);
  TEST_ASSERT_NOT_NULL(data);
  TEST_ASSERT_EQUAL_size_t((size_t)size, fread(data, 1, (size_t)size, f));
  TEST_ASSERT_EQUAL_INT(0, fclose(f));

  *out_size = (size_t)size;
  return data;
}

static void write_file_bytes(const char *path, const unsigned char *data,
                             size_t size) {
  TEST_ASSERT_NOT_NULL(path);
  TEST_ASSERT_NOT_NULL(data);

  FILE *f = fopen(path, "wb");
  TEST_ASSERT_NOT_NULL(f);
  TEST_ASSERT_EQUAL_size_t(size, fwrite(data, 1, size, f));
  TEST_ASSERT_EQUAL_INT(0, fclose(f));
}

static size_t find_chunk_offset(const unsigned char *data, size_t size,
                                const char id[4], uint64_t *out_chunk_size) {
  TEST_ASSERT_NOT_NULL(data);
  TEST_ASSERT_NOT_NULL(id);

  size_t offset = 8;
  while (offset + 12 <= size) {
    uint64_t chunk_size = read_u64_native(data + offset + 4);
    TEST_ASSERT_TRUE(chunk_size <= (uint64_t)(size - offset - 12));
    if (memcmp(data + offset, id, 4) == 0) {
      if (out_chunk_size)
        *out_chunk_size = chunk_size;
      return offset;
    }
    offset += 12 + (size_t)chunk_size;
  }

  TEST_FAIL_MESSAGE("chunk not found");
  return 0;
}

static NmModel *create_roundtrip_dense_model(void) {
  NmShape in = {.ndim = 1, .dims = {4}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m, &(NmDenseConfig){.units = 5, .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m,
                   &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));
  return m;
}

void setUp(void) {}
void tearDown(void) {}

void test_nm_save_load_should_roundtrip_dense_model(void) {
  const char *path = "tmp_nm_dense.nms2";

  NmShape in = {.ndim = 1, .dims = {4}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  NmDenseConfig d1 = {.units = 6, .activation = NM_ACT_RELU};
  NmDenseConfig d2 = {.units = 3, .activation = NM_ACT_SOFTMAX};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d2));

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, &cc));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_optimizer(m, NULL));
  NmOptimizerConfig opt = nm_optimizer_defaults();
  opt.type = NM_OPT_ADAM;
  opt.learning_rate = 0.002f;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_optimizer(m, &opt));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_autograd_mode(m, NM_AUTOGRAD_TAPE));
  NmBatchParallelConfig pcfg = {
      .enabled = true,
      .deterministic = false,
      .num_workers = 3,
  };
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_batch_parallel(m, &pcfg));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save(m, path));

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);

  TEST_ASSERT_EQUAL(NM_AUTOGRAD_TAPE, nm_autograd_mode(loaded));
  NmBatchParallelConfig got = nm_batch_parallel(loaded);
  TEST_ASSERT_TRUE(got.enabled);
  TEST_ASSERT_FALSE(got.deterministic);
  TEST_ASSERT_EQUAL_size_t(3, got.num_workers);

  FloatMatrix *x = sm_create(5, 4);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.1f);

  FloatMatrix *pred_a = NULL;
  FloatMatrix *pred_b = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_a));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(loaded, x, &pred_b));
  assert_matrix_close(pred_a, pred_b, EPS);

  sm_destroy(pred_a);
  sm_destroy(pred_b);
  sm_destroy(x);
  nm_free(m);
  nm_free(loaded);
  remove(path);
}

void test_nm_backend_capabilities_should_expose_runtime_features(void) {
  const NmBackendCapabilities *caps = nm_backend_capabilities();
  TEST_ASSERT_NOT_NULL(caps);
  TEST_ASSERT_NOT_NULL(caps->name);
  TEST_ASSERT_TRUE(caps->supports_training);
  TEST_ASSERT_TRUE(caps->supports_prepared_inference);
  TEST_ASSERT_TRUE(caps->supports_tensor_pool_reuse);

  NmShape in = {.ndim = 1, .dims = {4}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m,
                   &(NmDenseConfig){.units = 2, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));

  const NmBackendCapabilities *model_caps = nm_model_backend_capabilities(m);
  TEST_ASSERT_NOT_NULL(model_caps);
  TEST_ASSERT_NOT_NULL(model_caps->name);
  TEST_ASSERT_EQUAL_STRING(caps->name, model_caps->name);

  nm_free(m);
}

void test_nm_save_inference_should_roundtrip_and_mark_model_inference_only(
    void) {
  const char *path = "tmp_nm_infer_only.nms3";

  NmShape in = {.ndim = 1, .dims = {4}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m, &(NmDenseConfig){.units = 5, .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m,
                   &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));

  NmOptimizerConfig opt = nm_optimizer_defaults();
  opt.learning_rate = 0.002f;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_optimizer(m, &opt));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_autograd_mode(m, NM_AUTOGRAD_TAPE));
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_set_batch_parallel(m, &(NmBatchParallelConfig){
                                                 .enabled = true,
                                                 .deterministic = false,
                                                 .num_workers = 2,
                                             }));

  FloatMatrix *x = sm_create(4, 4);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.07f);

  FloatMatrix *pred_before = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_before));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference(m, path));

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);
  TEST_ASSERT_TRUE(nm_model_is_inference_only(loaded));
  TEST_ASSERT_FALSE(nm_model_was_bf16(loaded));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_set_autograd_mode(loaded, NM_AUTOGRAD_TAPE));

  FloatMatrix *pred_after = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(loaded, x, &pred_after));
  assert_matrix_close(pred_before, pred_after, EPS);

  sm_destroy(pred_before);
  sm_destroy(pred_after);
  sm_destroy(x);
  nm_free(m);
  nm_free(loaded);
  remove(path);
}

void test_nm_save_load_should_roundtrip_conv_model(void) {
  const char *path = "tmp_nm_conv.nms2";

  NmShape in = {.ndim = 3, .dims = {1, 8, 8}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_tensor_layout(m, NM_LAYOUT_NHWC));

  NmConv2dConfig conv = {
      .filters = 2,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .algo = NM_CONV_ALGO_GEMM,
      .layout = NM_LAYOUT_NHWC,
      .activation = NM_ACT_RELU,
  };
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_conv2d(m, &conv));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_flatten(m));

  NmDenseConfig out = {.units = 4, .activation = NM_ACT_SOFTMAX};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &out));

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, &cc));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save(m, path));

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);

  FloatMatrix *x = sm_create(3, 64);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.03f);

  FloatMatrix *pred_a = NULL;
  FloatMatrix *pred_b = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_a));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(loaded, x, &pred_b));
  assert_matrix_close(pred_a, pred_b, 1e-4f);

  sm_destroy(pred_a);
  sm_destroy(pred_b);
  sm_destroy(x);
  nm_free(m);
  nm_free(loaded);
  remove(path);
}

void test_nm_predict_into_should_reject_wrong_output_shape(void) {
  NmModel *m = create_roundtrip_dense_model();

  FloatMatrix *x = sm_create(2, 4);
  FloatMatrix *bad = sm_create(2, 2);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(bad);
  fill_matrix(x, 0.04f);

  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_predict_into(m, x, bad));

  sm_destroy(bad);
  sm_destroy(x);
  nm_free(m);
}

void test_nm_conv_pool_flatten_dense_predict_into_should_match_allocated(void) {
  NmShape in = {.ndim = 3, .dims = {1, 8, 8}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  NmConv2dConfig conv = {
      .filters = 2,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .algo = NM_CONV_ALGO_GEMM,
      .layout = NM_LAYOUT_NCHW,
      .activation = NM_ACT_RELU,
  };
  NmPool2dConfig pool = {
      .kernel_size = 2,
      .stride = 2,
      .padding = 0,
  };
  NmDenseConfig out = {.units = 3, .activation = NM_ACT_SOFTMAX};

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_conv2d(m, &conv));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_maxpool2d(m, &pool));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_flatten(m));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &out));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));

  FloatMatrix *x = sm_create(4, 64);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.02f);

  FloatMatrix *pred_alloc = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_alloc));
  TEST_ASSERT_NOT_NULL(pred_alloc);
  TEST_ASSERT_EQUAL_size_t(4, pred_alloc->rows);
  TEST_ASSERT_EQUAL_size_t(3, pred_alloc->cols);

  FloatMatrix *pred_into = sm_create(pred_alloc->rows, pred_alloc->cols);
  TEST_ASSERT_NOT_NULL(pred_into);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict_into(m, x, pred_into));
  assert_matrix_close(pred_alloc, pred_into, 1e-5f);

  sm_destroy(pred_into);
  sm_destroy(pred_alloc);
  sm_destroy(x);
  nm_free(m);
}

void test_nm_unsupported_layers_should_stay_explicitly_unsupported(void) {
  NmShape in = {.ndim = 3, .dims = {1, 4, 4}};
  NmPool2dConfig pool = {
      .kernel_size = 2,
      .stride = 2,
      .padding = 0,
  };
  NmBatchNorm2dConfig bn = {.epsilon = 1e-5f};

  NmModel *avg = nm_create(in);
  TEST_ASSERT_NOT_NULL(avg);
  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE, nm_add_avgpool2d(avg, &pool));
  nm_free(avg);

  NmModel *batchnorm = nm_create(in);
  TEST_ASSERT_NOT_NULL(batchnorm);
  TEST_ASSERT_EQUAL(NM_STATUS_UNSUPPORTED_TYPE,
                    nm_add_batchnorm2d(batchnorm, &bn));
  nm_free(batchnorm);
}

void test_nm_autograd_and_batch_parallel_config_should_apply(void) {
  NmShape in = {.ndim = 1, .dims = {3}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  NmDenseConfig d1 = {.units = 4, .activation = NM_ACT_RELU};
  NmDenseConfig d2 = {.units = 2, .activation = NM_ACT_SOFTMAX};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d2));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_autograd_mode(m, NM_AUTOGRAD_TAPE));
  TEST_ASSERT_EQUAL(NM_AUTOGRAD_TAPE, nm_autograd_mode(m));

  NmBatchParallelConfig pcfg = {
      .enabled = true,
      .deterministic = true,
      .num_workers = 2,
  };
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_batch_parallel(m, &pcfg));
  NmBatchParallelConfig got = nm_batch_parallel(m);
  TEST_ASSERT_TRUE(got.enabled);
  TEST_ASSERT_TRUE(got.deterministic);
  TEST_ASSERT_EQUAL_size_t(2, got.num_workers);

  FloatMatrix *x = sm_create(4, 3);
  FloatMatrix *y = sm_create(4, 2);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(y);
  fill_matrix(x, 0.05f);
  fill_matrix(y, 0.1f);

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_fit(m, x, y, NULL));
  TEST_ASSERT_TRUE(nm_last_graph_nodes(m) > 0);

  sm_destroy(x);
  sm_destroy(y);
  nm_free(m);
}

void test_nm_bf16_inference_export_should_roundtrip_and_stay_close(void) {
  const char *path = "tmp_nm_bf16_infer.nms4";

  NmShape in = {.ndim = 1, .dims = {4}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m, &(NmDenseConfig){.units = 6, .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m,
                   &(NmDenseConfig){.units = 3, .activation = NM_ACT_SOFTMAX}));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, NULL));

  FloatMatrix *x = sm_create(5, 4);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.1f);

  /* Reference prediction in f32. */
  FloatMatrix *pred_f32 = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_f32));

  /* Export with bf16 weights. */
  NmSaveInferenceOptions opts = {.use_bf16 = true};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference_ex(m, path, &opts));

  /* Load the bf16 snapshot — weights are auto-promoted to f32. */
  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);
  TEST_ASSERT_TRUE(nm_model_is_inference_only(loaded));
  TEST_ASSERT_TRUE(nm_model_was_bf16(loaded));

  /* Predictions should be close (bf16 precision ≈ 3 significant digits). */
  FloatMatrix *pred_bf16 = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(loaded, x, &pred_bf16));
  TEST_ASSERT_EQUAL_size_t(pred_f32->rows, pred_bf16->rows);
  TEST_ASSERT_EQUAL_size_t(pred_f32->cols, pred_bf16->cols);
  for (size_t i = 0; i < pred_f32->rows * pred_f32->cols; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-2f, pred_f32->values[i], pred_bf16->values[i]);
  }

  sm_destroy(pred_f32);
  sm_destroy(pred_bf16);
  sm_destroy(x);
  nm_free(m);
  nm_free(loaded);
  remove(path);
}

void test_nm_bf16_loaded_model_should_reject_training(void) {
  const char *path = "tmp_nm_bf16_no_train.nms4";

  NmShape in = {.ndim = 1, .dims = {3}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m, &(NmDenseConfig){.units = 4, .activation = NM_ACT_RELU}));
  TEST_ASSERT_EQUAL(
      NM_STATUS_OK,
      nm_add_dense(m,
                   &(NmDenseConfig){.units = 2, .activation = NM_ACT_SOFTMAX}));
  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, &cc));

  NmSaveInferenceOptions opts = {.use_bf16 = true};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference_ex(m, path, &opts));
  nm_free(m);

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);
  TEST_ASSERT_TRUE(nm_model_is_inference_only(loaded));

  /* Training must be rejected on an inference-only model. */
  FloatMatrix *x = sm_create(4, 3);
  FloatMatrix *y = sm_create(4, 2);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(y);
  fill_matrix(x, 0.05f);
  fill_matrix(y, 0.1f);

  NmStatus fit_status = nm_fit(loaded, x, y, NULL);
  TEST_ASSERT_NOT_EQUAL(NM_STATUS_OK, fit_status);

  sm_destroy(x);
  sm_destroy(y);
  nm_free(loaded);
  remove(path);
}

void test_nm_bf16_conv_model_should_roundtrip_close(void) {
  const char *path = "tmp_nm_bf16_conv.nms4";

  NmShape in = {.ndim = 3, .dims = {1, 8, 8}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_tensor_layout(m, NM_LAYOUT_NHWC));

  NmConv2dConfig conv = {
      .filters = 2,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .algo = NM_CONV_ALGO_GEMM,
      .layout = NM_LAYOUT_NHWC,
      .activation = NM_ACT_RELU,
  };
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_conv2d(m, &conv));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_flatten(m));

  NmDenseConfig out = {.units = 4, .activation = NM_ACT_SOFTMAX};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &out));

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, &cc));

  FloatMatrix *x = sm_create(3, 64);
  TEST_ASSERT_NOT_NULL(x);
  fill_matrix(x, 0.03f);

  /* Reference prediction in f32. */
  FloatMatrix *pred_f32 = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_f32));

  /* Export with bf16 weights. */
  NmSaveInferenceOptions opts = {.use_bf16 = true};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference_ex(m, path, &opts));

  /* Reload — weights are auto-promoted to f32. */
  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_load(path, &loaded));
  TEST_ASSERT_NOT_NULL(loaded);
  TEST_ASSERT_TRUE(nm_model_is_inference_only(loaded));
  TEST_ASSERT_TRUE(nm_model_was_bf16(loaded));

  FloatMatrix *pred_bf16 = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(loaded, x, &pred_bf16));
  TEST_ASSERT_EQUAL_size_t(pred_f32->rows, pred_bf16->rows);
  TEST_ASSERT_EQUAL_size_t(pred_f32->cols, pred_bf16->cols);
  for (size_t i = 0; i < pred_f32->rows * pred_f32->cols; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-2f, pred_f32->values[i], pred_bf16->values[i]);
  }

  sm_destroy(pred_f32);
  sm_destroy(pred_bf16);
  sm_destroy(x);
  nm_free(m);
  nm_free(loaded);
  remove(path);
}

void test_nm_facade_should_train_predict_and_predict_into(void) {
  NmShape in = {.ndim = 1, .dims = {3}};
  NmModel *m = nm_create(in);
  TEST_ASSERT_NOT_NULL(m);

  NmDenseConfig d1 = {.units = 5, .activation = NM_ACT_RELU};
  NmDenseConfig d2 = {.units = 2, .activation = NM_ACT_SOFTMAX};
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d1));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_add_dense(m, &d2));

  NmCompileConfig cc = nm_compile_defaults();
  cc.loss_type = NM_LOSS_CROSS_ENTROPY;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_compile(m, &cc));

  NmOptimizerConfig opt = nm_optimizer_defaults();
  opt.learning_rate = 0.002f;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_set_optimizer(m, &opt));

  FloatMatrix *x = sm_create(6, 3);
  FloatMatrix *y = sm_create(6, 2);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(y);
  fill_matrix(x, 0.02f);
  fill_matrix(y, 0.03f);

  NmTrainConfig tc = nm_train_defaults();
  tc.epochs = 3;
  tc.batch_size = 2;
  tc.learning_rate = 0.005f;
  tc.verbose = false;

  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_fit_epoch(m, x, y, &tc));
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_fit(m, x, y, &tc));

  FloatMatrix *pred_alloc = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict(m, x, &pred_alloc));
  TEST_ASSERT_NOT_NULL(pred_alloc);

  FloatMatrix *pred_into = sm_create(pred_alloc->rows, pred_alloc->cols);
  TEST_ASSERT_NOT_NULL(pred_into);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_predict_into(m, x, pred_into));
  assert_matrix_close(pred_alloc, pred_into, 1e-5f);

  float acc = 0.0f;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_evaluate(m, x, y, &acc));
  TEST_ASSERT_TRUE(acc >= 0.0f);
  TEST_ASSERT_TRUE(acc <= 1.0f);

  sm_destroy(pred_into);
  sm_destroy(pred_alloc);
  sm_destroy(x);
  sm_destroy(y);
  nm_free(m);
}

void test_nm_load_should_reject_wghz_with_mismatched_raw_size(void) {
  const char *path = "tmp_nm_corrupt_raw_size.nms4";
  NmModel *m = create_roundtrip_dense_model();
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference(m, path));

  size_t size = 0;
  unsigned char *data = read_file_bytes(path, &size);
  uint64_t chunk_size = 0;
  size_t chunk_offset = find_chunk_offset(data, size, "WGHZ", &chunk_size);
  size_t payload_offset = chunk_offset + 12;
  uint64_t raw_size = read_u64_native(data + payload_offset);
  write_u64_native(data + payload_offset, raw_size + 1);
  write_file_bytes(path, data, size);

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_load(path, &loaded));
  TEST_ASSERT_NULL(loaded);

  free(data);
  nm_free(m);
  remove(path);
  (void)chunk_size;
}

void test_nm_load_should_reject_wghz_with_oversized_compressed_length(void) {
  const char *path = "tmp_nm_corrupt_zsize.nms4";
  NmModel *m = create_roundtrip_dense_model();
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_save_inference(m, path));

  size_t size = 0;
  unsigned char *data = read_file_bytes(path, &size);
  uint64_t chunk_size = 0;
  size_t chunk_offset = find_chunk_offset(data, size, "WGHZ", &chunk_size);
  size_t payload_offset = chunk_offset + 12;
  write_u64_native(data + payload_offset + sizeof(uint64_t), chunk_size);
  write_file_bytes(path, data, size);

  NmModel *loaded = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_load(path, &loaded));
  TEST_ASSERT_NULL(loaded);

  free(data);
  nm_free(m);
  remove(path);
}
