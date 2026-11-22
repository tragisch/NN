/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * nm_model.c — Public facade for the unified nm model API.
 *
 * The implementation is split across internal modules:
 * - nm_model_runtime.c
 * - nm_model_layers.c
 * - nm_model_core.c
 * - nm_model_exec.c
 * - nm_model_io.c
 *
 * This file intentionally stays small and forwards the public API to the
 * internal `nm_seq_*` implementation.
 */

#include "nm_model_internal.h"

NmModel *nm_create(NmShape input_shape) { return nm_seq_create(input_shape); }

void nm_free(NmModel *model) { nm_seq_free(model); }

NmStatus nm_add_dense(NmModel *m, const NmDenseConfig *cfg) {
  return nm_seq_add_dense(m, cfg);
}

NmStatus nm_add_conv2d(NmModel *m, const NmConv2dConfig *cfg) {
  return nm_seq_add_conv2d(m, cfg);
}

NmStatus nm_add_maxpool2d(NmModel *m, const NmPool2dConfig *cfg) {
  return nm_seq_add_maxpool2d(m, cfg);
}

NmStatus nm_add_avgpool2d(NmModel *m, const NmPool2dConfig *cfg) {
  return nm_seq_add_avgpool2d(m, cfg);
}

NmStatus nm_add_batchnorm2d(NmModel *m, const NmBatchNorm2dConfig *cfg) {
  return nm_seq_add_batchnorm2d(m, cfg);
}

NmStatus nm_add_flatten(NmModel *m) { return nm_seq_add_flatten(m); }

NmStatus nm_compile(NmModel *m, const NmCompileConfig *cfg) {
  return nm_seq_compile(m, cfg);
}

NmStatus nm_set_optimizer(NmModel *m, const NmOptimizerConfig *cfg) {
  return nm_seq_set_optimizer(m, cfg);
}

NmOptimizerConfig nm_optimizer_defaults(void) {
  return nm_seq_optimizer_defaults();
}

NmStatus nm_set_backend(NmModel *m, NmBackendType backend) {
  return nm_seq_set_backend(m, backend);
}

const char *nm_model_backend(const NmModel *m) { return nm_seq_backend(m); }

const NmBackendCapabilities *nm_model_backend_capabilities(const NmModel *m) {
  return nm_seq_model_backend_capabilities(m);
}

NmStatus nm_set_tensor_layout(NmModel *m, NmTensorLayout layout) {
  return nm_seq_set_tensor_layout(m, layout);
}

NmTensorLayout nm_tensor_layout(const NmModel *m) {
  return nm_seq_tensor_layout(m);
}

NmStatus nm_set_autograd_mode(NmModel *m, NmAutogradMode mode) {
  return nm_seq_set_autograd_mode(m, mode);
}

NmAutogradMode nm_autograd_mode(const NmModel *m) {
  return nm_seq_autograd_mode(m);
}

NmStatus nm_set_batch_parallel(NmModel *m, const NmBatchParallelConfig *cfg) {
  return nm_seq_set_batch_parallel(m, cfg);
}

NmBatchParallelConfig nm_batch_parallel(const NmModel *m) {
  return nm_seq_batch_parallel(m);
}

size_t nm_last_graph_nodes(const NmModel *m) {
  return nm_seq_last_graph_nodes(m);
}

NmStatus nm_fit_epoch(NmModel *m, const FloatMatrix *X,
                      const FloatMatrix *Y, const NmTrainConfig *cfg) {
  NmTrainConfig one_epoch = cfg ? *cfg : nm_train_defaults();
  one_epoch.epochs = 1;
  return nm_seq_fit(m, X, Y, &one_epoch);
}

NmStatus nm_fit(NmModel *m, const FloatMatrix *X,
                const FloatMatrix *Y, const NmTrainConfig *cfg) {
  return nm_seq_fit(m, X, Y, cfg);
}

NmStatus nm_predict(const NmModel *m, const FloatMatrix *input,
                    FloatMatrix **out_prediction) {
  return nm_seq_predict(m, input, out_prediction);
}

NmStatus nm_predict_into(const NmModel *m, const FloatMatrix *input,
                         FloatMatrix *out_prediction) {
  return nm_seq_predict_into(m, input, out_prediction);
}

NmStatus nm_evaluate(const NmModel *m, const FloatMatrix *X,
                     const FloatMatrix *Y, float *out_accuracy) {
  return nm_seq_evaluate(m, X, Y, out_accuracy);
}

void nm_summary(const NmModel *m) { nm_seq_summary(m); }

NmStatus nm_save(const NmModel *m, const char *path) {
  return nm_seq_save(m, path);
}

NmStatus nm_save_inference(const NmModel *m, const char *path) {
  return nm_seq_save_inference(m, path);
}

NmStatus nm_save_inference_ex(const NmModel *m, const char *path,
                              const NmSaveInferenceOptions *opts) {
  return nm_seq_save_inference_ex(m, path, opts);
}

NmStatus nm_load(const char *path, NmModel **out_model) {
  return nm_seq_load(path, out_model);
}

bool nm_model_is_inference_only(const NmModel *m) {
  return nm_seq_is_inference_only(m);
}

bool nm_model_was_bf16(const NmModel *m) {
  const NmSeq *s = (const NmSeq *)m;
  return s ? s->storage.loaded_as_bf16 : false;
}
