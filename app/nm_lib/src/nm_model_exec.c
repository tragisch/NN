/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Training, inference and evaluation execution for nm models.
 */

#include "nm_model_internal.h"
#include <log.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#if defined(_OPENMP)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

static size_t trainable_total_params(const NmTrainable *t) {
  if (!t)
    return 0;
  size_t total = 0;
  for (size_t i = 0; i < t->num_layers; ++i)
    total += layer_param_count(&t->layers[i]);
  return total;
}

static size_t detect_hw_threads(void) {
#if defined(_OPENMP)
  int threads = omp_get_max_threads();
  return threads > 0 ? (size_t)threads : 1;
#else
  return 1;
#endif
}

static size_t choose_parallel_workers(const NmTrainable *t, const NmTrainConfig *cfg,
                                      size_t feature_cols) {
  if (!t || !cfg)
    return 1;
  if (!t->training->batch_parallel.enabled)
    return 1;
  if (t->training->batch_parallel.num_workers > 0)
    return t->training->batch_parallel.num_workers;

  const size_t hw = detect_hw_threads();
  const size_t params = trainable_total_params(t);
  const size_t batch = cfg->batch_size > 0 ? cfg->batch_size : 1;
  const uint64_t work =
      (uint64_t)batch * (uint64_t)feature_cols + (uint64_t)params;

  size_t workers = 1;
  if (work >= 2000000ULL)
    workers = 4;
  else if (work >= 500000ULL)
    workers = 2;

  if (workers > hw)
    workers = hw;
  if (workers == 0)
    workers = 1;
  return workers;
}

static NmStatus model_train_one_epoch(NmTrainable *t, const FloatMatrix *X,
                                      const FloatMatrix *Y, size_t batch_size,
                                      float lr, float *out_epoch_loss,
                                      bool track_loss) {
  if (trainable_is_dense_only(t)) {
    return model_train_one_epoch_dense_fast(t, X, Y, batch_size, lr,
                                            out_epoch_loss, track_loss);
  }

  size_t n = X->rows;
  size_t layers = t->num_layers;
  size_t num_batches = (n + batch_size - 1) / batch_size;
  bool shortcut = trainable_uses_softmax_ce_shortcut(t);

  float total_loss = 0.0f;
  size_t total_samples = 0;

  NmStatus status = NM_STATUS_OK;
  NmTensorPool pool;
  if (!tensor_pool_init(&pool))
    return NM_STATUS_ALLOC_FAILED;

  FloatTensor **acts =
      (FloatTensor **)calloc(layers + 1, sizeof(FloatTensor *));
  FloatTensor **pidx = (FloatTensor **)calloc(layers, sizeof(FloatTensor *));
  if (!acts || !pidx) {
    free(acts);
    free(pidx);
    tensor_pool_destroy(&pool);
    return NM_STATUS_ALLOC_FAILED;
  }

  if (t->training->autograd_mode == NM_AUTOGRAD_TAPE)
    exec_trace_reset(&t->training->last_trace);

  for (size_t b = 0; b < num_batches; ++b) {
    size_t start = b * batch_size;
    size_t end = (start + batch_size < n) ? (start + batch_size) : n;
    size_t bs = end - start;

    FloatTensor input_view;
    FloatTensor target_view;
    FloatTensor *input_t = &input_view;
    FloatTensor *target_t = &target_view;
    if (!init_batch_input_view(input_t, X, t->input_shape, start, bs) ||
        !init_batch_target_view(target_t, Y, start, bs)) {
      status = NM_STATUS_ALLOC_FAILED;
      break;
    }

    tensor_pool_reset(&pool);
    memset(acts, 0, (layers + 1) * sizeof(FloatTensor *));
    memset(pidx, 0, layers * sizeof(FloatTensor *));

    acts[0] = input_t;
    bool fwd_ok = true;
    for (size_t i = 0; i < layers; ++i) {
      acts[i + 1] = layer_forward(&t->layers[i], acts[i], &pidx[i], &pool,
                                  t->backend);
      if (!acts[i + 1]) {
        fwd_ok = false;
        break;
      }
      if (t->training->autograd_mode == NM_AUTOGRAD_TAPE) {
        NmShape out_shape = shape_from_tensor(acts[i + 1]);
        if (!exec_trace_push(&t->training->last_trace, t->layers[i].type,
                             out_shape, acts[i + 1]->numel)) {
          fwd_ok = false;
          break;
        }
      }
    }

    if (!fwd_ok) {
      status = NM_STATUS_INTERNAL_ERROR;
      goto batch_cleanup;
    }

    if (track_loss && acts[layers]->ndim == 2) {
      FloatMatrix out_v = {bs, acts[layers]->shape[1],
                           bs * acts[layers]->shape[1], acts[layers]->values};
      FloatMatrix tgt_v = {bs, Y->cols, bs * Y->cols, target_t->values};
      float batch_loss = (t->loss_type == NM_LOSS_CROSS_ENTROPY)
                             ? nm_loss_cross_entropy(&out_v, &tgt_v)
                             : nm_loss_mse(&out_v, &tgt_v);
      total_loss += batch_loss * (float)bs;
      total_samples += bs;
    }

    t->training->optimizer_step += 1;
    FloatTensor *grad = tensor_pool_clone(&pool, acts[layers]);
    if (!grad) {
      status = NM_STATUS_ALLOC_FAILED;
      goto batch_cleanup;
    }
    bool allow_nondet = t->training->batch_parallel.enabled &&
                        !t->training->batch_parallel.deterministic;
#pragma omp parallel for if (allow_nondet)
    for (size_t j = 0; j < grad->numel; ++j)
      grad->values[j] -= target_t->values[j];

    for (ssize_t i = (ssize_t)layers - 1; i >= 0; --i) {
      bool skip = (i == (ssize_t)layers - 1) && shortcut;
      FloatTensor *new_grad = layer_backward(
          &t->layers[i], acts[i], acts[i + 1], grad, pidx[i], lr, skip,
          &t->training->optimizer_config, t->training->optimizer_step, &pool,
          t->backend);
      grad = new_grad;
      if (!grad && i > 0) {
        status = NM_STATUS_INTERNAL_ERROR;
        goto batch_cleanup;
      }
    }

  batch_cleanup:
    if (status != NM_STATUS_OK)
      break;
  }

  free(acts);
  free(pidx);
  tensor_pool_destroy(&pool);

  if (out_epoch_loss && total_samples > 0)
    *out_epoch_loss = total_loss / (float)total_samples;
  return status;
}

NmStatus nm_train_fit(NmTrainable *t, const FloatMatrix *X, const FloatMatrix *Y,
                      const NmTrainConfig *cfg) {
  if (!t || !t->training || !t->backend || !X || !Y)
    return NM_STATUS_INVALID_ARGUMENT;
  if (!trainable_weights_are_f32(t)) {
    log_error("Training requires all model weights to be f32. bf16 weights are "
              "only supported for inference.");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmTrainConfig effective = cfg ? *cfg : nm_train_defaults();
  cfg = &effective;

  size_t flat = shape_flat_size(t->input_shape);
  if (X->cols != flat) {
    log_error("Input columns (%zu) != expected flat size (%zu)", X->cols, flat);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (X->rows != Y->rows) {
    log_error("X rows (%zu) != Y rows (%zu)", X->rows, Y->rows);
    return NM_STATUS_INVALID_ARGUMENT;
  }

  if (cfg->learning_rate > 0.0f)
    t->training->optimizer_config.learning_rate = cfg->learning_rate;

  const size_t workers = choose_parallel_workers(t, cfg, X->cols);

#if defined(_OPENMP)
  if (t->training->batch_parallel.enabled && workers > 0)
    omp_set_num_threads((int)workers);
#endif

  if (t->training->batch_parallel.enabled && cfg->verbose) {
    log_info("Batch-parallel config active (enabled=%d, deterministic=%d, "
             "workers=%zu%s). Parameter updates remain in stable batch order.",
             (int)t->training->batch_parallel.enabled,
             (int)t->training->batch_parallel.deterministic, workers,
             t->training->batch_parallel.num_workers == 0 ? ", auto" : "");
  }

  for (size_t e = 0; e < cfg->epochs; ++e) {
    float epoch_loss = 0.0f;
    NmStatus status =
        model_train_one_epoch(t, X, Y, cfg->batch_size, cfg->learning_rate,
                              &epoch_loss, cfg->verbose);
    if (status != NM_STATUS_OK) {
      log_error("Training failed at epoch %zu (status=%d)", e + 1, (int)status);
      return status;
    }
    if (cfg->verbose)
      log_info("Epoch %zu/%zu — loss: %.4f", e + 1, cfg->epochs, epoch_loss);
  }
  return NM_STATUS_OK;
}

NmStatus nm_seq_fit(NmSeq *m, const FloatMatrix *X, const FloatMatrix *Y,
                    const NmTrainConfig *cfg) {
  if (!m || !X || !Y)
    return NM_STATUS_INVALID_ARGUMENT;
  if (m->training.inference_only)
    return NM_STATUS_INVALID_ARGUMENT;
  if (!m->compiled) {
    log_error(
        "Model must be compiled before training. Call nm_seq_compile() first.");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  NmTrainable trainable = {.layers = m->layers,
                           .num_layers = m->num_layers,
                           .input_shape = m->input_shape,
                           .loss_type = m->compile_config.loss_type,
                           .training = &m->training,
                           .backend = model_backend_ops(m)};
  return nm_train_fit(&trainable, X, Y, cfg);
}

static NmStatus run_dense_only(const NmLayerRuntime *runtime,
                               NmInferencePlan *plan,
                               const NmBackendOps *ops,
                               const FloatMatrix *input,
                               FloatMatrix *out_prediction) {
  size_t max_units = 0;
  for (size_t i = 0; i < runtime->count; ++i) {
    const NmDenseParams *p = (const NmDenseParams *)runtime->layers[i].params;
    if (p->weights->cols > max_units)
      max_units = p->weights->cols;
  }

  FloatTensor *scratch_tensor_a = NULL;
  FloatTensor *scratch_tensor_b = NULL;
  FloatMatrix scratch_a = {0};
  FloatMatrix scratch_b = {0};
  if (runtime->count > 1) {
    scratch_tensor_a =
        tensor_pool_acquire(&plan->pool, 2, (size_t[]){input->rows, max_units});
    scratch_tensor_b =
        tensor_pool_acquire(&plan->pool, 2, (size_t[]){input->rows, max_units});
    if (!scratch_tensor_a || !scratch_tensor_b)
      return NM_STATUS_ALLOC_FAILED;
    scratch_a = (FloatMatrix){.rows = input->rows,
                              .cols = max_units,
                              .capacity = scratch_tensor_a->numel,
                              .values = scratch_tensor_a->values};
    scratch_b = (FloatMatrix){.rows = input->rows,
                              .cols = max_units,
                              .capacity = scratch_tensor_b->numel,
                              .values = scratch_tensor_b->values};
  }

  const FloatMatrix *current = input;
  for (size_t i = 0; i < runtime->count; ++i) {
    const NmLayerEntry *layer = &runtime->layers[i];
    const NmDenseParams *params = (const NmDenseParams *)layer->params;
    FloatMatrix *dst = (i + 1 == runtime->count) ? out_prediction
                       : ((i & 1U) == 0U)        ? &scratch_a
                                                 : &scratch_b;
    dst->rows = input->rows;
    dst->cols = params->weights->cols;
    if (!dense_matrix_forward_into(params, layer->activation, current, dst, ops))
      return NM_STATUS_INTERNAL_ERROR;
    current = dst;
  }
  inference_plan_capture(plan, &plan->pool, input->rows, out_prediction->cols);
  return NM_STATUS_OK;
}

NmStatus nm_run_graph_plan_into(const NmLayerRuntime *runtime,
                                NmInferencePlan *plan,
                                const NmExecPlan *exec_plan,
                                const NmBackendOps *ops,
                                const FloatMatrix *input,
                                FloatMatrix *out_prediction) {
  if (!runtime || !plan || !exec_plan || !ops || !input || !out_prediction)
    return NM_STATUS_INVALID_ARGUMENT;
  if (exec_plan->count == 0 || exec_plan->count != runtime->count ||
      exec_plan->output_cols == 0 || input->cols != exec_plan->input_cols ||
      input->rows == 0 || out_prediction->rows != input->rows ||
      out_prediction->cols != exec_plan->output_cols ||
      out_prediction->capacity < out_prediction->rows * out_prediction->cols)
    return NM_STATUS_INVALID_ARGUMENT;

  inference_plan_reset(plan);
  if (exec_plan->dense_only)
    return run_dense_only(runtime, plan, ops, input, out_prediction);

  FloatTensor input_view;
  if (!init_batch_input_view(&input_view, input, exec_plan->input_shape, 0,
                             input->rows))
    return NM_STATUS_ALLOC_FAILED;

  FloatTensor *current = &input_view;
  for (size_t i = 0; i < exec_plan->count; ++i) {
    const NmLayerEntry *layer = &runtime->layers[i];
    if (exec_plan->steps[i].kind == NM_EXEC_STEP_ACTIVATION) {
      current = tensor_pool_clone(&plan->pool, current);
      if (current)
        layer_apply_activation(exec_plan->steps[i].activation, current);
    } else {
      FloatTensor *pool_indices = NULL;
      current = layer_forward(layer, current, &pool_indices, &plan->pool, ops);
    }
    if (!current)
      return NM_STATUS_INTERNAL_ERROR;
  }
  if (current->ndim != 2 || current->shape[0] != out_prediction->rows ||
      current->shape[1] != out_prediction->cols)
    return NM_STATUS_INTERNAL_ERROR;
  memcpy(out_prediction->values, current->values,
         current->numel * sizeof(float));
  inference_plan_capture(plan, &plan->pool, input->rows, out_prediction->cols);
  return NM_STATUS_OK;
}

static NmStatus nm_seq_run_inference_plan_into(const NmSeq *m,
                                               const FloatMatrix *input,
                                               FloatMatrix *out_prediction) {
  NmSeq *mutable_model = (NmSeq *)m;
  if (!m->compiled)
    return NM_STATUS_INVALID_ARGUMENT;
  NmLayerRuntime runtime = {.layers = m->layers, .count = m->num_layers};
  return nm_run_graph_plan_into(&runtime, &mutable_model->runtime.inference_plan,
                                &m->runtime.exec_plan, model_backend_ops(m),
                                input, out_prediction);
}

NmStatus nm_seq_predict(const NmSeq *m, const FloatMatrix *input,
                        FloatMatrix **out_prediction) {
  if (!m || !input || !out_prediction)
    return NM_STATUS_INVALID_ARGUMENT;
  if (!m->compiled || m->runtime.exec_plan.output_cols == 0)
    return NM_STATUS_INVALID_ARGUMENT;

  *out_prediction = sm_create(input->rows, m->runtime.exec_plan.output_cols);
  if (!*out_prediction)
    return NM_STATUS_ALLOC_FAILED;

  NmStatus status = nm_seq_run_inference_plan_into(m, input, *out_prediction);
  if (status != NM_STATUS_OK) {
    sm_destroy(*out_prediction);
    *out_prediction = NULL;
  }
  return status;
}

NmStatus nm_seq_predict_into(const NmSeq *m, const FloatMatrix *input,
                             FloatMatrix *out_prediction) {
  return nm_seq_run_inference_plan_into(m, input, out_prediction);
}

NmStatus nm_seq_evaluate(const NmSeq *m, const FloatMatrix *X,
                         const FloatMatrix *Y, float *out_accuracy) {
  if (!m || !X || !Y || !out_accuracy)
    return NM_STATUS_INVALID_ARGUMENT;

  FloatMatrix *pred = NULL;
  NmStatus status = nm_seq_predict(m, X, &pred);
  if (status != NM_STATUS_OK)
    return status;

  status = nm_accuracy(pred, Y, out_accuracy);
  sm_destroy(pred);
  return status;
}
