/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Layer, backend and batched-tensor execution helpers for nm models.
 */

#include "nm_model_internal.h"
#include <log.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

const char *layer_type_name(NmLayerType t) {
  switch (t) {
  case NM_LAYER_DENSE:
    return "Dense";
  case NM_LAYER_CONV2D:
    return "Conv2D";
  case NM_LAYER_MAXPOOL2D:
    return "MaxPool2D";
  case NM_LAYER_AVGPOOL2D:
    return "AvgPool2D";
  case NM_LAYER_BATCHNORM2D:
    return "BatchNorm2D";
  case NM_LAYER_FLATTEN:
    return "Flatten";
  default:
    return "Unknown";
  }
}

size_t layer_param_count(const NmLayerEntry *e) {
  switch (e->type) {
  case NM_LAYER_DENSE: {
    const NmDenseParams *p = (const NmDenseParams *)e->params;
    return p->weights->rows * p->weights->cols + p->bias->cols;
  }
  case NM_LAYER_CONV2D: {
    const NmConv2dInternalParams *p = (const NmConv2dInternalParams *)e->params;
    return p->weight->numel + p->bias->numel;
  }
  default:
    return 0;
  }
}

FloatTensor *he_init_4d(size_t cout, size_t cin, size_t kh, size_t kw) {
  size_t fan_in = cin * kh * kw;
  size_t numel = cout * cin * kh * kw;
  FloatMatrix *tmp = sm_create_random_he(cout, cin * kh * kw, fan_in);
  if (!tmp)
    return NULL;

  size_t shape[] = {cout, cin, kh, kw};
  FloatTensor *t = st_create(4, shape);
  if (!t) {
    sm_destroy(tmp);
    return NULL;
  }
  memcpy(t->values, tmp->values, numel * sizeof(float));
  sm_destroy(tmp);
  return t;
}

void destroy_layer_params(NmLayerEntry *e) {
  if (!e || !e->params)
    return;
  switch (e->type) {
  case NM_LAYER_DENSE: {
    NmDenseParams *p = e->params;
    sm_destroy(p->weights);
    sm_destroy(p->bias);
    optimizer_slot_free(&p->w_slot);
    optimizer_slot_free(&p->b_slot);
    break;
  }
  case NM_LAYER_CONV2D: {
    NmConv2dInternalParams *p = e->params;
    st_destroy(p->weight);
    st_destroy(p->bias);
    optimizer_slot_free(&p->w_slot);
    optimizer_slot_free(&p->b_slot);
    break;
  }
  case NM_LAYER_MAXPOOL2D:
  case NM_LAYER_AVGPOOL2D:
    break;
  default:
    break;
  }
  free(e->params);
  e->params = NULL;
}

/* Activations that sm_gemm_bias_act fuses into the GEMM; softmax needs whole
 * rows and stays a separate pass. */
static bool fused_activation(NmActivation act, SmActivation *out) {
  switch (act) {
  case NM_ACT_NONE:
  case NM_ACT_SOFTMAX:
    *out = SM_ACT_NONE;
    return true;
  case NM_ACT_RELU:
    *out = SM_ACT_RELU;
    return true;
  case NM_ACT_SIGMOID:
    *out = SM_ACT_SIGMOID;
    return true;
  case NM_ACT_TANH:
    *out = SM_ACT_TANH;
    return true;
  }
  return false;
}

static void apply_activation_tensor(NmActivation act, FloatTensor *t) {
  switch (act) {
  case NM_ACT_RELU:
    st_apply_relu(t);
    break;
  case NM_ACT_SIGMOID:
    st_apply_sigmoid(t);
    break;
  case NM_ACT_TANH:
    st_apply_tanh(t);
    break;
  case NM_ACT_SOFTMAX: {
    if (t->ndim != 2)
      break;
    FloatMatrix view = {.rows = t->shape[0],
                        .cols = t->shape[1],
                        .capacity = t->numel,
                        .values = t->values};
    nm_softmax(&view);
    break;
  }
  case NM_ACT_NONE:
    break;
  }
}

void layer_apply_activation(NmActivation act, FloatTensor *t) {
  apply_activation_tensor(act, t);
}

static void apply_activation_backward_tensor(NmActivation act,
                                             const FloatTensor *activation,
                                             FloatTensor *grad) {
  switch (act) {
  case NM_ACT_RELU:
    st_apply_relu_backward(activation, grad);
    break;
  case NM_ACT_SIGMOID:
    st_apply_sigmoid_backward(activation, grad);
    break;
  case NM_ACT_TANH:
    st_apply_tanh_backward(activation, grad);
    break;
  case NM_ACT_SOFTMAX:
  case NM_ACT_NONE:
    break;
  }
}

static void apply_activation_matrix(NmActivation act, FloatMatrix *mat,
                                    const NmBackendOps *backend) {
  if (!mat)
    return;
  switch (act) {
  case NM_ACT_RELU:
    nm_relu(mat);
    break;
  case NM_ACT_SIGMOID:
    nm_sigmoid(mat);
    break;
  case NM_ACT_TANH:
    nm_tanh(mat);
    break;
  case NM_ACT_SOFTMAX:
    backend->softmax(mat);
    break;
  case NM_ACT_NONE:
    break;
  }
}

static void apply_activation_backward_matrix(NmActivation act,
                                             const FloatMatrix *activation,
                                             FloatMatrix *grad) {
  if (!activation || !grad)
    return;
  switch (act) {
  case NM_ACT_RELU:
    nm_relu_grad(activation, grad);
    break;
  case NM_ACT_SIGMOID:
    nm_sigmoid_grad(activation, grad);
    break;
  case NM_ACT_TANH:
    nm_tanh_grad(activation, grad);
    break;
  case NM_ACT_SOFTMAX:
  case NM_ACT_NONE:
    break;
  }
}

static bool matrix_init_borrowed_view(FloatMatrix *view, size_t rows,
                                      size_t cols, float *values,
                                      size_t capacity) {
  if (!view || !values || rows == 0 || cols == 0 || capacity < rows * cols)
    return false;
  view->rows = rows;
  view->cols = cols;
  view->capacity = capacity;
  view->values = values;
  return true;
}

bool dense_matrix_forward_into(const NmDenseParams *p, NmActivation act,
                                      const FloatMatrix *input,
                                      FloatMatrix *output,
                                      const NmBackendOps *backend) {
  if (!p || !input || !output || !backend || !p->weights || !p->bias ||
      input->cols != p->weights->rows || output->rows != input->rows ||
      output->cols != p->weights->cols ||
      output->capacity < output->rows * output->cols) {
    return false;
  }

  SmActivation fused;
  if (!fused_activation(act, &fused) ||
      !backend->gemm_bias_act(output, input, SM_NO_TRANSPOSE, p->weights,
                              SM_NO_TRANSPOSE, p->bias, fused)) {
    return false;
  }
  if (act == NM_ACT_SOFTMAX)
    apply_activation_matrix(act, output, backend);
  return true;
}

bool trainable_is_dense_only(const NmTrainable *t) {
  if (!t || t->num_layers == 0 || t->input_shape.ndim != 1)
    return false;
  for (size_t i = 0; i < t->num_layers; ++i) {
    if (t->layers[i].type != NM_LAYER_DENSE)
      return false;
  }
  return true;
}

bool trainable_weights_are_f32(const NmTrainable *t) {
  if (!t)
    return true;
  for (size_t i = 0; i < t->num_layers; ++i) {
    const NmLayerEntry *e = &t->layers[i];
    if (e->type == NM_LAYER_CONV2D && e->params) {
      const NmConv2dInternalParams *p =
          (const NmConv2dInternalParams *)e->params;
      if ((p->weight && p->weight->dtype != ST_DTYPE_F32) ||
          (p->bias && p->bias->dtype != ST_DTYPE_F32))
        return false;
    }
  }
  return true;
}

bool trainable_uses_softmax_ce_shortcut(const NmTrainable *t) {
  if (!t || t->num_layers == 0 || t->loss_type != NM_LOSS_CROSS_ENTROPY)
    return false;
  return t->layers[t->num_layers - 1].activation == NM_ACT_SOFTMAX;
}

NmStatus model_train_one_epoch_dense_fast(NmTrainable *t, const FloatMatrix *X,
                                          const FloatMatrix *Y,
                                          size_t batch_size, float lr,
                                          float *out_epoch_loss,
                                          bool track_loss) {
  (void)lr;

  if (!t || !X || !Y || !trainable_is_dense_only(t) ||
      batch_size == 0) {
    return NM_STATUS_INVALID_ARGUMENT;
  }

  const size_t num_samples = X->rows;
  const size_t num_batches = (num_samples + batch_size - 1) / batch_size;
  const size_t num_layers = t->num_layers;
  const bool shortcut = trainable_uses_softmax_ce_shortcut(t);
  const NmBackendOps *backend = t->backend;

  float total_loss = 0.0f;
  size_t total_samples = 0;

  FloatMatrix **acts = (FloatMatrix **)calloc(num_layers + 1, sizeof(*acts));
  if (!acts)
    return NM_STATUS_ALLOC_FAILED;

  size_t max_units = 0;
  size_t max_grad_weight = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    const NmDenseParams *p = (const NmDenseParams *)t->layers[i].params;
    if (!p || !p->weights || !p->bias) {
      free(acts);
      return NM_STATUS_INTERNAL_ERROR;
    }
    if (p->weights->cols > max_units)
      max_units = p->weights->cols;
    size_t grad_weight_cap = p->weights->rows * p->weights->cols;
    if (grad_weight_cap > max_grad_weight)
      max_grad_weight = grad_weight_cap;
  }

  for (size_t i = 0; i < num_layers; ++i) {
    const NmDenseParams *p = (const NmDenseParams *)t->layers[i].params;
    acts[i + 1] = sm_create(batch_size, p->weights->cols);
    if (!acts[i + 1]) {
      for (size_t j = 1; j < i + 1; ++j)
        sm_destroy(acts[j]);
      free(acts);
      return NM_STATUS_ALLOC_FAILED;
    }
  }

  FloatMatrix *grad_a = sm_create(batch_size, max_units);
  FloatMatrix *grad_b = sm_create(batch_size, max_units);
  FloatMatrix *scratch_gw = sm_create(1, max_grad_weight);
  FloatMatrix *scratch_gb = sm_create(1, max_units);
  if (!grad_a || !grad_b || !scratch_gw || !scratch_gb) {
    sm_destroy(grad_a);
    sm_destroy(grad_b);
    sm_destroy(scratch_gw);
    sm_destroy(scratch_gb);
    for (size_t i = 1; i <= num_layers; ++i)
      sm_destroy(acts[i]);
    free(acts);
    return NM_STATUS_ALLOC_FAILED;
  }

  if (t->training->autograd_mode == NM_AUTOGRAD_TAPE)
    exec_trace_reset(&t->training->last_trace);

  NmStatus status = NM_STATUS_OK;

  for (size_t batch = 0; batch < num_batches; ++batch) {
    const size_t start = batch * batch_size;
    const size_t end =
        (start + batch_size < num_samples) ? (start + batch_size) : num_samples;
    const size_t current_batch = end - start;

    FloatMatrix input_batch = {0};
    FloatMatrix target_batch = {0};
    if (!matrix_init_borrowed_view(&input_batch, current_batch, X->cols,
                                   (float *)X->values + start * X->cols,
                                   current_batch * X->cols) ||
        !matrix_init_borrowed_view(&target_batch, current_batch, Y->cols,
                                   (float *)Y->values + start * Y->cols,
                                   current_batch * Y->cols)) {
      status = NM_STATUS_INTERNAL_ERROR;
      break;
    }
    acts[0] = &input_batch;

    bool forward_ok = true;
    for (size_t i = 0; i < num_layers; ++i) {
      const NmLayerEntry *layer = &t->layers[i];
      const NmDenseParams *p = (const NmDenseParams *)layer->params;
      acts[i + 1]->rows = current_batch;
      acts[i + 1]->cols = p->weights->cols;
      if (!dense_matrix_forward_into(p, layer->activation, acts[i], acts[i + 1],
                                     backend)) {
        forward_ok = false;
        break;
      }
      if (t->training->autograd_mode == NM_AUTOGRAD_TAPE) {
        NmShape out_shape = {.ndim = 1, .dims = {acts[i + 1]->cols, 0, 0}};
        if (!exec_trace_push(&t->training->last_trace, layer->type, out_shape,
                             acts[i + 1]->rows * acts[i + 1]->cols)) {
          forward_ok = false;
          break;
        }
      }
    }

    if (!forward_ok) {
      status = NM_STATUS_INTERNAL_ERROR;
      break;
    }

    if (track_loss) {
      float batch_loss =
          (t->loss_type == NM_LOSS_CROSS_ENTROPY)
              ? nm_loss_cross_entropy(acts[num_layers], &target_batch)
              : nm_loss_mse(acts[num_layers], &target_batch);
      total_loss += batch_loss * (float)current_batch;
      total_samples += current_batch;
    }

    t->training->optimizer_step += 1;
    grad_a->rows = current_batch;
    grad_a->cols = acts[num_layers]->cols;
    memcpy(grad_a->values, acts[num_layers]->values,
           current_batch * acts[num_layers]->cols * sizeof(float));

    if (shortcut) {
      nm_softmax_ce_grad(acts[num_layers], &target_batch, grad_a);
    } else {
#pragma omp parallel for simd if (current_batch * acts[num_layers]->cols > 16384)
      for (size_t j = 0; j < current_batch * acts[num_layers]->cols; ++j)
        grad_a->values[j] -= target_batch.values[j];
    }

    FloatMatrix *grad = grad_a;
    for (size_t rev = num_layers; rev-- > 0;) {
      size_t i = rev;
      NmLayerEntry *layer = &t->layers[i];
      NmDenseParams *p = (NmDenseParams *)layer->params;
      const FloatMatrix *layer_input = acts[i];
      const FloatMatrix *layer_activation = acts[i + 1];
      const bool skip_activation = (i + 1 == num_layers) && shortcut;

      if (!skip_activation && layer->activation != NM_ACT_NONE) {
        apply_activation_backward_matrix(layer->activation, layer_activation,
                                         grad);
      }

      FloatMatrix *next_grad = NULL;
      if (i > 0) {
        next_grad = (grad == grad_a) ? grad_b : grad_a;
        next_grad->rows = current_batch;
        next_grad->cols = layer_input->cols;
        if (!backend->gemm(next_grad, 1.0f, grad, SM_NO_TRANSPOSE, p->weights,
                           SM_TRANSPOSE, 0.0f)) {
          status = NM_STATUS_INTERNAL_ERROR;
          break;
        }
      }

      scratch_gw->rows = layer_input->cols;
      scratch_gw->cols = grad->cols;
      if (!backend->gemm(scratch_gw, 1.0f, layer_input, SM_TRANSPOSE, grad,
                         SM_NO_TRANSPOSE, 0.0f)) {
        status = NM_STATUS_INTERNAL_ERROR;
        break;
      }

      scratch_gb->rows = 1;
      scratch_gb->cols = grad->cols;
      for (size_t col = 0; col < grad->cols; ++col) {
        float sum = 0.0f;
        for (size_t row = 0; row < current_batch; ++row)
          sum += grad->values[row * grad->cols + col];
        scratch_gb->values[col] = sum;
      }

      if (!optimizer_apply_matrix(&t->training->optimizer_config,
                                  t->training->optimizer_step, &p->w_slot,
                                  p->weights, scratch_gw,
                                  1.0f / (float)current_batch) ||
          !optimizer_apply_matrix(
              &t->training->optimizer_config, t->training->optimizer_step,
              &p->b_slot, p->bias, scratch_gb, 1.0f / (float)current_batch)) {
        status = NM_STATUS_INTERNAL_ERROR;
        break;
      }

      grad = next_grad;
    }

    if (status != NM_STATUS_OK)
      break;
  }

  if (out_epoch_loss && total_samples > 0)
    *out_epoch_loss = total_loss / (float)total_samples;

  sm_destroy(grad_a);
  sm_destroy(grad_b);
  sm_destroy(scratch_gw);
  sm_destroy(scratch_gb);
  for (size_t i = 1; i <= num_layers; ++i)
    sm_destroy(acts[i]);
  free(acts);
  return status;
}

static bool backend_gemm_default(FloatMatrix *c, float alpha,
                                 const FloatMatrix *a, SmTranspose ta,
                                 const FloatMatrix *b, SmTranspose tb,
                                 float beta) {
  return sm_gemm(c, alpha, a, ta, b, tb, beta);
}

static bool backend_gemm_bias_relu_default(FloatMatrix *out,
                                           const FloatMatrix *in,
                                           SmTranspose ta, const FloatMatrix *w,
                                           SmTranspose tw,
                                           const FloatMatrix *bias) {
  return sm_gemm_bias_relu(out, in, ta, w, tw, bias);
}

static bool backend_gemm_bias_act_default(FloatMatrix *out,
                                          const FloatMatrix *in,
                                          SmTranspose ta, const FloatMatrix *w,
                                          SmTranspose tw,
                                          const FloatMatrix *bias,
                                          SmActivation act) {
  return sm_gemm_bias_act(out, in, ta, w, tw, bias, act);
}

static void backend_add_rowwise_default(FloatMatrix *mat,
                                        const FloatMatrix *row) {
  nm_add_rowwise(mat, row);
}

static void backend_softmax_default(FloatMatrix *mat) { nm_softmax(mat); }

static bool backend_conv2d_fwd_default(const FloatTensor *input,
                                       const FloatTensor *weight,
                                       const FloatTensor *bias,
                                       const StConv2dParams *params,
                                       FloatTensor *output) {
  return st_conv2d_nchw(input, weight, bias, params, output);
}

static bool backend_conv2d_bwd_data_default(const FloatTensor *grad_output,
                                            const FloatTensor *weight,
                                            const StConv2dParams *params,
                                            FloatTensor *grad_input) {
  return st_conv2d_backward_data_nchw(grad_output, weight, params, grad_input);
}

static void backend_conv2d_bwd_weight_default(const FloatTensor *input,
                                              const FloatTensor *grad_output,
                                              const StConv2dParams *params,
                                              FloatTensor *grad_weight) {
  st_conv2d_backward_weight_nchw(input, grad_output, params, grad_weight);
}

static void backend_conv2d_bwd_bias_default(const FloatTensor *grad_output,
                                            FloatTensor *grad_bias) {
  st_conv2d_backward_bias(grad_output, grad_bias);
}

static bool backend_maxpool2d_fwd_default(const FloatTensor *input,
                                          size_t kernel_h, size_t kernel_w,
                                          size_t stride_h, size_t stride_w,
                                          size_t pad_h, size_t pad_w,
                                          FloatTensor *output,
                                          FloatTensor *indices) {
  return st_maxpool2d_nchw(input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w, output, indices);
}

static bool backend_maxpool2d_bwd_default(const FloatTensor *grad_output,
                                          const FloatTensor *indices,
                                          size_t in_h, size_t in_w,
                                          FloatTensor *grad_input) {
  return st_maxpool2d_backward_nchw(grad_output, indices, in_h, in_w,
                                    grad_input);
}

const NmBackendOps NM_BACKEND_DEFAULT_OPS = {
    .gemm = backend_gemm_default,
    .gemm_bias_relu = backend_gemm_bias_relu_default,
    .gemm_bias_act = backend_gemm_bias_act_default,
    .add_rowwise = backend_add_rowwise_default,
    .softmax = backend_softmax_default,
    .conv2d_fwd = backend_conv2d_fwd_default,
    .conv2d_bwd_data = backend_conv2d_bwd_data_default,
    .conv2d_bwd_weight = backend_conv2d_bwd_weight_default,
    .conv2d_bwd_bias = backend_conv2d_bwd_bias_default,
    .maxpool2d_fwd = backend_maxpool2d_fwd_default,
    .maxpool2d_bwd = backend_maxpool2d_bwd_default,
};

static StConvBackend conv_backend_for_algo(const NmConv2dInternalParams *p,
                                           const FloatTensor *input) {
  if (!p)
    return ST_CONV_BACKEND_AUTO;

  switch (p->algo) {
  case NM_CONV_ALGO_DIRECT:
    return ST_CONV_BACKEND_CPU_OPT;
  case NM_CONV_ALGO_GEMM:
    return ST_CONV_BACKEND_GEMM;
  case NM_CONV_ALGO_REFERENCE:
    return ST_CONV_BACKEND_REFERENCE;
  case NM_CONV_ALGO_AUTO:
  default:
    break;
  }

  size_t kh = p->weight ? p->weight->shape[2] : 0;
  size_t kw = p->weight ? p->weight->shape[3] : 0;
  size_t cin = p->weight ? p->weight->shape[1] : 0;
  size_t cout = p->weight ? p->weight->shape[0] : 0;
  size_t n = input ? input->shape[0] : 1;

  if ((kh == 3 && kw == 3 && cin <= 128 && cout <= 256 && n <= 32) ||
      (kh == 1 && kw == 1)) {
    return ST_CONV_BACKEND_CPU_OPT;
  }
  if ((kh * kw) >= 9 || (cin * cout) >= 4096)
    return ST_CONV_BACKEND_GEMM;
  return ST_CONV_BACKEND_AUTO;
}

static FloatTensor *dense_fwd(const NmDenseParams *p, NmActivation act,
                              const FloatTensor *input, NmTensorPool *pool,
                              const NmBackendOps *backend) {
  size_t n = input->shape[0];
  size_t units = p->weights->cols;

  FloatTensor *output = tensor_pool_acquire(pool, 2, (size_t[]){n, units});
  if (!output)
    return NULL;

  FloatMatrix in_v = {.rows = n,
                      .cols = input->shape[1],
                      .capacity = n * input->shape[1],
                      .values = (float *)input->values};
  FloatMatrix out_v = {.rows = n,
                       .cols = units,
                       .capacity = n * units,
                       .values = output->values};

  SmActivation fused;
  if (!fused_activation(act, &fused) ||
      !backend->gemm_bias_act(&out_v, &in_v, SM_NO_TRANSPOSE, p->weights,
                              SM_NO_TRANSPOSE, p->bias, fused)) {
    return NULL;
  }
  if (act == NM_ACT_SOFTMAX)
    apply_activation_tensor(act, output);
  return output;
}

static FloatTensor *conv2d_fwd(const NmConv2dInternalParams *p,
                               NmActivation act, const FloatTensor *input,
                               NmTensorPool *pool,
                               const NmBackendOps *backend) {
  size_t n = input->shape[0];
  size_t cout = p->weight->shape[0];
  size_t in_h = input->shape[2];
  size_t in_w = input->shape[3];
  size_t kh = p->weight->shape[2];
  size_t kw = p->weight->shape[3];

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(in_h, in_w, kh, kw, &p->conv, &out_h, &out_w))
    return NULL;

  FloatTensor *output =
      tensor_pool_acquire(pool, 4, (size_t[]){n, cout, out_h, out_w});
  if (!output)
    return NULL;

  StConv2dParams run_params = p->conv;
  run_params.backend = conv_backend_for_algo(p, input);

  if (!backend->conv2d_fwd(input, p->weight, p->bias, &run_params, output))
    return NULL;

  if (act != NM_ACT_NONE)
    apply_activation_tensor(act, output);
  return output;
}

static FloatTensor *maxpool_fwd(const NmPool2dInternalParams *p,
                                const FloatTensor *input,
                                FloatTensor **out_indices, NmTensorPool *pool,
                                const NmBackendOps *backend) {
  size_t n = input->shape[0];
  size_t c = input->shape[1];
  size_t in_h = input->shape[2];
  size_t in_w = input->shape[3];

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_pool2d_output_hw(in_h, in_w, p->kernel_h, p->kernel_w, p->stride_h,
                           p->stride_w, p->pad_h, p->pad_w, &out_h, &out_w)) {
    return NULL;
  }

  FloatTensor *output =
      tensor_pool_acquire(pool, 4, (size_t[]){n, c, out_h, out_w});
  FloatTensor *indices =
      tensor_pool_acquire(pool, 4, (size_t[]){n, c, out_h, out_w});
  if (!output || !indices)
    return NULL;

  if (!backend->maxpool2d_fwd(input, p->kernel_h, p->kernel_w, p->stride_h,
                              p->stride_w, p->pad_h, p->pad_w, output,
                              indices)) {
    return NULL;
  }

  if (out_indices)
    *out_indices = indices;
  return output;
}

static FloatTensor *flatten_fwd(const FloatTensor *input, NmTensorPool *pool) {
  size_t n = input->shape[0];
  size_t flat = 1;
  for (size_t d = 1; d < input->ndim; ++d)
    flat *= input->shape[d];

  FloatTensor *output = tensor_pool_acquire(pool, 2, (size_t[]){n, flat});
  if (!output)
    return NULL;
  memcpy(output->values, input->values, input->numel * sizeof(float));
  return output;
}

static FloatTensor *dense_bwd(NmDenseParams *p, NmActivation act,
                              const FloatTensor *input,
                              const FloatTensor *activation, FloatTensor *grad,
                              const NmOptimizerConfig *opt_cfg,
                              uint64_t opt_step, bool skip_act,
                              NmTensorPool *pool, const NmBackendOps *backend) {
  size_t n = input->shape[0];
  size_t input_dim = input->shape[1];
  size_t units = p->weights->cols;

  if (!skip_act && act != NM_ACT_NONE)
    apply_activation_backward_tensor(act, activation, grad);

  FloatMatrix in_v = {n, input_dim, n * input_dim, (float *)input->values};
  FloatMatrix grad_v = {n, units, n * units, grad->values};

  FloatTensor *grad_input =
      tensor_pool_acquire(pool, 2, (size_t[]){n, input_dim});
  if (!grad_input)
    return NULL;
  FloatMatrix gi_v = {n, input_dim, n * input_dim, grad_input->values};
  if (!backend->gemm(&gi_v, 1.0f, &grad_v, SM_NO_TRANSPOSE, p->weights,
                     SM_TRANSPOSE, 0.0f)) {
    return NULL;
  }

  FloatTensor *grad_w_t =
      tensor_pool_acquire(pool, 2, (size_t[]){input_dim, units});
  if (!grad_w_t)
    return NULL;
  FloatMatrix grad_w_v = {input_dim, units, input_dim * units,
                          grad_w_t->values};
  if (!backend->gemm(&grad_w_v, 1.0f, &in_v, SM_TRANSPOSE, &grad_v,
                     SM_NO_TRANSPOSE, 0.0f)) {
    return NULL;
  }

  FloatTensor *grad_b_t = tensor_pool_acquire(pool, 1, (size_t[]){units});
  if (!grad_b_t)
    return NULL;
  for (size_t j = 0; j < units; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
      sum += grad->values[i * units + j];
    grad_b_t->values[j] = sum;
  }

  FloatMatrix grad_b_v = {1, units, units, grad_b_t->values};
  if (!optimizer_apply_matrix(opt_cfg, opt_step, &p->w_slot, p->weights,
                              &grad_w_v, 1.0f / (float)n)) {
    return NULL;
  }
  if (!optimizer_apply_matrix(opt_cfg, opt_step, &p->b_slot, p->bias, &grad_b_v,
                              1.0f / (float)n)) {
    return NULL;
  }

  return grad_input;
}

static FloatTensor *conv2d_bwd(NmConv2dInternalParams *p, NmActivation act,
                               const FloatTensor *input,
                               const FloatTensor *activation, FloatTensor *grad,
                               const NmOptimizerConfig *opt_cfg,
                               uint64_t opt_step, bool skip_act,
                               NmTensorPool *pool,
                               const NmBackendOps *backend) {
  size_t n = input->shape[0];

  if (!skip_act && act != NM_ACT_NONE)
    apply_activation_backward_tensor(act, activation, grad);

  FloatTensor *grad_input =
      tensor_pool_acquire(pool, input->ndim, input->shape);
  if (!grad_input)
    return NULL;
  StConv2dParams run_params = p->conv;
  run_params.backend = conv_backend_for_algo(p, input);
  if (!backend->conv2d_bwd_data(grad, p->weight, &run_params, grad_input))
    return NULL;

  FloatTensor *gw =
      tensor_pool_acquire(pool, p->weight->ndim, p->weight->shape);
  if (!gw)
    return NULL;
  backend->conv2d_bwd_weight(input, grad, &run_params, gw);

  size_t cout = p->weight->shape[0];
  FloatTensor *gb = tensor_pool_acquire(pool, 1, (size_t[]){cout});
  if (!gb)
    return NULL;
  backend->conv2d_bwd_bias(grad, gb);

  if (!optimizer_apply_tensor(opt_cfg, opt_step, &p->w_slot, p->weight, gw,
                              1.0f / (float)n)) {
    return NULL;
  }
  if (!optimizer_apply_tensor(opt_cfg, opt_step, &p->b_slot, p->bias, gb,
                              1.0f / (float)n)) {
    return NULL;
  }

  return grad_input;
}

static FloatTensor *maxpool_bwd(const NmPool2dInternalParams *p,
                                const FloatTensor *input, FloatTensor *grad,
                                FloatTensor *indices, NmTensorPool *pool,
                                const NmBackendOps *backend) {
  (void)p;
  FloatTensor *grad_input =
      tensor_pool_acquire(pool, input->ndim, input->shape);
  if (!grad_input)
    return NULL;
  st_fill(grad_input, 0.0f);

  size_t in_h = input->shape[2];
  size_t in_w = input->shape[3];
  if (!backend->maxpool2d_bwd(grad, indices, in_h, in_w, grad_input))
    return NULL;
  return grad_input;
}

static FloatTensor *flatten_bwd(NmShape orig, const FloatTensor *grad,
                                NmTensorPool *pool) {
  size_t n = grad->shape[0];
  FloatTensor *gi = NULL;
  if (orig.ndim == 3) {
    gi = tensor_pool_acquire(
        pool, 4, (size_t[]){n, orig.dims[0], orig.dims[1], orig.dims[2]});
  } else {
    gi = tensor_pool_acquire(pool, grad->ndim, grad->shape);
  }
  if (!gi)
    return NULL;
  memcpy(gi->values, grad->values, grad->numel * sizeof(float));
  return gi;
}

static FloatTensor *dense_layer_forward(const NmLayerEntry *layer,
                                        const NmLayerFwdCtx *ctx) {
  return dense_fwd((const NmDenseParams *)layer->params, layer->activation,
                   ctx->input, ctx->pool, ctx->backend);
}

static FloatTensor *dense_layer_backward(NmLayerEntry *layer,
                                         const NmLayerBwdCtx *ctx) {
  return dense_bwd((NmDenseParams *)layer->params, layer->activation,
                   ctx->input, ctx->activation, ctx->grad, ctx->opt_cfg,
                   ctx->opt_step, ctx->skip_act, ctx->pool, ctx->backend);
}

static FloatTensor *conv2d_layer_forward(const NmLayerEntry *layer,
                                         const NmLayerFwdCtx *ctx) {
  return conv2d_fwd((const NmConv2dInternalParams *)layer->params,
                    layer->activation, ctx->input, ctx->pool, ctx->backend);
}

static FloatTensor *conv2d_layer_backward(NmLayerEntry *layer,
                                          const NmLayerBwdCtx *ctx) {
  return conv2d_bwd((NmConv2dInternalParams *)layer->params, layer->activation,
                    ctx->input, ctx->activation, ctx->grad, ctx->opt_cfg,
                    ctx->opt_step, ctx->skip_act, ctx->pool, ctx->backend);
}

static FloatTensor *maxpool_layer_forward(const NmLayerEntry *layer,
                                          const NmLayerFwdCtx *ctx) {
  return maxpool_fwd((const NmPool2dInternalParams *)layer->params, ctx->input,
                     ctx->out_pool_idx, ctx->pool, ctx->backend);
}

static FloatTensor *maxpool_layer_backward(NmLayerEntry *layer,
                                           const NmLayerBwdCtx *ctx) {
  return maxpool_bwd((const NmPool2dInternalParams *)layer->params, ctx->input,
                     ctx->grad, ctx->pool_indices, ctx->pool, ctx->backend);
}

static FloatTensor *flatten_layer_forward(const NmLayerEntry *layer,
                                          const NmLayerFwdCtx *ctx) {
  (void)layer;
  return flatten_fwd(ctx->input, ctx->pool);
}

static FloatTensor *flatten_layer_backward(NmLayerEntry *layer,
                                           const NmLayerBwdCtx *ctx) {
  return flatten_bwd(layer->input_shape, ctx->grad, ctx->pool);
}

static const NmLayerOps NM_LAYER_OPS_DENSE = {
    .forward = dense_layer_forward,
    .backward = dense_layer_backward,
};

static const NmLayerOps NM_LAYER_OPS_CONV2D = {
    .forward = conv2d_layer_forward,
    .backward = conv2d_layer_backward,
};

static const NmLayerOps NM_LAYER_OPS_MAXPOOL2D = {
    .forward = maxpool_layer_forward,
    .backward = maxpool_layer_backward,
};

static const NmLayerOps NM_LAYER_OPS_FLATTEN = {
    .forward = flatten_layer_forward,
    .backward = flatten_layer_backward,
};

const NmLayerOps *layer_ops_for_type(NmLayerType type) {
  switch (type) {
  case NM_LAYER_DENSE:
    return &NM_LAYER_OPS_DENSE;
  case NM_LAYER_CONV2D:
    return &NM_LAYER_OPS_CONV2D;
  case NM_LAYER_MAXPOOL2D:
    return &NM_LAYER_OPS_MAXPOOL2D;
  case NM_LAYER_FLATTEN:
    return &NM_LAYER_OPS_FLATTEN;
  case NM_LAYER_AVGPOOL2D:
  case NM_LAYER_BATCHNORM2D:
    return NULL;
  }
  return NULL;
}

FloatTensor *layer_forward(const NmLayerEntry *e, const FloatTensor *input,
                           FloatTensor **out_pool_idx, NmTensorPool *pool,
                           const NmBackendOps *backend) {
  if (out_pool_idx)
    *out_pool_idx = NULL;

  if (!e || !e->ops || !e->ops->forward) {
    log_error("Layer forward op missing for type %d", (int)(e ? e->type : -1));
    return NULL;
  }

  NmLayerFwdCtx ctx = {
      .input = input,
      .out_pool_idx = out_pool_idx,
      .pool = pool,
      .backend = backend,
  };
  return e->ops->forward(e, &ctx);
}

FloatTensor *layer_backward(NmLayerEntry *e, const FloatTensor *input,
                            const FloatTensor *activation, FloatTensor *grad,
                            FloatTensor *pool_indices, float lr, bool skip_act,
                            const NmOptimizerConfig *opt_cfg, uint64_t opt_step,
                            NmTensorPool *pool, const NmBackendOps *backend) {
  (void)lr;

  if (!e || !e->ops || !e->ops->backward) {
    log_error("Layer backward op missing for type %d", (int)(e ? e->type : -1));
    return NULL;
  }

  NmLayerBwdCtx ctx = {
      .input = input,
      .activation = activation,
      .grad = grad,
      .pool_indices = pool_indices,
      .skip_act = skip_act,
      .opt_cfg = opt_cfg,
      .opt_step = opt_step,
      .pool = pool,
      .backend = backend,
  };
  return e->ops->backward(e, &ctx);
}

bool init_batch_input_view(FloatTensor *out, const FloatMatrix *X,
                           NmShape shape, size_t start, size_t count) {
  size_t flat = shape_flat_size(shape);
  float *ptr = (float *)X->values + start * flat;
  if (shape.ndim == 1) {
    return tensor_init_borrowed_view(out, 2, (size_t[]){count, flat}, ptr,
                                     count * flat, ST_LAYOUT_CONTIGUOUS);
  }
  return tensor_init_borrowed_view(
      out, 4, (size_t[]){count, shape.dims[0], shape.dims[1], shape.dims[2]},
      ptr, count * flat, ST_LAYOUT_NCHW);
}

bool init_batch_target_view(FloatTensor *out, const FloatMatrix *Y,
                            size_t start, size_t count) {
  size_t cols = Y->cols;
  float *ptr = (float *)Y->values + start * cols;
  return tensor_init_borrowed_view(out, 2, (size_t[]){count, cols}, ptr,
                                   count * cols, ST_LAYOUT_CONTIGUOUS);
}
