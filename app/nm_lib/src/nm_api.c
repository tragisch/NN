/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_api.h"
#include "nm_engine_c.h"
#include "nm_export_graphir.h"
#include "nm_graph_ir_internal.h"
#include "nm.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <pthread.h>

/* ---- Error message ---------------------------------------------------- */

static _Thread_local char g_error[256];

static void set_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(g_error, sizeof(g_error), fmt, args);
  va_end(args);
}

const char *nm_last_error(void) { return g_error; }

/* ---- Tensors ---------------------------------------------------------- */

NmTensor nm_tensor_2d(float *data, size_t rows, size_t cols) {
  return (NmTensor){.data = data, .ndim = 2, .dims = {rows, cols, 0, 0}};
}

size_t nm_tensor_numel(const NmTensor *tensor) {
  if (!tensor || tensor->ndim < 1 || tensor->ndim > 4)
    return 0;
  size_t n = 1;
  for (size_t i = 0; i < tensor->ndim; ++i)
    n *= tensor->dims[i];
  return n;
}

void nm_tensor_free(NmTensor *tensor) {
  if (!tensor)
    return;
  free(tensor->data);
  *tensor = (NmTensor){0};
}

/* A matrix view of a tensor: one row per example. */
static bool tensor_view(const NmTensor *t, const char *what, FloatMatrix *out) {
  if (!t || !t->data || t->ndim < 2 || t->ndim > 4) {
    set_error("%s: tensor needs data and 2 to 4 dimensions", what);
    return false;
  }
  for (size_t i = 0; i < t->ndim; ++i) {
    if (t->dims[i] == 0) {
      set_error("%s: dimension %zu is zero", what, i);
      return false;
    }
  }
  size_t rows = t->dims[0];
  *out = (FloatMatrix){.rows = rows,
                       .cols = nm_tensor_numel(t) / rows,
                       .capacity = nm_tensor_numel(t),
                       .values = t->data};
  return true;
}

/* ---- Spec ------------------------------------------------------------- */

typedef struct {
  NmLayerType type;
  NmDenseConfig dense;
  NmConv2dConfig conv;
  NmPool2dConfig pool;
} SpecLayer;

struct NmSpec {
  NmShape input;
  SpecLayer *layers;
  size_t count;
  size_t capacity;
};

NmSpec *nm_spec_new(NmShape input) {
  if (input.ndim < 1 || input.ndim > 3) {
    set_error("nm_spec_new: input shape needs 1 to 3 dimensions");
    return NULL;
  }
  for (size_t i = 0; i < input.ndim; ++i) {
    if (input.dims[i] == 0) {
      set_error("nm_spec_new: input dimension %zu is zero", i);
      return NULL;
    }
  }
  NmSpec *spec = calloc(1, sizeof(*spec));
  if (!spec) {
    set_error("nm_spec_new: out of memory");
    return NULL;
  }
  spec->input = input;
  return spec;
}

void nm_spec_free(NmSpec *spec) {
  if (!spec)
    return;
  free(spec->layers);
  free(spec);
}

static SpecLayer *spec_push(NmSpec *spec, NmLayerType type) {
  if (!spec) {
    set_error("spec is NULL");
    return NULL;
  }
  if (spec->count == spec->capacity) {
    size_t cap = spec->capacity ? spec->capacity * 2 : 8;
    SpecLayer *grown = realloc(spec->layers, cap * sizeof(*grown));
    if (!grown) {
      set_error("out of memory");
      return NULL;
    }
    spec->layers = grown;
    spec->capacity = cap;
  }
  SpecLayer *layer = &spec->layers[spec->count++];
  memset(layer, 0, sizeof(*layer));
  layer->type = type;
  return layer;
}

NmStatus nm_spec_dense(NmSpec *spec, size_t units, NmActivation activation) {
  if (units == 0) {
    set_error("nm_spec_dense: units must be positive");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  SpecLayer *layer = spec_push(spec, NM_LAYER_DENSE);
  if (!layer)
    return NM_STATUS_INVALID_ARGUMENT;
  layer->dense = (NmDenseConfig){.units = units, .activation = activation};
  return NM_STATUS_OK;
}

NmStatus nm_spec_conv2d(NmSpec *spec, size_t filters, size_t kernel,
                        size_t stride, size_t padding, NmActivation activation) {
  if (filters == 0 || kernel == 0 || stride == 0) {
    set_error("nm_spec_conv2d: filters, kernel and stride must be positive");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  SpecLayer *layer = spec_push(spec, NM_LAYER_CONV2D);
  if (!layer)
    return NM_STATUS_INVALID_ARGUMENT;
  layer->conv = (NmConv2dConfig){.filters = filters,
                                 .kernel_size = kernel,
                                 .stride = stride,
                                 .padding = padding,
                                 .algo = NM_CONV_ALGO_AUTO,
                                 .layout = NM_LAYOUT_NCHW,
                                 .activation = activation};
  return NM_STATUS_OK;
}

NmStatus nm_spec_maxpool2d(NmSpec *spec, size_t kernel, size_t stride) {
  if (kernel == 0 || stride == 0) {
    set_error("nm_spec_maxpool2d: kernel and stride must be positive");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  SpecLayer *layer = spec_push(spec, NM_LAYER_MAXPOOL2D);
  if (!layer)
    return NM_STATUS_INVALID_ARGUMENT;
  layer->pool =
      (NmPool2dConfig){.kernel_size = kernel, .stride = stride, .padding = 0};
  return NM_STATUS_OK;
}

NmStatus nm_spec_flatten(NmSpec *spec) {
  return spec_push(spec, NM_LAYER_FLATTEN) ? NM_STATUS_OK
                                           : NM_STATUS_INVALID_ARGUMENT;
}

/* ---- Model ------------------------------------------------------------ */

struct NmNetwork {
  NmFrontendModel *frontend; /* built models; NULL for loaded ones */
  NmGraphModel *imported;    /* loaded models */
  NmGraphModel *graph;       /* whichever of the two is in use */
  NmEngineC *engine;
  NmEngineCConfig config;
  uint64_t rng;
  /* The engine writes scratch state while predicting, so calls on one network
   * take turns. */
  pthread_mutex_t lock;
};

NmBuildOptions nm_build_options_default(void) {
  return (NmBuildOptions){
      .seed = 0, .deterministic = true, .backend = NM_BACKEND_AUTO};
}

NmFitOptions nm_fit_options_default(void) {
  return (NmFitOptions){.loss = NM_LOSS_NONE,
                        .optimizer = NM_OPT_SGD,
                        .learning_rate = 0.1f,
                        .batch_size = 32,
                        .epochs = 10,
                        .shuffle = true,
                        .validation_split = 0.0f};
}

static const char *status_text(NmStatus s) {
  switch (s) {
  case NM_STATUS_OK: return "ok";
  case NM_STATUS_INVALID_ARGUMENT: return "invalid argument";
  case NM_STATUS_IO_ERROR: return "I/O error";
  case NM_STATUS_ALLOC_FAILED: return "out of memory";
  case NM_STATUS_FORMAT_ERROR: return "format error";
  case NM_STATUS_UNSUPPORTED_TYPE: return "unsupported by this backend";
  default: return "internal error";
  }
}

static void model_close(NmNetwork *model) {
  if (!model)
    return;
  pthread_mutex_destroy(&model->lock);
  nm_engine_c_free(model->engine);
  nm_frontend_free(model->frontend);
  nm_graph_model_free(model->imported);
  free(model);
}

void nm_network_free(NmNetwork *model) { model_close(model); }

static NmNetwork *model_alloc(const NmBuildOptions *options) {
  NmBuildOptions opts = options ? *options : nm_build_options_default();
  NmNetwork *model = calloc(1, sizeof(*model));
  if (!model) {
    set_error("out of memory");
    return NULL;
  }
  model->config = nm_engine_c_defaults();
  model->config.backend = opts.backend;
  model->config.batch_parallel.deterministic = opts.deterministic;
  pthread_mutex_init(&model->lock, NULL);
  model->rng = opts.seed ? opts.seed : 0x9E3779B97F4A7C15ULL;
  model->engine = nm_engine_c_create(&model->config);
  if (!model->engine) {
    set_error("out of memory");
    pthread_mutex_destroy(&model->lock);
    free(model);
    return NULL;
  }
  return model;
}

NmNetwork *nm_network_build(const NmSpec *spec, const NmBuildOptions *options) {
  if (!spec) {
    set_error("nm_network_build: spec is NULL");
    return NULL;
  }
  if (spec->count == 0) {
    set_error("nm_network_build: the spec has no layers");
    return NULL;
  }
  NmNetwork *model = model_alloc(options);
  if (!model)
    return NULL;
  if (options && options->seed)
    nm_set_seed(options->seed);

  model->frontend = nm_frontend_create(spec->input);
  if (!model->frontend) {
    set_error("nm_network_build: cannot create the model");
    model_close(model);
    return NULL;
  }
  for (size_t i = 0; i < spec->count; ++i) {
    const SpecLayer *l = &spec->layers[i];
    NmStatus status = NM_STATUS_INTERNAL_ERROR;
    switch (l->type) {
    case NM_LAYER_DENSE:
      status = nm_frontend_add_dense(model->frontend, &l->dense);
      break;
    case NM_LAYER_CONV2D:
      status = nm_frontend_add_conv2d(model->frontend, &l->conv);
      break;
    case NM_LAYER_MAXPOOL2D:
      status = nm_frontend_add_maxpool2d(model->frontend, &l->pool);
      break;
    case NM_LAYER_FLATTEN:
      status = nm_frontend_add_flatten(model->frontend);
      break;
    default:
      break;
    }
    if (status != NM_STATUS_OK) {
      set_error("nm_network_build: layer %zu rejected (%s); check shapes", i + 1,
                status_text(status));
      model_close(model);
      return NULL;
    }
  }
  NmStatus status = nm_frontend_finalize(model->frontend);
  if (status == NM_STATUS_OK)
    status = nm_engine_c_prepare(model->engine, model->frontend);
  if (status != NM_STATUS_OK) {
    set_error("nm_network_build: %s", status_text(status));
    model_close(model);
    return NULL;
  }
  model->graph = nm_frontend_graph(model->frontend);
  return model;
}

/* ---- Training --------------------------------------------------------- */

static uint64_t rng_next(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

static float loss_value(NmLoss loss, const FloatMatrix *pred,
                        const FloatMatrix *target) {
  return loss == NM_LOSS_CROSS_ENTROPY ? nm_loss_cross_entropy(pred, target)
                                       : nm_loss_mse(pred, target);
}

/* Loss of the model on rows [first, first + count) of x and y. */
static NmStatus rows_loss(const NmNetwork *model, NmLoss loss,
                          const FloatMatrix *x, const FloatMatrix *y,
                          size_t first, size_t count, float *out) {
  FloatMatrix xv = {.rows = count, .cols = x->cols,
                    .capacity = count * x->cols,
                    .values = x->values + first * x->cols};
  FloatMatrix yv = {.rows = count, .cols = y->cols,
                    .capacity = count * y->cols,
                    .values = y->values + first * y->cols};
  FloatMatrix *pred = NULL;
  NmStatus status =
      nm_engine_c_predict_graph(model->engine, model->graph, &xv, &pred);
  if (status != NM_STATUS_OK)
    return status;
  if (pred->cols != yv.cols) {
    sm_destroy(pred);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  *out = loss_value(loss, pred, &yv);
  sm_destroy(pred);
  return NM_STATUS_OK;
}

/* Makes sure the engine uses `loss` and the optimizer of `opts`; a changed
 * setting needs a fresh engine (optimizer state restarts). */
static NmStatus configure_engine(NmNetwork *model, NmLoss loss,
                                 const NmFitOptions *opts) {
  if (model->config.loss_type == loss &&
      model->config.optimizer.type == opts->optimizer)
    return NM_STATUS_OK; /* the learning rate travels with the train config */
  NmEngineCConfig wanted = model->config;
  wanted.loss_type = loss;
  wanted.optimizer = nm_engine_c_defaults().optimizer;
  wanted.optimizer.type = opts->optimizer;
  wanted.optimizer.learning_rate = opts->learning_rate;
  NmEngineC *engine = nm_engine_c_create(&wanted);
  if (!engine)
    return NM_STATUS_ALLOC_FAILED;
  NmStatus status = nm_engine_c_prepare_graph(engine, model->graph);
  if (status != NM_STATUS_OK) {
    nm_engine_c_free(engine);
    return status;
  }
  nm_engine_c_free(model->engine);
  model->engine = engine;
  model->config = wanted;
  return NM_STATUS_OK;
}

static NmStatus fit_locked(NmNetwork *model, NmTensor inputs,
                           const NmTensor *targets, const NmFitOptions *options,
                           NmProgressFn progress, void *user) {
  if (!model) {
    set_error("nm_network_fit: model is NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  NmFitOptions opts = options ? *options : nm_fit_options_default();
  if (opts.epochs == 0 || opts.batch_size == 0 || !(opts.learning_rate > 0.0f)) {
    set_error("nm_network_fit: epochs, batch_size and learning_rate must be "
              "positive");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (!(opts.validation_split >= 0.0f && opts.validation_split < 1.0f)) {
    set_error("nm_network_fit: validation_split must be in [0, 1)");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  FloatMatrix x;
  FloatMatrix y;
  if (!tensor_view(&inputs, "inputs", &x))
    return NM_STATUS_INVALID_ARGUMENT;
  if (targets) {
    if (!tensor_view(targets, "targets", &y))
      return NM_STATUS_INVALID_ARGUMENT;
    if (y.rows != x.rows) {
      set_error("nm_network_fit: %zu inputs but %zu targets", x.rows, y.rows);
      return NM_STATUS_INVALID_ARGUMENT;
    }
  } else {
    y = x; /* autoencoder */
  }
  NmLoss loss = opts.loss;
  if (loss == NM_LOSS_NONE)
    loss = targets ? NM_LOSS_CROSS_ENTROPY : NM_LOSS_MSE;

  size_t val_rows = (size_t)((double)x.rows * (double)opts.validation_split);
  size_t train_rows = x.rows - val_rows;
  if (train_rows == 0) {
    set_error("nm_network_fit: no training rows left after validation_split");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (model->graph && nm_graph_bound_version(model->graph) != 0) {
    set_error("nm_network_fit: the weights are bound read-only (connector)");
    return NM_STATUS_INVALID_ARGUMENT;
  }

  NmStatus status = configure_engine(model, loss, &opts);
  if (status != NM_STATUS_OK) {
    set_error("nm_network_fit: cannot set up the engine (%s)", status_text(status));
    return status;
  }

  /* Training rows are copied once so that shuffling never touches the caller's
   * data. Without shuffle the caller's memory is used directly. */
  float *xs = NULL;
  float *ys = NULL;
  size_t *perm = NULL;
  FloatMatrix xt = {.rows = train_rows, .cols = x.cols,
                    .capacity = train_rows * x.cols, .values = x.values};
  FloatMatrix yt = {.rows = train_rows, .cols = y.cols,
                    .capacity = train_rows * y.cols, .values = y.values};
  if (opts.shuffle) {
    xs = malloc(train_rows * x.cols * sizeof(float));
    ys = targets ? malloc(train_rows * y.cols * sizeof(float)) : NULL;
    perm = malloc(train_rows * sizeof(size_t));
    if (!xs || !perm || (targets && !ys)) {
      free(xs);
      free(ys);
      free(perm);
      set_error("nm_network_fit: out of memory");
      return NM_STATUS_ALLOC_FAILED;
    }
    for (size_t i = 0; i < train_rows; ++i)
      perm[i] = i;
    xt.values = xs;
    yt.values = targets ? ys : xs;
  }

  NmTrainConfig train = nm_train_defaults();
  train.batch_size = opts.batch_size;
  train.learning_rate = opts.learning_rate;
  train.epochs = 1;
  train.verbose = false;

  for (size_t epoch = 1; epoch <= opts.epochs && status == NM_STATUS_OK;
       ++epoch) {
    if (opts.shuffle) {
      for (size_t i = train_rows - 1; i > 0; --i) {
        size_t j = (size_t)(rng_next(&model->rng) % (i + 1));
        size_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
      }
      for (size_t i = 0; i < train_rows; ++i) {
        memcpy(xs + i * x.cols, x.values + perm[i] * x.cols,
               x.cols * sizeof(float));
        if (targets)
          memcpy(ys + i * y.cols, y.values + perm[i] * y.cols,
                 y.cols * sizeof(float));
      }
    }
    status = nm_engine_c_fit_graph(model->engine, model->graph, &xt, &yt, &train);
    if (status != NM_STATUS_OK) {
      set_error("nm_network_fit: training failed in epoch %zu (%s)", epoch,
                status_text(status));
      break;
    }
    if (progress) {
      NmProgress p = {.epoch = epoch, .epochs = opts.epochs,
                      .loss = NAN, .val_loss = NAN};
      status = rows_loss(model, loss, &xt, &yt, 0, train_rows, &p.loss);
      if (status == NM_STATUS_OK && val_rows > 0)
        status = rows_loss(model, loss, &x, &y, train_rows, val_rows,
                           &p.val_loss);
      if (status != NM_STATUS_OK) {
        set_error("nm_network_fit: cannot compute the loss (%s)",
                  status_text(status));
        break;
      }
      if (progress(&p, user) != 0)
        break;
    }
  }
  free(xs);
  free(ys);
  free(perm);
  return status;
}

NmStatus nm_network_fit(NmNetwork *model, NmTensor inputs,
                        const NmTensor *targets, const NmFitOptions *options,
                        NmProgressFn progress, void *user) {
  if (!model) {
    set_error("nm_network_fit: model is NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  pthread_mutex_lock(&model->lock);
  NmStatus status = fit_locked(model, inputs, targets, options, progress, user);
  pthread_mutex_unlock(&model->lock);
  return status;
}

/* ---- Using ------------------------------------------------------------ */

/* Copies a prediction into a tensor the caller owns. */
static NmStatus prediction_tensor(FloatMatrix *pred, NmTensor *out,
                                  const char *what) {
  size_t count = pred->rows * pred->cols;
  float *data = malloc((count ? count : 1) * sizeof(float));
  if (!data) {
    sm_destroy(pred);
    set_error("%s: out of memory", what);
    return NM_STATUS_ALLOC_FAILED;
  }
  memcpy(data, pred->values, count * sizeof(float));
  *out = nm_tensor_2d(data, pred->rows, pred->cols);
  sm_destroy(pred);
  return NM_STATUS_OK;
}

NmStatus nm_network_predict(const NmNetwork *model, NmTensor inputs, NmTensor *out) {
  if (!model || !out) {
    set_error("nm_network_predict: model and out must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  *out = (NmTensor){0};
  FloatMatrix x;
  if (!tensor_view(&inputs, "inputs", &x))
    return NM_STATUS_INVALID_ARGUMENT;
  FloatMatrix *pred = NULL;
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status =
      nm_engine_c_predict_graph(model->engine, model->graph, &x, &pred);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK) {
    if (nm_network_input_count(model) > 1)
      set_error("nm_network_predict: the model has %zu inputs; use "
                "nm_network_predict_inputs",
                nm_network_input_count(model));
    else
      set_error("nm_network_predict: %s (example size %zu matches the model input?)",
                status_text(status), x.cols);
    return status;
  }
  return prediction_tensor(pred, out, "nm_network_predict");
}

NmStatus nm_network_predict_inputs(const NmNetwork *model, const NmInput *inputs,
                                   size_t count, NmTensor *out) {
  if (!model || !out || (!inputs && count)) {
    set_error("nm_network_predict_inputs: model, inputs and out must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  *out = (NmTensor){0};
  FloatMatrix *pred = NULL;
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status = nm_engine_c_predict_inputs(model->engine, model->graph,
                                               inputs, count, &pred);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK) {
    set_error("nm_network_predict_inputs: %s (do %zu inputs match the model's "
              "%zu by name, type and shape?)",
              status_text(status), count, nm_network_input_count(model));
    return status;
  }
  return prediction_tensor(pred, out, "nm_network_predict_inputs");
}

NmStatus nm_network_predict_batches(const NmNetwork *model, const NmInput *inputs,
                                    size_t count, NmTensor *out) {
  if (!model || !out || (!inputs && count)) {
    set_error("nm_network_predict_batches: model, inputs and out must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  *out = (NmTensor){0};
  FloatMatrix *pred = NULL;
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status = nm_engine_c_predict_batches(model->engine, model->graph,
                                                inputs, count, &pred);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK) {
    set_error("nm_network_predict_batches: %s (inputs must match the model's by "
              "name, type and shape apart from the number of examples)",
              status_text(status));
    return status;
  }
  return prediction_tensor(pred, out, "nm_network_predict_batches");
}

size_t nm_network_input_count(const NmNetwork *model) {
  if (!model || !model->graph)
    return 0;
  return model->graph->input_count ? model->graph->input_count : 1;
}

NmStatus nm_network_input_info(const NmNetwork *model, size_t index,
                               NmInput *out) {
  if (!model || !out || index >= nm_network_input_count(model)) {
    set_error("nm_network_input_info: no input %zu", index);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  const NmIrGraph *graph = model->graph;
  uint32_t id = graph->input_count ? graph->input_ids[index] : graph->model_input_id;
  const NmIrValue *value = &graph->values[id];
  *out = (NmInput){.name = value->name, .data = NULL, .ndim = value->shape.ndim};
  out->dtype = value->dtype == NM_IR_DTYPE_I64    ? NM_DTYPE_INT64
               : value->dtype == NM_IR_DTYPE_BOOL ? NM_DTYPE_BOOL
                                                  : NM_DTYPE_FLOAT32;
  memcpy(out->dims, value->shape.dims, value->shape.ndim * sizeof(size_t));
  return NM_STATUS_OK;
}

NmStatus nm_network_evaluate(const NmNetwork *model, NmTensor inputs,
                           NmTensor targets, NmLoss loss, NmMetrics *out) {
  if (!model || !out) {
    set_error("nm_network_evaluate: model and out must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (loss != NM_LOSS_MSE && loss != NM_LOSS_CROSS_ENTROPY) {
    set_error("nm_network_evaluate: choose NM_LOSS_MSE or NM_LOSS_CROSS_ENTROPY");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  FloatMatrix x;
  FloatMatrix y;
  if (!tensor_view(&inputs, "inputs", &x) ||
      !tensor_view(&targets, "targets", &y))
    return NM_STATUS_INVALID_ARGUMENT;
  if (x.rows != y.rows) {
    set_error("nm_network_evaluate: %zu inputs but %zu targets", x.rows, y.rows);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  FloatMatrix *pred = NULL;
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status =
      nm_engine_c_predict_graph(model->engine, model->graph, &x, &pred);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK) {
    set_error("nm_network_evaluate: %s", status_text(status));
    return status;
  }
  if (pred->cols != y.cols) {
    set_error("nm_network_evaluate: model outputs %zu values, targets have %zu",
              pred->cols, y.cols);
    sm_destroy(pred);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  out->loss = loss_value(loss, pred, &y);
  out->accuracy = -1.0f;
  if (loss == NM_LOSS_CROSS_ENTROPY)
    status = nm_accuracy(pred, &y, &out->accuracy);
  sm_destroy(pred);
  return status;
}

/* ---- Saving ----------------------------------------------------------- */

NmStatus nm_network_export_graphir(const NmNetwork *model, const char *path) {
  if (!model || !path) {
    set_error("nm_network_export_graphir: model and path must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  NmGraphIrExportOptions options = {.batch_size = 1};
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status = nm_export_graphir_graph_json(model->graph, &options, path);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK)
    set_error("cannot write %s (%s)", path, status_text(status));
  return status;
}

static bool file_in(char *out, size_t size, const char *dir, const char *name) {
  int n = snprintf(out, size, "%s/%s", dir, name);
  if (n < 0 || (size_t)n >= size) {
    set_error("path too long: %s", dir);
    return false;
  }
  return true;
}

static bool model_path(char *out, size_t size, const char *dir) {
  return file_in(out, size, dir, "model.graphir.json");
}

NmStatus nm_network_save(const NmNetwork *model, const char *dir) {
  if (!model || !dir) {
    set_error("nm_network_save: model and dir must not be NULL");
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
    set_error("nm_network_save: cannot create %s: %s", dir, strerror(errno));
    return NM_STATUS_IO_ERROR;
  }
  char path[1024];
  char weights[1024];
  if (!model_path(path, sizeof(path), dir) ||
      !file_in(weights, sizeof(weights), dir, "weights.girw"))
    return NM_STATUS_INVALID_ARGUMENT;
  NmGraphIrExportOptions options = {.batch_size = 1, .weights_path = weights};
  pthread_mutex_lock((pthread_mutex_t *)&model->lock);
  NmStatus status = nm_export_graphir_graph_json(model->graph, &options, path);
  pthread_mutex_unlock((pthread_mutex_t *)&model->lock);
  if (status != NM_STATUS_OK)
    set_error("cannot write %s (%s)", path, status_text(status));
  return status;
}

NmNetwork *nm_network_import_graphir(const char *path) {
  if (!path) {
    set_error("nm_network_import_graphir: path is NULL");
    return NULL;
  }
  NmNetwork *model = model_alloc(NULL);
  if (!model)
    return NULL;
  NmStatus status = nm_engine_c_prepare_graphir_json(model->engine, path,
                                                     &model->imported);
  if (status != NM_STATUS_OK) {
    set_error("cannot load %s (%s)", path, status_text(status));
    model_close(model);
    return NULL;
  }
  model->graph = model->imported;
  return model;
}

NmNetwork *nm_network_load(const char *dir) {
  if (!dir) {
    set_error("nm_network_load: dir is NULL");
    return NULL;
  }
  char path[1024];
  if (!model_path(path, sizeof(path), dir))
    return NULL;
  return nm_network_import_graphir(path);
}

/* ---- Inspecting ------------------------------------------------------- */

size_t nm_network_parameter_count(const NmNetwork *network) {
  if (!network || !network->graph)
    return 0;
  size_t total = 0;
  for (size_t i = 0; i < network->graph->value_count; ++i) {
    const NmIrValue *v = &network->graph->values[i];
    if (v->kind == NM_IR_VALUE_CONSTANT && !v->raw) /* not int64/bool data */
      total += v->data_count;
  }
  return total;
}

typedef struct {
  char *buffer;
  size_t size;
  size_t length; /* what the full text needs */
} Out;

static void out_printf(Out *out, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  size_t room = out->size > out->length ? out->size - out->length : 0;
  int n = vsnprintf(out->buffer ? out->buffer + out->length : NULL, room, fmt,
                    args);
  va_end(args);
  if (n > 0)
    out->length += (size_t)n;
}

static const char *activation_name(NmIrOpType op) {
  switch (op) {
  case NM_IR_OP_RELU: return "ReLU";
  case NM_IR_OP_SIGMOID: return "Sigmoid";
  case NM_IR_OP_TANH: return "Tanh";
  case NM_IR_OP_SOFTMAX: return "Softmax";
  default: return NULL;
  }
}

static void shape_text(char *text, size_t size, const NmIrShape *shape) {
  size_t used = 0;
  text[0] = '\0';
  used += (size_t)snprintf(text, size, "[");
  for (size_t d = 1; d < shape->ndim && used < size; ++d)
    used += (size_t)snprintf(text + used, size - used, d > 1 ? ", %zu" : "%zu",
                             shape->dims[d]);
  if (used < size)
    snprintf(text + used, size - used, "]");
}

size_t nm_network_summary(const NmNetwork *network, char *buffer, size_t size) {
  Out out = {.buffer = buffer, .size = size};
  if (buffer && size)
    buffer[0] = '\0';
  if (!network || !network->graph)
    return 0;
  const NmIrGraph *g = network->graph;
  out_printf(&out, "%-24s %-16s %10s\n", "Layer", "Output", "Params");

  for (size_t i = 0; i < g->node_count; ++i) {
    const NmIrNode *n = &g->nodes[i];
    const char *label = NULL;
    size_t params = 0;
    uint32_t out_id = n->output_id;
    char name[48];

    switch ((NmIrOpType)n->op_type) {
    case NM_IR_OP_MATMUL: {
      const NmIrValue *w = &g->values[n->input_ids[1]];
      params = w->data_count;
      snprintf(name, sizeof(name), "Dense %zu", w->shape.dims[w->shape.ndim - 1]);
      label = name;
      if (i + 1 < g->node_count && g->nodes[i + 1].op_type == NM_IR_OP_ADD) {
        ++i; /* the bias belongs to the dense layer */
        params += g->values[g->nodes[i].input_ids[1]].data_count;
        out_id = g->nodes[i].output_id;
      }
      break;
    }
    case NM_IR_OP_CONV2D: {
      const NmIrValue *w = &g->values[n->input_ids[1]];
      for (uint32_t k = 1; k < n->input_count; ++k)
        params += g->values[n->input_ids[k]].data_count;
      snprintf(name, sizeof(name), "Conv2D %zu@%zux%zu", w->shape.dims[0],
               w->shape.dims[2], w->shape.dims[3]);
      label = name;
      break;
    }
    case NM_IR_OP_MAXPOOL2D:
      snprintf(name, sizeof(name), "MaxPool2D %zux%zu", n->cfg.maxpool2d.kernel_h,
               n->cfg.maxpool2d.kernel_w);
      label = name;
      break;
    case NM_IR_OP_RESHAPE:
      label = g->values[out_id].shape.ndim == 2 ? "Flatten" : "Reshape";
      break;
    case NM_IR_OP_LAYERNORM:
      for (uint32_t k = 1; k < n->input_count; ++k)
        params += g->values[n->input_ids[k]].data_count;
      label = "LayerNorm";
      break;
    case NM_IR_OP_ADD:
    case NM_IR_OP_SUB:
    case NM_IR_OP_MUL:
    case NM_IR_OP_DIV:
    case NM_IR_OP_TRANSPOSE:
    case NM_IR_OP_GELU:
    case NM_IR_OP_EQUAL:
    case NM_IR_OP_NOT:
    case NM_IR_OP_WHERE:
    case NM_IR_OP_CAST:
    case NM_IR_OP_EXPAND:
    case NM_IR_OP_SLICE:
    case NM_IR_OP_REDUCEMAX:
    case NM_IR_OP_RANGE:
    case NM_IR_OP_SQUEEZE:
    case NM_IR_OP_UNSQUEEZE:
      label = nm_ir_op_name((NmIrOpType)n->op_type);
      break;
    case NM_IR_OP_GATHER: /* an embedding table counts as parameters */
      if (g->values[n->input_ids[0]].kind == NM_IR_VALUE_CONSTANT)
        params = g->values[n->input_ids[0]].data_count;
      label = "Gather";
      break;
    default:
      break; /* inputs, constants */
    }
    if (!label)
      continue;

    /* A following activation is part of the same row. */
    char full[80];
    if (i + 1 < g->node_count) {
      const char *act = activation_name((NmIrOpType)g->nodes[i + 1].op_type);
      if (act) {
        snprintf(full, sizeof(full), "%s + %s", label, act);
        label = full;
        out_id = g->nodes[++i].output_id;
      }
    }
    char shape[48];
    shape_text(shape, sizeof(shape), &g->values[out_id].shape);
    out_printf(&out, "%-24s %-16s %10zu\n", label, shape, params);
  }
  out_printf(&out, "Total parameters: %zu\n", nm_network_parameter_count(network));
  return out.length;
}
