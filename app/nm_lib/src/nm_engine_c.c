/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_engine_c.h"
#include "nm_frontend_internal.h"
#include "nm_model_internal.h"
#include <stdlib.h>
#include <string.h>

struct NmEngineC {
  NmEngineCConfig config;
  const NmBackendOps *backend_ops;
  NmExecPlan exec_plan;
  NmLayerRuntime runtime;
  NmInferencePlan inference_plan;
  NmTrainingState training;
  const NmIrGraph *prepared_graph;
  const NmFrontendModel *prepared_model; /* NULL for imported graphs */
  /* Set when the graph runs on the interpreter instead of the execution plan
   * (inference only). */
  bool interpreted;
  size_t interp_input_cols;
  size_t interp_output_cols;
  size_t interp_fixed_batch; /* 0: any batch size */
  NmInterp *interp;
};

NmEngineCConfig nm_engine_c_defaults(void) {
  return (NmEngineCConfig){
      .loss_type = NM_LOSS_CROSS_ENTROPY,
      .optimizer = nm_seq_optimizer_defaults(),
      .backend = NM_BACKEND_AUTO,
      .autograd_mode = NM_AUTOGRAD_DISABLED,
      .batch_parallel = {.enabled = false, .deterministic = true,
                         .num_workers = 0},
  };
}

NmEngineC *nm_engine_c_create(const NmEngineCConfig *config) {
  NmEngineC *engine = calloc(1, sizeof(*engine));
  if (!engine)
    return NULL;
  engine->config = config ? *config : nm_engine_c_defaults();
  engine->backend_ops = &NM_BACKEND_DEFAULT_OPS;
  exec_plan_init(&engine->exec_plan);
  if (!inference_plan_init(&engine->inference_plan)) {
    free(engine);
    return NULL;
  }
  exec_trace_init(&engine->training.last_trace);
  engine->training.optimizer_config = engine->config.optimizer;
  engine->training.autograd_mode = engine->config.autograd_mode;
  engine->training.batch_parallel = engine->config.batch_parallel;
  return engine;
}

void nm_engine_c_free(NmEngineC *engine) {
  if (!engine)
    return;
  graph_interp_free(engine->interp);
  layer_runtime_destroy(&engine->runtime);
  exec_plan_destroy(&engine->exec_plan);
  inference_plan_destroy(&engine->inference_plan);
  exec_trace_destroy(&engine->training.last_trace);
  free(engine);
}

static NmStatus prepare_graph(NmEngineC *engine, const NmIrGraph *graph) {
  switch (engine->config.backend) {
  case NM_BACKEND_AUTO:
  case NM_BACKEND_DEFAULT:
    engine->backend_ops = &NM_BACKEND_DEFAULT_OPS;
    break;
  default:
    return NM_STATUS_UNSUPPORTED_TYPE;
  }

  engine->prepared_graph = NULL;
  engine->prepared_model = NULL;
  engine->interpreted = false;
  graph_interp_free(engine->interp);
  engine->interp = NULL;
  layer_runtime_destroy(&engine->runtime);
  exec_plan_reset(&engine->exec_plan);
  /* Layer chains read one float input; graphs with several inputs or with
   * integer and boolean values go to the interpreter directly. */
  bool plain = graph->input_count <= 1;
  for (size_t i = 0; plain && i < graph->value_count; ++i)
    plain = nm_ir_dtype_is_float(graph->values[i].dtype);
  NmStatus status = plain ? exec_plan_build(&engine->exec_plan, graph)
                          : NM_STATUS_UNSUPPORTED_TYPE;
  if (status == NM_STATUS_UNSUPPORTED_TYPE || status == NM_STATUS_FORMAT_ERROR) {
    /* Not a layer chain: run it on the interpreter if it can. */
    NmStatus interp = graph_interp_prepare(graph, &engine->interp,
                                           &engine->interp_input_cols,
                                           &engine->interp_output_cols,
                                           &engine->interp_fixed_batch);
    if (interp != NM_STATUS_OK)
      return status;
    engine->interpreted = true;
    engine->prepared_graph = graph;
    return NM_STATUS_OK;
  }
  if (status != NM_STATUS_OK)
    return status;
  status = layer_runtime_build(&engine->runtime, graph, &engine->exec_plan);
  if (status != NM_STATUS_OK) {
    exec_plan_reset(&engine->exec_plan);
    return status;
  }
  if (!inference_plan_reinit(&engine->inference_plan))
    return NM_STATUS_ALLOC_FAILED;
  engine->training.optimizer_config = engine->config.optimizer;
  engine->training.optimizer_step = 0;
  engine->training.autograd_mode = engine->config.autograd_mode;
  engine->training.batch_parallel = engine->config.batch_parallel;
  exec_trace_reset(&engine->training.last_trace);
  engine->prepared_graph = graph;
  return NM_STATUS_OK;
}

NmStatus nm_engine_c_prepare(NmEngineC *engine, NmFrontendModel *model) {
  if (!engine || !model)
    return NM_STATUS_INVALID_ARGUMENT;
  NmGraphModel *graph = nm_frontend_graph(model);
  if (!graph)
    return NM_STATUS_INVALID_ARGUMENT; /* not finalized */
  NmStatus status = prepare_graph(engine, graph);
  if (status == NM_STATUS_OK)
    engine->prepared_model = model;
  return status;
}

NmStatus nm_engine_c_prepare_graph(NmEngineC *engine,
                                   const NmGraphModel *graph) {
  if (!engine || !graph)
    return NM_STATUS_INVALID_ARGUMENT;
  return prepare_graph(engine, graph);
}

NmStatus nm_engine_c_prepare_graphir_json(NmEngineC *engine, const char *path,
                                          NmGraphModel **out_graph) {
  if (!engine || !path || !out_graph)
    return NM_STATUS_INVALID_ARGUMENT;
  NmStatus status = nm_import_graphir_json(path, out_graph);
  if (status != NM_STATUS_OK)
    return status;
  status = prepare_graph(engine, *out_graph);
  if (status != NM_STATUS_OK) {
    nm_graph_model_free(*out_graph);
    *out_graph = NULL;
  }
  return status;
}

static bool prepared_for(const NmEngineC *engine, const NmFrontendModel *model) {
  return engine && model && engine->prepared_model == model;
}

NmStatus nm_engine_c_bind_weights(NmEngineC *engine, NmGraphModel *graph,
                                  NmWeightSet *set) {
  if (!engine || !graph || engine->prepared_graph != graph)
    return NM_STATUS_INVALID_ARGUMENT;
  NmStatus status = nm_graph_bind_weights(graph, set);
  if (status != NM_STATUS_OK)
    return status;
  if (engine->interpreted) {
    /* Values folded from the old weights must be computed again. */
    NmInterp *interp = NULL;
    size_t input_cols, output_cols, fixed_batch;
    status = graph_interp_prepare(graph, &interp, &input_cols, &output_cols,
                                  &fixed_batch);
    graph_interp_free(engine->interp);
    engine->interp = interp;
    if (status != NM_STATUS_OK)
      engine->prepared_graph = NULL; /* needs a new prepare */
    return status;
  }
  /* The layer entries hold weight pointers; rebuild them for the new memory. */
  layer_runtime_destroy(&engine->runtime);
  status = layer_runtime_build(&engine->runtime, graph, &engine->exec_plan);
  if (status != NM_STATUS_OK)
    engine->prepared_graph = NULL; /* needs a new prepare */
  return status;
}

static NmStatus fit_prepared(NmEngineC *engine, const FloatMatrix *inputs,
                             const FloatMatrix *targets,
                             const NmTrainConfig *config) {
  if (!inputs || !targets)
    return NM_STATUS_INVALID_ARGUMENT;
  if (engine->interpreted)
    return NM_STATUS_UNSUPPORTED_TYPE; /* the interpreter has no backward pass */
  if (engine->prepared_graph->binding)
    return NM_STATUS_INVALID_ARGUMENT; /* weights are borrowed and read-only */
  for (size_t i = 0; i < engine->prepared_graph->value_count; ++i) {
    /* 16-bit graphs run promoted; training them needs real 16-bit kernels. */
    if (engine->prepared_graph->values[i].dtype != NM_IR_DTYPE_F32)
      return NM_STATUS_UNSUPPORTED_TYPE;
  }
  for (size_t i = 0; i < engine->exec_plan.count; ++i) {
    if (engine->exec_plan.steps[i].kind == NM_EXEC_STEP_ACTIVATION)
      return NM_STATUS_UNSUPPORTED_TYPE; /* no backward kernel */
  }
  NmTrainable trainable = {.layers = engine->runtime.layers,
                           .num_layers = engine->runtime.count,
                           .input_shape = engine->exec_plan.input_shape,
                           .loss_type = engine->config.loss_type,
                           .training = &engine->training,
                           .backend = engine->backend_ops};
  return nm_train_fit(&trainable, inputs, targets, config);
}

NmStatus nm_engine_c_fit(NmEngineC *engine, NmFrontendModel *model,
                         const FloatMatrix *inputs, const FloatMatrix *targets,
                         const NmTrainConfig *config) {
  if (!prepared_for(engine, model))
    return NM_STATUS_INVALID_ARGUMENT;
  return fit_prepared(engine, inputs, targets, config);
}

NmStatus nm_engine_c_fit_graph(NmEngineC *engine, NmGraphModel *graph,
                               const FloatMatrix *inputs,
                               const FloatMatrix *targets,
                               const NmTrainConfig *config) {
  if (!engine || !graph || engine->prepared_graph != graph)
    return NM_STATUS_INVALID_ARGUMENT;
  return fit_prepared(engine, inputs, targets, config);
}

static NmStatus predict_prepared(const NmEngineC *engine,
                                 const FloatMatrix *inputs,
                                 FloatMatrix **out_prediction) {
  if (engine->interpreted) {
    /* One float input; the rows are the batch. */
    const NmIrGraph *graph = engine->prepared_graph;
    if (!inputs || !out_prediction || inputs->rows == 0 ||
        inputs->cols != engine->interp_input_cols)
      return NM_STATUS_INVALID_ARGUMENT;
    const NmIrShape *shape = &graph->values[graph->model_input_id].shape;
    NmInput input = {.dtype = NM_DTYPE_FLOAT32, .data = inputs->values,
                     .ndim = shape->ndim};
    if (shape->ndim < 2)
      return NM_STATUS_INVALID_ARGUMENT;
    memcpy(input.dims, shape->dims, shape->ndim * sizeof(size_t));
    input.dims[0] = inputs->rows;
    if (engine->interp_fixed_batch && inputs->rows != engine->interp_fixed_batch)
      return NM_STATUS_INVALID_ARGUMENT;
    NmInput ordered[NM_IR_MAX_INPUTS];
    NmStatus status = graph_interp_bind(graph, &input, 1, ordered);
    if (status == NM_STATUS_OK)
      status = graph_interp_run(engine->interp, ordered, out_prediction);
    return status;
  }
  if (!inputs || !out_prediction || engine->exec_plan.output_cols == 0)
    return NM_STATUS_INVALID_ARGUMENT;

  *out_prediction = sm_create(inputs->rows, engine->exec_plan.output_cols);
  if (!*out_prediction)
    return NM_STATUS_ALLOC_FAILED;

  NmEngineC *mutable_engine = (NmEngineC *)engine;
  NmStatus status = nm_run_graph_plan_into(
      &engine->runtime, &mutable_engine->inference_plan,
      &engine->exec_plan, engine->backend_ops, inputs, *out_prediction);
  if (status != NM_STATUS_OK) {
    sm_destroy(*out_prediction);
    *out_prediction = NULL;
  }
  return status;
}

NmStatus nm_engine_c_predict(const NmEngineC *engine,
                             const NmFrontendModel *model,
                             const FloatMatrix *inputs,
                             FloatMatrix **out_prediction) {
  if (!prepared_for(engine, model))
    return NM_STATUS_INVALID_ARGUMENT;
  return predict_prepared(engine, inputs, out_prediction);
}

NmStatus nm_engine_c_predict_graph(const NmEngineC *engine,
                                   const NmGraphModel *graph,
                                   const FloatMatrix *inputs,
                                   FloatMatrix **out_prediction) {
  if (!engine || !graph || engine->prepared_graph != graph)
    return NM_STATUS_INVALID_ARGUMENT;
  return predict_prepared(engine, inputs, out_prediction);
}

NmStatus nm_engine_c_predict_inputs(const NmEngineC *engine,
                                    const NmGraphModel *graph,
                                    const NmInput *inputs, size_t count,
                                    FloatMatrix **out_prediction) {
  if (!engine || !graph || engine->prepared_graph != graph || !out_prediction)
    return NM_STATUS_INVALID_ARGUMENT;
  NmInput ordered[NM_IR_MAX_INPUTS];
  NmStatus status = graph_interp_bind(graph, inputs, count, ordered);
  if (status != NM_STATUS_OK)
    return status;
  if (engine->interpreted) {
    if (engine->interp_fixed_batch && ordered[0].ndim > 0 &&
        ordered[0].dims[0] != engine->interp_fixed_batch)
      return NM_STATUS_INVALID_ARGUMENT;
    return graph_interp_run(engine->interp, ordered, out_prediction);
  }
  /* A layer chain: its single float input as one row per example. */
  size_t numel = 1;
  for (size_t i = 0; i < ordered[0].ndim; ++i)
    numel *= ordered[0].dims[i];
  if (ordered[0].ndim < 2)
    return NM_STATUS_INVALID_ARGUMENT;
  FloatMatrix x = {.rows = ordered[0].dims[0],
                   .cols = numel / ordered[0].dims[0],
                   .capacity = numel,
                   .values = (float *)ordered[0].data};
  return predict_prepared(engine, &x, out_prediction);
}

NmStatus nm_engine_c_predict_batches(const NmEngineC *engine,
                                     const NmGraphModel *graph,
                                     const NmInput *inputs, size_t count,
                                     FloatMatrix **out_prediction) {
  if (!engine || !graph || engine->prepared_graph != graph || !out_prediction ||
      (!inputs && count) || count == 0 || count > NM_IR_MAX_INPUTS)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_prediction = NULL;
  /* The declared batch: the first dimension every input shares. */
  NmInput declared[NM_IR_MAX_INPUTS];
  const size_t inputs_count = graph->input_count ? graph->input_count : 1;
  if (count != inputs_count)
    return NM_STATUS_INVALID_ARGUMENT;
  size_t batch = 0, examples = 0;
  for (size_t i = 0; i < count; ++i) {
    /* The graph input this one feeds: by name, else by position. */
    size_t slot = i;
    for (size_t j = 0; inputs[i].name && j < count; ++j) {
      uint32_t id = graph->input_count ? graph->input_ids[j] : graph->model_input_id;
      if (!strcmp(graph->values[id].name, inputs[i].name))
        slot = j;
    }
    uint32_t id = graph->input_count ? graph->input_ids[slot] : graph->model_input_id;
    const NmIrShape *shape = &graph->values[id].shape;
    if (shape->ndim == 0 || inputs[i].ndim == 0)
      return NM_STATUS_INVALID_ARGUMENT;
    if (shape->dims[0] == 0) /* the batch is open: one call does it */
      return nm_engine_c_predict_inputs(engine, graph, inputs, count,
                                        out_prediction);
    if ((batch && shape->dims[0] != batch) ||
        (examples && inputs[i].dims[0] != examples))
      return NM_STATUS_INVALID_ARGUMENT;
    batch = shape->dims[0];
    examples = inputs[i].dims[0];
    declared[i] = inputs[i];
  }
  if (examples == 0)
    return NM_STATUS_INVALID_ARGUMENT;
  if (examples == batch)
    return nm_engine_c_predict_inputs(engine, graph, inputs, count,
                                      out_prediction);

  /* Bytes of one example per input. */
  size_t example_bytes[NM_IR_MAX_INPUTS];
  for (size_t i = 0; i < count; ++i) {
    size_t elems = 1;
    for (size_t d = 1; d < inputs[i].ndim; ++d)
      elems *= inputs[i].dims[d];
    example_bytes[i] = elems * (inputs[i].dtype == NM_DTYPE_INT64  ? sizeof(int64_t)
                                : inputs[i].dtype == NM_DTYPE_BOOL ? 1
                                                                   : sizeof(float));
  }
  void *padded[NM_IR_MAX_INPUTS] = {NULL};
  FloatMatrix *result = NULL;
  NmStatus status = NM_STATUS_OK;
  size_t rows_per_example = 0;
  for (size_t first = 0; first < examples && status == NM_STATUS_OK; first += batch) {
    const size_t take = examples - first < batch ? examples - first : batch;
    for (size_t i = 0; i < count; ++i) {
      const uint8_t *from = (const uint8_t *)inputs[i].data + first * example_bytes[i];
      declared[i].dims[0] = batch;
      if (take == batch) {
        declared[i].data = from;
        continue;
      }
      /* The last, partial chunk: repeat its final example. */
      padded[i] = malloc(batch * example_bytes[i]);
      if (!padded[i]) {
        status = NM_STATUS_ALLOC_FAILED;
        break;
      }
      memcpy(padded[i], from, take * example_bytes[i]);
      for (size_t k = take; k < batch; ++k)
        memcpy((uint8_t *)padded[i] + k * example_bytes[i],
               from + (take - 1) * example_bytes[i], example_bytes[i]);
      declared[i].data = padded[i];
    }
    FloatMatrix *chunk = NULL;
    if (status == NM_STATUS_OK)
      status = nm_engine_c_predict_inputs(engine, graph, declared, count, &chunk);
    if (status == NM_STATUS_OK) {
      /* The output must be batch-major: its rows split evenly by example. */
      if (chunk->rows % batch != 0) {
        status = NM_STATUS_UNSUPPORTED_TYPE;
      } else {
        rows_per_example = chunk->rows / batch;
        if (!result)
          result = sm_create(examples * rows_per_example, chunk->cols);
        if (!result)
          status = NM_STATUS_ALLOC_FAILED;
        else
          memcpy(result->values + first * rows_per_example * chunk->cols,
                 chunk->values, take * rows_per_example * chunk->cols * sizeof(float));
      }
    }
    if (chunk)
      sm_destroy(chunk);
  }
  for (size_t i = 0; i < count; ++i)
    free(padded[i]);
  if (status != NM_STATUS_OK) {
    if (result)
      sm_destroy(result);
    return status;
  }
  *out_prediction = result;
  return NM_STATUS_OK;
}

NmStatus nm_engine_c_evaluate(const NmEngineC *engine,
                              const NmFrontendModel *model,
                              const FloatMatrix *inputs,
                              const FloatMatrix *targets,
                              float *out_accuracy) {
  if (!prepared_for(engine, model) || !inputs || !targets || !out_accuracy)
    return NM_STATUS_INVALID_ARGUMENT;

  FloatMatrix *pred = NULL;
  NmStatus status = nm_engine_c_predict(engine, model, inputs, &pred);
  if (status != NM_STATUS_OK)
    return status;

  status = nm_accuracy(pred, targets, out_accuracy);
  sm_destroy(pred);
  return status;
}

const NmBackendCapabilities *nm_engine_c_capabilities(const NmEngineC *engine) {
  (void)engine;
  return nm_backend_capabilities();
}
