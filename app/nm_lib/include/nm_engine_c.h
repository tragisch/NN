/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Optional local training and inference engine. */
#ifndef NM_ENGINE_C_H
#define NM_ENGINE_C_H

#include "nm_frontend.h"
#include "nm_import_graphir.h"
#include "nm_weightset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NmEngineC NmEngineC;

typedef struct {
  NmLoss loss_type;
  NmOptimizerConfig optimizer;
  NmBackendType backend;
  NmAutogradMode autograd_mode;
  NmBatchParallelConfig batch_parallel;
} NmEngineCConfig;

NmEngineCConfig nm_engine_c_defaults(void);
NmEngineC *nm_engine_c_create(const NmEngineCConfig *config);
void nm_engine_c_free(NmEngineC *engine);

/*
 * Builds or refreshes the local execution plan for model.
 * Lifetime: `model` must remain valid for any subsequent fit/predict/evaluate
 * calls on `engine`.
 */
NmStatus nm_engine_c_prepare(NmEngineC *engine, NmFrontendModel *model);
/* Prepares the engine for a GraphIR graph, for example an imported one. The
 * graph must outlive the engine's use of it. */
NmStatus nm_engine_c_prepare_graph(NmEngineC *engine, const NmGraphModel *graph);

/* Imports a GraphIR JSON document and prepares the engine for it in one step.
 * On success the caller owns *out_graph (nm_graph_model_free). */
NmStatus nm_engine_c_prepare_graphir_json(NmEngineC *engine, const char *path,
                                          NmGraphModel **out_graph);

NmStatus nm_engine_c_fit(NmEngineC *engine, NmFrontendModel *model,
                         const FloatMatrix *inputs, const FloatMatrix *targets,
                         const NmTrainConfig *config);
/* Switches a prepared graph to the weights of `set` (see nm_weightset.h) without
 * copying them. Call it between two predicts; the engine keeps the graph and
 * only re-reads the weight pointers. The graph's weights are read-only
 * afterwards, so fit is rejected. On failure the previous weights stay. */
NmStatus nm_engine_c_bind_weights(NmEngineC *engine, NmGraphModel *graph,
                                  NmWeightSet *set);

/* Trains the graph in place: its weights are updated and stay with the graph.
 * Works for any prepared graph made of Dense, Conv2D, MaxPool2D and Reshape
 * layers, including graphs imported from GraphIR. */
NmStatus nm_engine_c_fit_graph(NmEngineC *engine, NmGraphModel *graph,
                               const FloatMatrix *inputs,
                               const FloatMatrix *targets,
                               const NmTrainConfig *config);

NmStatus nm_engine_c_predict(const NmEngineC *engine,
                             const NmFrontendModel *model,
                             const FloatMatrix *inputs,
                             FloatMatrix **out_prediction);
NmStatus nm_engine_c_predict_graph(const NmEngineC *engine,
                                   const NmGraphModel *graph,
                                   const FloatMatrix *inputs,
                                   FloatMatrix **out_prediction);

/* Predicts from one or more named inputs of any supported type (float32,
 * int64, bool), for example token ids and a padding mask. Inputs are matched
 * by name, or by position where the name is NULL; their shapes must match the
 * graph's, except for a batch dimension the graph leaves open. The output is
 * converted to float, one row per entry of its first dimension. */
NmStatus nm_engine_c_predict_inputs(const NmEngineC *engine,
                                    const NmGraphModel *graph,
                                    const NmInput *inputs, size_t count,
                                    FloatMatrix **out_prediction);

/* Like nm_engine_c_predict_inputs for any number of examples along the first
 * dimension of every input, for graphs whose declared shapes fix the batch
 * (graphs with several inputs, say): the examples run in chunks of the
 * declared batch, the last chunk padded with copies of the final example,
 * and the output rows of the chunks are joined. This presumes the examples
 * do not interact (no operation mixes values across the batch), which the
 * graph cannot state; the caller vouches for it. */
NmStatus nm_engine_c_predict_batches(const NmEngineC *engine,
                                     const NmGraphModel *graph,
                                     const NmInput *inputs, size_t count,
                                     FloatMatrix **out_prediction);

NmStatus nm_engine_c_evaluate(const NmEngineC *engine,
                              const NmFrontendModel *model,
                              const FloatMatrix *inputs,
                              const FloatMatrix *targets,
                              float *out_accuracy);

/* Returns static backend capabilities configured for this engine. Safe to call
 * at any time (even before prepare or after model destruction). */
const NmBackendCapabilities *nm_engine_c_capabilities(const NmEngineC *engine);

#ifdef __cplusplus
}
#endif

#endif /* NM_ENGINE_C_H */
