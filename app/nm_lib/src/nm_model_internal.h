/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/*
 * Internal shared declarations for the nm model implementation.
 */

#ifndef NM_MODEL_INTERNAL_H
#define NM_MODEL_INTERNAL_H

#include "nm_model_types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const NmBackendOps NM_BACKEND_DEFAULT_OPS;

/* Shared nm_seq API across implementation modules. */
NmSeq *nm_seq_create(NmShape input_shape);
void nm_seq_free(NmSeq *m);
NmStatus nm_seq_add_dense(NmSeq *m, const NmDenseConfig *cfg);
NmStatus nm_seq_add_conv2d(NmSeq *m, const NmConv2dConfig *cfg);
NmStatus nm_seq_add_maxpool2d(NmSeq *m, const NmPool2dConfig *cfg);
NmStatus nm_seq_add_avgpool2d(NmSeq *m, const NmPool2dConfig *cfg);
NmStatus nm_seq_add_batchnorm2d(NmSeq *m, const NmBatchNorm2dConfig *cfg);
NmStatus nm_seq_add_flatten(NmSeq *m);
NmStatus nm_seq_compile(NmSeq *m, const NmCompileConfig *cfg);
NmStatus nm_seq_set_optimizer(NmSeq *m, const NmOptimizerConfig *cfg);
NmOptimizerConfig nm_seq_optimizer_defaults(void);
NmStatus nm_seq_set_backend(NmSeq *m, NmBackendType backend);
const char *nm_seq_backend(const NmSeq *m);
const NmBackendCapabilities *nm_seq_model_backend_capabilities(const NmSeq *m);
bool nm_seq_is_inference_only(const NmSeq *m);
NmStatus nm_seq_set_tensor_layout(NmSeq *m, NmTensorLayout layout);
NmTensorLayout nm_seq_tensor_layout(const NmSeq *m);
NmStatus nm_seq_set_autograd_mode(NmSeq *m, NmAutogradMode mode);
NmAutogradMode nm_seq_autograd_mode(const NmSeq *m);
NmStatus nm_seq_set_batch_parallel(NmSeq *m, const NmBatchParallelConfig *cfg);
NmBatchParallelConfig nm_seq_batch_parallel(const NmSeq *m);
size_t nm_seq_last_graph_nodes(const NmSeq *m);
NmStatus nm_seq_fit(NmSeq *m, const FloatMatrix *X, const FloatMatrix *Y,
                    const NmTrainConfig *cfg);
NmStatus nm_seq_predict(const NmSeq *m, const FloatMatrix *input,
                        FloatMatrix **out_prediction);
NmStatus nm_seq_predict_into(const NmSeq *m, const FloatMatrix *input,
                             FloatMatrix *out_prediction);
NmStatus nm_seq_evaluate(const NmSeq *m, const FloatMatrix *X,
                         const FloatMatrix *Y, float *out_accuracy);
void nm_seq_summary(const NmSeq *m);
NmStatus nm_seq_save(const NmSeq *m, const char *path);
NmStatus nm_seq_save_inference(const NmSeq *m, const char *path);
NmStatus nm_seq_save_inference_ex(const NmSeq *m, const char *path,
                                  const NmSaveInferenceOptions *opts);
NmStatus nm_seq_load(const char *path, NmSeq **out_model);
NmStatus nm_seq_lower_to_ir(NmSeq *m);
NmStatus nm_seq_finalize(NmSeq *m);
NmStatus nm_seq_prepare_engine(NmSeq *m, const NmCompileConfig *cfg);
NmStatus nm_run_graph_plan_into(const NmLayerRuntime *runtime,
                                NmInferencePlan *plan,
                                const NmExecPlan *exec_plan,
                                const NmBackendOps *ops,
                                const FloatMatrix *input,
                                FloatMatrix *out_prediction);

/* Shared runtime/helpers. */
bool model_weights_are_f32(const NmSeq *m);
size_t shape_flat_size(NmShape s);
NmShape shape_from_tensor(const FloatTensor *t);
NmShape model_tail_shape(const NmSeq *m);
NmStatus ensure_capacity(NmSeq *m);
bool compile_uses_softmax_ce_shortcut(const NmSeq *m, NmLoss loss_type);
NmOptimizerConfig optimizer_defaults_internal(void);
bool optimizer_config_valid(NmOptimizerConfig cfg);
void optimizer_slot_free(NmOptSlot *slot);
bool optimizer_slot_ensure(NmOptSlot *slot, size_t size, bool need_m,
                           bool need_v);
uint64_t align_u64(uint64_t value, uint32_t alignment);
const NmBackendOps *model_backend_ops(const NmSeq *m);
const NmBackendCapabilities *model_backend_caps(const NmSeq *m);

void exec_plan_init(NmExecPlan *plan);
void exec_plan_reset(NmExecPlan *plan);
void exec_plan_destroy(NmExecPlan *plan);
NmStatus exec_plan_build(NmExecPlan *plan, const NmIrGraph *graph);

bool inference_plan_init(NmInferencePlan *plan);
void inference_plan_reset(NmInferencePlan *plan);
void inference_plan_destroy(NmInferencePlan *plan);
bool inference_plan_reinit(NmInferencePlan *plan);
void inference_plan_capture(NmInferencePlan *plan, const NmTensorPool *pool,
                            size_t batch_rows, size_t output_cols);

bool tensor_init_borrowed_view(FloatTensor *view, size_t ndim,
                               const size_t *shape, float *data,
                               size_t capacity, StLayout layout);
bool tensor_pool_init(NmTensorPool *pool);
void tensor_pool_reset(NmTensorPool *pool);
void tensor_pool_destroy(NmTensorPool *pool);
FloatTensor *tensor_pool_acquire(NmTensorPool *pool, size_t ndim,
                                 const size_t *shape);
FloatTensor *tensor_pool_clone(NmTensorPool *pool, const FloatTensor *src);

void exec_trace_init(NmExecTrace *t);
void exec_trace_reset(NmExecTrace *t);
void exec_trace_destroy(NmExecTrace *t);
bool exec_trace_push(NmExecTrace *t, NmLayerType layer_type,
                     NmShape output_shape, size_t tensor_numel);

bool optimizer_apply_array(const NmOptimizerConfig *cfg, uint64_t step,
                           NmOptSlot *slot, float *params, const float *grads,
                           size_t n, float grad_scale);
bool optimizer_apply_matrix(const NmOptimizerConfig *cfg, uint64_t step,
                            NmOptSlot *slot, FloatMatrix *param,
                            const FloatMatrix *grad, float grad_scale);
bool optimizer_apply_tensor(const NmOptimizerConfig *cfg, uint64_t step,
                            NmOptSlot *slot, FloatTensor *param,
                            const FloatTensor *grad, float grad_scale);

/* Shared model/layer helpers. */
const char *layer_type_name(NmLayerType t);
size_t layer_param_count(const NmLayerEntry *e);
FloatTensor *he_init_4d(size_t cout, size_t cin, size_t kh, size_t kw);
void destroy_layer_params(NmLayerEntry *e);
const NmLayerOps *layer_ops_for_type(NmLayerType type);
bool trainable_is_dense_only(const NmTrainable *t);
bool trainable_weights_are_f32(const NmTrainable *t);
bool trainable_uses_softmax_ce_shortcut(const NmTrainable *t);
NmStatus nm_train_fit(NmTrainable *t, const FloatMatrix *X, const FloatMatrix *Y,
                      const NmTrainConfig *cfg);
NmStatus layer_runtime_build(NmLayerRuntime *runtime, const NmIrGraph *graph,
                             const NmExecPlan *plan);
void layer_runtime_destroy(NmLayerRuntime *runtime);
/* Graph interpreter for inference on graphs the execution plan cannot run
 * (nm_graph_interp.c). `prepare` validates the graph with a dry run, folds
 * what depends on constants alone and plans the runs; it reports the flat
 * example sizes of the first input and of the output. A graph whose open
 * batch only works at the declared size (a batch-sized constant, say) reports
 * that size in *fixed_batch; otherwise it is 0. The prepared state reads the
 * graph's constants at every run but caches values derived from them:
 * prepare again after the weights change. One run at a time per state. */
typedef struct NmInterp NmInterp;
NmStatus graph_interp_prepare(const NmIrGraph *graph, NmInterp **out,
                              size_t *input_cols, size_t *output_cols,
                              size_t *fixed_batch);
void graph_interp_free(NmInterp *interp);
/* Matches `inputs` to the graph inputs (by name, or by position when a name
 * is NULL), checks their types and shapes and writes them in graph order to
 * `ordered`, which holds NM_IR_MAX_INPUTS entries. */
NmStatus graph_interp_bind(const NmIrGraph *graph, const NmInput *inputs,
                           size_t count, NmInput *ordered);
/* Runs the graph on inputs in graph order (see graph_interp_bind). *out has
 * one row per entry of the output's first dimension. */
NmStatus graph_interp_run(NmInterp *interp, const NmInput *inputs,
                          FloatMatrix **out);
bool dense_matrix_forward_into(const NmDenseParams *p, NmActivation act,
                               const FloatMatrix *input, FloatMatrix *output,
                               const NmBackendOps *backend);
void layer_apply_activation(NmActivation act, FloatTensor *t);
NmStatus model_train_one_epoch_dense_fast(NmTrainable *t, const FloatMatrix *X,
                                          const FloatMatrix *Y,
                                          size_t batch_size, float lr,
                                          float *out_epoch_loss,
                                          bool track_loss);
FloatTensor *layer_forward(const NmLayerEntry *e, const FloatTensor *input,
                           FloatTensor **out_pool_idx, NmTensorPool *pool,
                           const NmBackendOps *backend);
FloatTensor *layer_backward(NmLayerEntry *e, const FloatTensor *input,
                            const FloatTensor *activation, FloatTensor *grad,
                            FloatTensor *pool_indices, float lr, bool skip_act,
                            const NmOptimizerConfig *opt_cfg, uint64_t opt_step,
                            NmTensorPool *pool, const NmBackendOps *backend);
bool init_batch_input_view(FloatTensor *out, const FloatMatrix *X,
                           NmShape shape, size_t start, size_t count);
bool init_batch_target_view(FloatTensor *out, const FloatMatrix *Y,
                            size_t start, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* NM_MODEL_INTERNAL_H */
