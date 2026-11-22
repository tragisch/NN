/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Public user API of nm_lib: describe a model, build it, train it, run it and
 * save it. One header, opaque handles, plain C types (importable from Swift
 * through include/module.modulemap).
 *
 * Threading: calls on one network take turns (an internal lock), so it may be
 * shared between threads, but predict calls do not run in parallel; use one
 * network per thread for that (build the same spec, or load the saved model).
 * Different networks are independent.
 * Errors: every function reports an NmStatus (or NULL) and leaves a readable
 * message for the calling thread in nm_last_error(). */
#ifndef NM_API_H
#define NM_API_H

#include "nm_types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__clang__)
#define NM_NONNULL _Nonnull
#define NM_NULLABLE _Nullable
#else
#define NM_NONNULL
#define NM_NULLABLE
#endif

/* ---- Data ------------------------------------------------------------- */

/* A borrowed, row-major float32 array. dims[0] is the batch (number of
 * examples); the other dims are the example's shape, so a 28x28 image batch is
 * {4, {n, 1, 28, 28}} and an MLP batch {2, {n, features}}. Tensors passed in
 * are never copied or freed. A tensor filled by nm_lib owns its data until
 * nm_tensor_free. */
typedef struct {
  float *NM_NULLABLE data;
  size_t ndim; /* 2..4 */
  size_t dims[4];
} NmTensor;

NmTensor nm_tensor_2d(float *NM_NULLABLE data, size_t rows, size_t cols);
size_t nm_tensor_numel(const NmTensor *NM_NONNULL tensor);
void nm_tensor_free(NmTensor *NM_NONNULL tensor); /* only for nm_lib-owned data */

/* ---- Describing a model ----------------------------------------------- */

typedef struct NmSpec NmSpec;

/* input: shape of one example, e.g. {1, {784}} or {3, {1, 28, 28}}. */
NmSpec *NM_NULLABLE nm_spec_new(NmShape input);
void nm_spec_free(NmSpec *NM_NULLABLE spec);

NmStatus nm_spec_dense(NmSpec *NM_NONNULL spec, size_t units,
                       NmActivation activation);
NmStatus nm_spec_conv2d(NmSpec *NM_NONNULL spec, size_t filters, size_t kernel,
                        size_t stride, size_t padding, NmActivation activation);
NmStatus nm_spec_maxpool2d(NmSpec *NM_NONNULL spec, size_t kernel,
                           size_t stride);
NmStatus nm_spec_flatten(NmSpec *NM_NONNULL spec);

/* ---- Building ---------------------------------------------------------- */

typedef struct {
  uint64_t seed;      /* 0: keep the library's random state */
  bool deterministic; /* same seed and data give the same weights */
  NmBackendType backend;
} NmBuildOptions;

NmBuildOptions nm_build_options_default(void);

typedef struct NmNetwork NmNetwork;

/* Validates the spec, draws the initial weights and prepares the engine. A spec
 * can be built many times (ensembles, hyper-parameter runs). NULL on error. */
NmNetwork *NM_NULLABLE nm_network_build(const NmSpec *NM_NONNULL spec,
                                    const NmBuildOptions *NM_NULLABLE options);
void nm_network_free(NmNetwork *NM_NULLABLE model);

/* ---- Training ---------------------------------------------------------- */

typedef struct {
  NmLoss loss;           /* NM_LOSS_NONE: cross-entropy with targets, MSE
                            without (autoencoder) */
  NmOptimizerType optimizer;
  float learning_rate;
  size_t batch_size;
  size_t epochs;
  bool shuffle;          /* reshuffle the training rows every epoch */
  float validation_split; /* fraction of the last rows held out, 0 .. <1 */
} NmFitOptions;

NmFitOptions nm_fit_options_default(void);

typedef struct {
  size_t epoch;   /* 1-based */
  size_t epochs;
  float loss;     /* on the training rows after this epoch */
  float val_loss; /* on the held-out rows, NAN without validation_split */
} NmProgress;

/* Called after every epoch in the training thread. Return non-zero to stop
 * training early (fit then returns NM_STATUS_OK). */
typedef int (*NmProgressFn)(const NmProgress *NM_NONNULL progress,
                            void *NM_NULLABLE user);

/* targets == NULL trains an autoencoder: the target is the input. */
NmStatus nm_network_fit(NmNetwork *NM_NONNULL model, NmTensor inputs,
                      const NmTensor *NM_NULLABLE targets,
                      const NmFitOptions *NM_NULLABLE options,
                      NmProgressFn NM_NULLABLE progress,
                      void *NM_NULLABLE user);

/* ---- Using ------------------------------------------------------------- */

/* Allocates *out (free with nm_tensor_free), shape {2, {batch, outputs}}. */
NmStatus nm_network_predict(const NmNetwork *NM_NONNULL model, NmTensor inputs,
                          NmTensor *NM_NONNULL out);

/* Models with several inputs or with integer and boolean inputs (token ids
 * and a padding mask, say) take NmInput values (nm_types.h): one per model
 * input, matched by name or, where the name is NULL, by position. Shapes must
 * be the declared ones; a model with a single input also accepts any batch
 * size. Allocates *out (free with nm_tensor_free): float32, shape
 * {2, {rows, values per row}} where rows is the output's first dimension.
 * Integer and boolean outputs are converted to float. */
NmStatus nm_network_predict_inputs(const NmNetwork *NM_NONNULL model,
                                   const NmInput *NM_NULLABLE inputs,
                                   size_t count, NmTensor *NM_NONNULL out);

/* Like nm_network_predict_inputs for any number of examples, for models whose
 * declared shapes fix the batch (several inputs, say): the examples run in
 * chunks of the declared batch, the last chunk padded, and the output rows
 * are joined, one row group per example as for the declared batch. Use it
 * only for models whose examples do not interact (true for per-sequence
 * encoders such as a transformer with a padding mask); the model cannot state
 * that itself. */
NmStatus nm_network_predict_batches(const NmNetwork *NM_NONNULL model,
                                    const NmInput *NM_NULLABLE inputs,
                                    size_t count, NmTensor *NM_NONNULL out);

/* Number of model inputs, and the name, type and declared shape of one of
 * them (dims[0] == 0: any batch size). `out->data` is NULL; `out->name`
 * stays valid as long as the network. */
size_t nm_network_input_count(const NmNetwork *NM_NONNULL model);
NmStatus nm_network_input_info(const NmNetwork *NM_NONNULL model, size_t index,
                               NmInput *NM_NONNULL out);

typedef struct {
  float loss;
  float accuracy; /* classification (cross-entropy) only, else -1 */
} NmMetrics;

NmStatus nm_network_evaluate(const NmNetwork *NM_NONNULL model, NmTensor inputs,
                           NmTensor targets, NmLoss loss,
                           NmMetrics *NM_NONNULL out);

/* ---- Inspecting -------------------------------------------------------- */

/* Number of trainable values (weights and biases). */
size_t nm_network_parameter_count(const NmNetwork *NM_NONNULL network);

/* Writes a table of the layers, their output shapes (without the batch
 * dimension) and parameter counts, ending with the total. Like snprintf it
 * returns the length the full text needs, so a too small buffer is detected;
 * `buffer` may be NULL to ask for the size. */
size_t nm_network_summary(const NmNetwork *NM_NONNULL network,
                          char *NM_NULLABLE buffer, size_t size);

/* ---- Saving ------------------------------------------------------------ */

/* Writes dir/model.graphir.json (GraphIR v4) and dir/weights.girw (checksummed
 * weight package). Creates dir. */
NmStatus nm_network_save(const NmNetwork *NM_NONNULL model,
                       const char *NM_NONNULL dir);
NmNetwork *NM_NULLABLE nm_network_load(const char *NM_NONNULL dir);
/* Loads any GraphIR v3/v4 document, inline or with external `.girw` weights
 * (resolved relative to the document's directory). The model must use the
 * operations in nm_capabilities.h. */
NmNetwork *NM_NULLABLE nm_network_import_graphir(const char *NM_NONNULL path);
/* One self-contained GraphIR document with inline weights at any path. */
NmStatus nm_network_export_graphir(const NmNetwork *NM_NONNULL model,
                                 const char *NM_NONNULL path);

/* Message of the last failed call on this thread ("" if none). */
const char *NM_NONNULL nm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NM_API_H */
