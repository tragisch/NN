/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_frontend.h"
#include "nm_frontend_internal.h"
#include "nm_model_internal.h"
#include <stdlib.h>

/* The frontend collects layers in a builder (which validates shapes and draws
 * the initial weights) and hands the result over as a graph that owns its
 * weights. The builder is gone after finalize. */
struct NmFrontendModel {
  NmSeq *builder;
  NmIrGraph graph;
  bool finalized;
};

static NmFrontendModel *frontend_new(NmSeq *builder) {
  NmFrontendModel *model = calloc(1, sizeof(*model));
  if (!model)
    return NULL;
  nm_ir_graph_init(&model->graph);
  model->builder = builder;
  return model;
}

NmFrontendModel *nm_frontend_create(NmShape input_shape) {
  NmSeq *builder = nm_seq_create(input_shape);
  if (!builder)
    return NULL;
  NmFrontendModel *model = frontend_new(builder);
  if (!model)
    nm_seq_free(builder);
  return model;
}

void nm_frontend_free(NmFrontendModel *model) {
  if (!model)
    return;
  nm_seq_free(model->builder);
  nm_ir_graph_destroy(&model->graph);
  free(model);
}

static NmSeq *open_builder(NmFrontendModel *model) {
  return model && !model->finalized ? model->builder : NULL;
}

NmStatus nm_frontend_add_dense(NmFrontendModel *model,
                               const NmDenseConfig *config) {
  NmSeq *builder = open_builder(model);
  return builder ? nm_seq_add_dense(builder, config)
                 : NM_STATUS_INVALID_ARGUMENT;
}

NmStatus nm_frontend_add_conv2d(NmFrontendModel *model,
                                const NmConv2dConfig *config) {
  NmSeq *builder = open_builder(model);
  return builder ? nm_seq_add_conv2d(builder, config)
                 : NM_STATUS_INVALID_ARGUMENT;
}

NmStatus nm_frontend_add_maxpool2d(NmFrontendModel *model,
                                   const NmPool2dConfig *config) {
  NmSeq *builder = open_builder(model);
  return builder ? nm_seq_add_maxpool2d(builder, config)
                 : NM_STATUS_INVALID_ARGUMENT;
}

NmStatus nm_frontend_add_flatten(NmFrontendModel *model) {
  NmSeq *builder = open_builder(model);
  return builder ? nm_seq_add_flatten(builder) : NM_STATUS_INVALID_ARGUMENT;
}

/* Moves a finalized builder's graph into `model` and releases the builder. */
static NmStatus detach_graph(NmFrontendModel *model) {
  NmSeq *builder = model->builder;
  if (!nm_ir_graph_clone_owned(&model->graph, &builder->compiled_graph))
    return NM_STATUS_ALLOC_FAILED;
  nm_seq_free(builder);
  model->builder = NULL;
  model->finalized = true;
  return NM_STATUS_OK;
}

NmStatus nm_frontend_finalize(NmFrontendModel *model) {
  NmSeq *builder = open_builder(model);
  if (!builder)
    return NM_STATUS_INVALID_ARGUMENT;
  NmStatus status = nm_seq_finalize(builder);
  return status == NM_STATUS_OK ? detach_graph(model) : status;
}

NmGraphModel *nm_frontend_graph(NmFrontendModel *model) {
  return model && model->finalized ? &model->graph : NULL;
}

NmStatus nm_frontend_adopt_seq(NmSeq *seq, NmFrontendModel **out_model) {
  if (!seq || !out_model)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_model = NULL;
  if (!seq->compiled) {
    NmStatus status = nm_seq_finalize(seq);
    if (status != NM_STATUS_OK)
      return status;
  }
  NmFrontendModel *model = frontend_new(seq);
  if (!model)
    return NM_STATUS_ALLOC_FAILED;
  NmStatus status = detach_graph(model);
  if (status != NM_STATUS_OK) {
    model->builder = NULL; /* the caller still owns `seq` */
    nm_frontend_free(model);
    return status;
  }
  *out_model = model;
  return NM_STATUS_OK;
}
