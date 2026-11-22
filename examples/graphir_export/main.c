/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_engine_c.h"
#include "nm_export_graphir.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  float inputs[] = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
  };
  float targets[] = {
      1.0f, 0.0f,
      0.0f, 1.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
  };
  FloatMatrix *x = sm_create_with_values(4, 2, inputs);
  FloatMatrix *y = sm_create_with_values(4, 2, targets);
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 1, .dims = {2}});
  NmEngineC *engine = NULL;
  FloatMatrix *prediction = NULL;
  NmStatus status = (!x || !y || !model) ? NM_STATUS_ALLOC_FAILED : NM_STATUS_OK;

  if (status == NM_STATUS_OK)
    status = nm_frontend_add_dense(
        model, &(NmDenseConfig){.units = 8, .activation = NM_ACT_RELU});
  if (status == NM_STATUS_OK)
    status = nm_frontend_add_dense(
        model, &(NmDenseConfig){.units = 2, .activation = NM_ACT_SOFTMAX});
  if (status == NM_STATUS_OK)
    status = nm_frontend_finalize(model);
  if (status == NM_STATUS_OK) {
    engine = nm_engine_c_create(NULL);
    status = engine ? nm_engine_c_prepare(engine, model) : NM_STATUS_ALLOC_FAILED;
  }
  if (status == NM_STATUS_OK) {
    NmTrainConfig train = {
        .batch_size = 4, .learning_rate = 0.5f, .epochs = 2500, .verbose = false};
    status = nm_engine_c_fit(engine, model, x, y, &train);
  }
  if (status == NM_STATUS_OK)
    status = nm_engine_c_predict(engine, model, x, &prediction);
  if (status == NM_STATUS_OK) {
    for (size_t row = 0; row < prediction->rows; ++row) {
      printf("XOR(%.0f, %.0f) -> [%.3f, %.3f]\n", x->values[row * 2],
             x->values[row * 2 + 1], prediction->values[row * 2],
             prediction->values[row * 2 + 1]);
    }
    status = nm_export_graphir_json(
        model, &(NmGraphIrExportOptions){.batch_size = 4}, "xor.graphir.json");
  }
  if (status != NM_STATUS_OK)
    fprintf(stderr, "graphir_export failed: status=%d\n", (int)status);

  sm_destroy(prediction);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  sm_destroy(y);
  sm_destroy(x);
  return status == NM_STATUS_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
