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

int main(int argc, char **argv) {
  /* usage: fixture model.json expected.txt [weights.girw] */
  if (argc != 3 && argc != 4)
    return 2;
  NmFrontendModel *model =
      nm_frontend_create((NmShape){.ndim = 1, .dims = {3}});
  if (!model ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 2,
                                                    .activation = NM_ACT_RELU}) ||
      nm_frontend_add_dense(model, &(NmDenseConfig){.units = 2,
                                                    .activation = NM_ACT_SOFTMAX}) ||
      nm_frontend_finalize(model)) {
    nm_frontend_free(model);
    return 1;
  }
  NmEngineC *engine = nm_engine_c_create(NULL);
  FloatMatrix *input = sm_create(2, 3);
  FloatMatrix *prediction = NULL;
  if (input) {
    const float values[] = {0.0f, 0.25f, -0.5f, 1.0f, -0.75f, 0.5f};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i)
      input->values[i] = values[i];
  }
  if (!engine || !input || nm_engine_c_prepare(engine, model) ||
      nm_engine_c_predict(engine, model, input, &prediction) ||
      nm_export_graphir_json(model,
                             &(NmGraphIrExportOptions){
                                 .batch_size = 2,
                                 .weights_path = argc == 4 ? argv[3] : NULL},
                             argv[1])) {
    sm_destroy(prediction);
    sm_destroy(input);
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    return 1;
  }
  FILE *expected = fopen(argv[2], "w");
  if (!expected) {
    sm_destroy(prediction);
    sm_destroy(input);
    nm_engine_c_free(engine);
    nm_frontend_free(model);
    return 1;
  }
  for (size_t i = 0; i < prediction->rows * prediction->cols; ++i)
    fprintf(expected, "%.9g\n", (double)prediction->values[i]);
  int close_status = fclose(expected);
  sm_destroy(prediction);
  sm_destroy(input);
  nm_engine_c_free(engine);
  nm_frontend_free(model);
  return close_status == 0 ? 0 : 1;
}
