/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Prints nm_lib's capability profile; used to refresh GraphIR's copy. */
#include "nm_capabilities.h"
#include <stdio.h>

int main(void) {
  fputs(nm_capabilities_json(), stdout);
  return 0;
}
