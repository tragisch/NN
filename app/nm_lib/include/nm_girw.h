/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Reader for GraphIR `.girw` weight packages (versions 1 and 2).
 *
 * Version 2 packages carry XXH64 checksums for the index and every blob and
 * start each blob on a 64-byte boundary. The reader verifies all of it when
 * the package is opened, so entry data can be used without further checks. The
 * format is specified in GraphIR's docs/ir_design.md; the byte-exact reference
 * file is tests/fixtures/weights_v2.girw in that repository. Checksums detect
 * corruption, not tampering. */
#ifndef NM_GIRW_H
#define NM_GIRW_H

#include "nm_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NM_GIRW_FLOAT16 = 1,
  NM_GIRW_BFLOAT16 = 2,
  NM_GIRW_FLOAT32 = 3,
  NM_GIRW_INT4 = 4,
  NM_GIRW_UINT4 = 5,
  NM_GIRW_INT8 = 6,
  NM_GIRW_UINT8 = 7,
} NmGirwDType;

typedef struct {
  const char *name;
  NmGirwDType dtype;
  uint64_t element_count;
  uint64_t offset;      /* byte offset in the file */
  uint64_t byte_length;
  const uint8_t *data;  /* points into the package, valid until close */
} NmGirwEntry;

typedef struct NmGirwPackage NmGirwPackage;

/* How much `nm_girw_open_memory` verifies. The structure, ranges, alignment and
 * the index checksum are always checked. */
typedef enum {
  NM_GIRW_VERIFY_ALL = 0,        /* also every blob checksum */
  NM_GIRW_VERIFY_INDEX_ONLY = 1, /* trusted source, e.g. shared memory */
} NmGirwVerify;

/* Parses a package that already is in memory, without copying. `data` must stay
 * valid and unchanged until nm_girw_close, and offsets in the package are
 * relative to its start, so a 64-byte aligned `data` yields aligned entries. */
NmStatus nm_girw_open_memory(const void *data, size_t size, NmGirwVerify verify,
                             NmGirwPackage **out_package);

/* Writes a version 2 package. `nm_girw_image_size` returns the exact size (0 on
 * overflow); the image is byte-identical to what GraphIR's builder writes. */
typedef struct {
  const char *name;
  NmGirwDType dtype;
  uint64_t element_count;
  const void *data;
} NmGirwSource;

size_t nm_girw_image_size(const NmGirwSource *sources, size_t count);
NmStatus nm_girw_write_image(const NmGirwSource *sources, size_t count,
                             void *out, size_t out_size);

/* Reads and verifies a package. NM_STATUS_IO_ERROR if the file cannot be read,
 * NM_STATUS_FORMAT_ERROR for malformed packages and checksum mismatches. */
NmStatus nm_girw_open(const char *path, NmGirwPackage **out_package);
void nm_girw_close(NmGirwPackage *package);

size_t nm_girw_entry_count(const NmGirwPackage *package);
const NmGirwEntry *nm_girw_entry(const NmGirwPackage *package, size_t index);

/* Finds the entry that a GraphIR external descriptor points at. */
const NmGirwEntry *nm_girw_find(const NmGirwPackage *package, uint64_t offset,
                                uint64_t byte_length, uint64_t element_count,
                                NmGirwDType dtype);

#ifdef __cplusplus
}
#endif

#endif /* NM_GIRW_H */
