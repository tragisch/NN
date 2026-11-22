/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_girw.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <xxhash.h>

#define GIRW_LEGACY_VERSION 1U
#define GIRW_VERSION 2U
#define GIRW_LEGACY_HEADER 24U
#define GIRW_HEADER 32U
#define GIRW_ALIGNMENT 64U
#define GIRW_MAX_NAME 65535U

struct NmGirwPackage {
  uint8_t *bytes;
  bool owns_bytes;
  NmGirwVerify verify;
  size_t size;
  NmGirwEntry *entries;
  char **names;
  size_t count;
};

typedef struct {
  const uint8_t *bytes;
  size_t size;
  size_t offset;
} Cursor;

static bool take(Cursor *c, size_t length, const uint8_t **out) {
  if (length > c->size - c->offset)
    return false;
  *out = c->bytes + c->offset;
  c->offset += length;
  return true;
}

static bool read_le(Cursor *c, size_t width, uint64_t *out) {
  const uint8_t *p;
  if (!take(c, width, &p))
    return false;
  uint64_t value = 0;
  for (size_t i = width; i > 0; --i)
    value = (value << 8) | p[i - 1];
  *out = value;
  return true;
}

static bool dtype_bytes(uint64_t code, uint64_t count, uint64_t *out) {
  switch (code) {
  case NM_GIRW_FLOAT16:
  case NM_GIRW_BFLOAT16:
    if (count > UINT64_MAX / 2)
      return false;
    *out = count * 2;
    return true;
  case NM_GIRW_FLOAT32:
    if (count > UINT64_MAX / 4)
      return false;
    *out = count * 4;
    return true;
  case NM_GIRW_INT4:
  case NM_GIRW_UINT4:
    if (count == UINT64_MAX)
      return false;
    *out = (count + 1) / 2;
    return true;
  case NM_GIRW_INT8:
  case NM_GIRW_UINT8:
    *out = count;
    return true;
  default:
    return false;
  }
}

static uint8_t *read_file_aligned(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  uint8_t *buffer = NULL;
  long length = 0;
  if (fseek(file, 0, SEEK_END) == 0 && (length = ftell(file)) >= 0 &&
      fseek(file, 0, SEEK_SET) == 0) {
    size_t rounded = ((size_t)length + GIRW_ALIGNMENT - 1) / GIRW_ALIGNMENT *
                     GIRW_ALIGNMENT;
    void *memory = NULL;
    if (posix_memalign(&memory, GIRW_ALIGNMENT, rounded ? rounded : GIRW_ALIGNMENT) == 0) {
      buffer = memory;
      if (fread(buffer, 1, (size_t)length, file) != (size_t)length) {
        free(buffer);
        buffer = NULL;
      } else {
        *size = (size_t)length;
      }
    }
  }
  fclose(file);
  return buffer;
}

void nm_girw_close(NmGirwPackage *package) {
  if (!package)
    return;
  if (package->names) {
    for (size_t i = 0; i < package->count; ++i)
      free(package->names[i]);
  }
  free(package->names);
  free(package->entries);
  if (package->owns_bytes)
    free(package->bytes);
  free(package);
}

static NmStatus parse(NmGirwPackage *package) {
  Cursor c = {package->bytes, package->size, 0};
  const uint8_t *magic;
  uint64_t version, count, data_start, index_checksum = 0;
  if (!take(&c, 8, &magic) || memcmp(magic, "GIRWPKG\0", 8) != 0 ||
      !read_le(&c, 4, &version) || (version != GIRW_VERSION && version != GIRW_LEGACY_VERSION))
    return NM_STATUS_FORMAT_ERROR;
  const bool checksummed = version == GIRW_VERSION;
  if (!read_le(&c, 4, &count) || !read_le(&c, 8, &data_start) ||
      (checksummed && !read_le(&c, 8, &index_checksum)) ||
      data_start > package->size)
    return NM_STATUS_FORMAT_ERROR;
  if (checksummed && data_start % GIRW_ALIGNMENT != 0)
    return NM_STATUS_FORMAT_ERROR;
  const size_t min_entry = checksummed ? 36 : 28;
  if (count > (package->size - c.offset) / min_entry)
    return NM_STATUS_FORMAT_ERROR;

  package->entries = calloc(count ? count : 1, sizeof(*package->entries));
  package->names = calloc(count ? count : 1, sizeof(*package->names));
  if (!package->entries || !package->names)
    return NM_STATUS_ALLOC_FAILED;
  package->count = (size_t)count;

  const size_t index_start = c.offset;
  for (size_t i = 0; i < count; ++i) {
    uint64_t name_length, dtype, reserved, elements, offset, length, checksum = 0;
    const uint8_t *name;
    if (!read_le(&c, 2, &name_length) || !take(&c, (size_t)name_length, &name) ||
        !read_le(&c, 1, &dtype) || !read_le(&c, 1, &reserved) ||
        !read_le(&c, 8, &elements) || !read_le(&c, 8, &offset) ||
        !read_le(&c, 8, &length) || (checksummed && !read_le(&c, 8, &checksum)))
      return NM_STATUS_FORMAT_ERROR;
    uint64_t expected;
    if (!dtype_bytes(dtype, elements, &expected) || expected != length ||
        offset < data_start || offset > package->size ||
        length > package->size - offset ||
        (checksummed && offset % GIRW_ALIGNMENT != 0))
      return NM_STATUS_FORMAT_ERROR;
    char *copy = malloc((size_t)name_length + 1);
    if (!copy)
      return NM_STATUS_ALLOC_FAILED;
    memcpy(copy, name, (size_t)name_length);
    copy[name_length] = '\0';
    package->names[i] = copy;
    package->entries[i] = (NmGirwEntry){.name = copy,
                                        .dtype = (NmGirwDType)dtype,
                                        .element_count = elements,
                                        .offset = offset,
                                        .byte_length = length,
                                        .data = package->bytes + offset};
    if (checksummed && package->verify == NM_GIRW_VERIFY_ALL &&
        XXH64(package->bytes + offset, (size_t)length, 0) != checksum)
      return NM_STATUS_FORMAT_ERROR;
  }
  if (checksummed &&
      (c.offset > data_start ||
       XXH64(package->bytes + index_start, c.offset - index_start, 0) !=
           index_checksum))
    return NM_STATUS_FORMAT_ERROR;

  for (size_t i = 0; i < package->count; ++i) {
    for (size_t j = i + 1; j < package->count; ++j) {
      const NmGirwEntry *a = &package->entries[i];
      const NmGirwEntry *b = &package->entries[j];
      if (a->offset < b->offset + b->byte_length &&
          b->offset < a->offset + a->byte_length)
        return NM_STATUS_FORMAT_ERROR;
    }
  }
  return NM_STATUS_OK;
}

NmStatus nm_girw_open(const char *path, NmGirwPackage **out_package) {
  if (!path || !out_package)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_package = NULL;
  NmGirwPackage *package = calloc(1, sizeof(*package));
  if (!package)
    return NM_STATUS_ALLOC_FAILED;
  package->bytes = read_file_aligned(path, &package->size);
  package->owns_bytes = true;
  package->verify = NM_GIRW_VERIFY_ALL;
  if (!package->bytes) {
    free(package);
    return NM_STATUS_IO_ERROR;
  }
  NmStatus status = parse(package);
  if (status != NM_STATUS_OK) {
    nm_girw_close(package);
    return status;
  }
  *out_package = package;
  return NM_STATUS_OK;
}

size_t nm_girw_entry_count(const NmGirwPackage *package) {
  return package ? package->count : 0;
}

const NmGirwEntry *nm_girw_entry(const NmGirwPackage *package, size_t index) {
  return package && index < package->count ? &package->entries[index] : NULL;
}

const NmGirwEntry *nm_girw_find(const NmGirwPackage *package, uint64_t offset,
                                uint64_t byte_length, uint64_t element_count,
                                NmGirwDType dtype) {
  if (!package)
    return NULL;
  for (size_t i = 0; i < package->count; ++i) {
    const NmGirwEntry *e = &package->entries[i];
    if (e->offset == offset && e->byte_length == byte_length &&
        e->element_count == element_count && e->dtype == dtype)
      return e;
  }
  return NULL;
}

NmStatus nm_girw_open_memory(const void *data, size_t size, NmGirwVerify verify,
                             NmGirwPackage **out_package) {
  if (!data || !out_package)
    return NM_STATUS_INVALID_ARGUMENT;
  *out_package = NULL;
  NmGirwPackage *package = calloc(1, sizeof(*package));
  if (!package)
    return NM_STATUS_ALLOC_FAILED;
  package->bytes = (uint8_t *)(uintptr_t)data;
  package->size = size;
  package->owns_bytes = false;
  package->verify = verify;
  NmStatus status = parse(package);
  if (status != NM_STATUS_OK) {
    nm_girw_close(package);
    return status;
  }
  *out_package = package;
  return NM_STATUS_OK;
}

/* ---- Writer --------------------------------------------------------------- */

#define GIRW_ENTRY_FIXED 36U

static bool layout(const NmGirwSource *sources, size_t count, size_t *data_start,
                   size_t *total, uint64_t *lengths) {
  size_t index = 0;
  for (size_t i = 0; i < count; ++i) {
    size_t name_length = sources[i].name ? strlen(sources[i].name) : 0;
    if (name_length > GIRW_MAX_NAME ||
        !dtype_bytes(sources[i].dtype, sources[i].element_count, &lengths[i]))
      return false;
    index += GIRW_ENTRY_FIXED + name_length;
  }
  size_t start = (GIRW_HEADER + index + GIRW_ALIGNMENT - 1) / GIRW_ALIGNMENT *
                 GIRW_ALIGNMENT;
  size_t offset = start;
  size_t end = start;
  for (size_t i = 0; i < count; ++i) {
    if (lengths[i] > SIZE_MAX - offset)
      return false;
    end = offset + (size_t)lengths[i];
    offset = (end + GIRW_ALIGNMENT - 1) / GIRW_ALIGNMENT * GIRW_ALIGNMENT;
  }
  *data_start = start;
  *total = count ? end : start;
  return true;
}

size_t nm_girw_image_size(const NmGirwSource *sources, size_t count) {
  if (count && !sources)
    return 0;
  uint64_t *lengths = calloc(count ? count : 1, sizeof(*lengths));
  if (!lengths)
    return 0;
  size_t start = 0, total = 0;
  bool ok = layout(sources, count, &start, &total, lengths);
  free(lengths);
  return ok ? total : 0;
}

static void put_le(uint8_t *out, size_t width, uint64_t value) {
  for (size_t i = 0; i < width; ++i)
    out[i] = (uint8_t)(value >> (8 * i));
}

NmStatus nm_girw_write_image(const NmGirwSource *sources, size_t count,
                             void *out, size_t out_size) {
  if (!out || (count && !sources))
    return NM_STATUS_INVALID_ARGUMENT;
  uint64_t *lengths = calloc(count ? count : 1, sizeof(*lengths));
  if (!lengths)
    return NM_STATUS_ALLOC_FAILED;
  size_t data_start = 0, total = 0;
  if (!layout(sources, count, &data_start, &total, lengths)) {
    free(lengths);
    return NM_STATUS_INVALID_ARGUMENT;
  }
  if (out_size < total) {
    free(lengths);
    return NM_STATUS_INVALID_ARGUMENT;
  }

  uint8_t *image = out;
  memset(image, 0, total);
  uint8_t *entry = image + GIRW_HEADER;
  size_t offset = data_start;
  for (size_t i = 0; i < count; ++i) {
    size_t name_length = sources[i].name ? strlen(sources[i].name) : 0;
    put_le(entry, 2, name_length);
    memcpy(entry + 2, sources[i].name, name_length);
    uint8_t *fixed = entry + 2 + name_length;
    fixed[0] = (uint8_t)sources[i].dtype;
    fixed[1] = 0;
    put_le(fixed + 2, 8, sources[i].element_count);
    put_le(fixed + 10, 8, offset);
    put_le(fixed + 18, 8, lengths[i]);
    put_le(fixed + 26, 8, XXH64(sources[i].data, (size_t)lengths[i], 0));
    memcpy(image + offset, sources[i].data, (size_t)lengths[i]);
    entry += GIRW_ENTRY_FIXED + name_length;
    offset = (offset + (size_t)lengths[i] + GIRW_ALIGNMENT - 1) / GIRW_ALIGNMENT *
             GIRW_ALIGNMENT;
  }
  memcpy(image, "GIRWPKG\0", 8);
  put_le(image + 8, 4, GIRW_VERSION);
  put_le(image + 12, 4, count);
  put_le(image + 16, 8, data_start);
  put_le(image + 24, 8, XXH64(image + GIRW_HEADER, (size_t)(entry - (image + GIRW_HEADER)), 0));
  free(lengths);
  return NM_STATUS_OK;
}
