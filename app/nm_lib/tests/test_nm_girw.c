/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm_girw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

/* Byte-exact reference written by GraphIR's WeightPackageBuilder (version 2). */
#define FIXTURE "app/nm_lib/tests/fixtures/weights_v2.girw"
#define COPY "tmp_nm_girw_copy.girw"

void setUp(void) {}
void tearDown(void) { remove(COPY); }

static uint8_t *read_all(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  TEST_ASSERT_NOT_NULL(file);
  TEST_ASSERT_EQUAL_INT(0, fseek(file, 0, SEEK_END));
  long length = ftell(file);
  TEST_ASSERT_TRUE(length > 0);
  TEST_ASSERT_EQUAL_INT(0, fseek(file, 0, SEEK_SET));
  uint8_t *bytes = malloc((size_t)length);
  TEST_ASSERT_NOT_NULL(bytes);
  TEST_ASSERT_EQUAL_size_t((size_t)length, fread(bytes, 1, (size_t)length, file));
  fclose(file);
  *size = (size_t)length;
  return bytes;
}

static void write_all(const char *path, const uint8_t *bytes, size_t size) {
  FILE *file = fopen(path, "wb");
  TEST_ASSERT_NOT_NULL(file);
  TEST_ASSERT_EQUAL_size_t(size, fwrite(bytes, 1, size, file));
  fclose(file);
}

void test_reference_package_opens_and_exposes_entries(void) {
  NmGirwPackage *package = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_girw_open(FIXTURE, &package));
  TEST_ASSERT_EQUAL_size_t(2, nm_girw_entry_count(package));

  const NmGirwEntry *weight = nm_girw_entry(package, 0);
  TEST_ASSERT_EQUAL_STRING("layer.0.weight", weight->name);
  TEST_ASSERT_EQUAL(NM_GIRW_FLOAT32, weight->dtype);
  TEST_ASSERT_EQUAL_UINT64(24, weight->element_count);
  TEST_ASSERT_EQUAL_UINT64(96, weight->byte_length);
  TEST_ASSERT_EQUAL_UINT64(0, weight->offset % 64);
  TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)(uintptr_t)weight->data % 64);
  const float *weights = (const float *)weight->data;
  for (int i = 0; i < 24; ++i)
    TEST_ASSERT_EQUAL_FLOAT((float)i * 0.25f - 3.0f, weights[i]);

  const NmGirwEntry *bias = nm_girw_entry(package, 1);
  TEST_ASSERT_EQUAL_STRING("layer.0.bias", bias->name);
  const float *values = (const float *)bias->data;
  TEST_ASSERT_EQUAL_FLOAT(0.5f, values[0]);
  TEST_ASSERT_EQUAL_FLOAT(-1.5f, values[1]);
  TEST_ASSERT_EQUAL_FLOAT(2.0f, values[2]);

  TEST_ASSERT_EQUAL_PTR(bias, nm_girw_find(package, bias->offset, 12, 3,
                                           NM_GIRW_FLOAT32));
  TEST_ASSERT_NULL(nm_girw_find(package, bias->offset, 12, 3, NM_GIRW_INT8));
  nm_girw_close(package);
}

void test_corrupted_blob_is_rejected(void) {
  size_t size = 0;
  uint8_t *bytes = read_all(FIXTURE, &size);
  NmGirwPackage *package = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_girw_open(FIXTURE, &package));
  size_t flipped = (size_t)nm_girw_entry(package, 1)->offset;
  nm_girw_close(package);

  bytes[flipped] ^= 0x01;
  write_all(COPY, bytes, size);
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_girw_open(COPY, &package));
  TEST_ASSERT_NULL(package);
  free(bytes);
}

void test_corrupted_index_is_rejected(void) {
  size_t size = 0;
  uint8_t *bytes = read_all(FIXTURE, &size);
  bytes[33] ^= 0x01; /* inside the first name length */
  write_all(COPY, bytes, size);
  NmGirwPackage *package = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, nm_girw_open(COPY, &package));
  TEST_ASSERT_NULL(package);
  free(bytes);
}

void test_truncated_and_missing_files_are_rejected(void) {
  size_t size = 0;
  uint8_t *bytes = read_all(FIXTURE, &size);
  NmGirwPackage *package = NULL;
  for (size_t keep = 0; keep < size; keep += 37) {
    write_all(COPY, bytes, keep ? keep : 1);
    NmStatus status = nm_girw_open(COPY, &package);
    TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR, status);
    TEST_ASSERT_NULL(package);
  }
  free(bytes);
  TEST_ASSERT_EQUAL(NM_STATUS_IO_ERROR, nm_girw_open("does_not_exist.girw", &package));
  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT, nm_girw_open(NULL, &package));
}

/* Version 1: 24-byte header, 28 bytes per entry, no checksums, no alignment. */
void test_version_1_package_is_readable(void) {
  const float blob[2] = {1.0f, 2.0f};
  uint8_t bytes[24 + 28 + 1 + 8];
  memset(bytes, 0, sizeof(bytes));
  memcpy(bytes, "GIRWPKG\0", 8);
  bytes[8] = 1;   /* version */
  bytes[12] = 1;  /* entries */
  bytes[16] = 24 + 28 + 1; /* data start */
  bytes[24] = 1;  /* name length */
  bytes[26] = 'w';
  bytes[27] = 3;  /* float32 */
  bytes[29] = 2;  /* element count */
  bytes[37] = 24 + 28 + 1; /* offset */
  bytes[45] = 8;  /* byte length */
  memcpy(bytes + 24 + 28 + 1, blob, sizeof(blob));
  write_all(COPY, bytes, sizeof(bytes));

  NmGirwPackage *package = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_girw_open(COPY, &package));
  const NmGirwEntry *entry = nm_girw_entry(package, 0);
  TEST_ASSERT_EQUAL_STRING("w", entry->name);
  TEST_ASSERT_EQUAL_FLOAT(2.0f, ((const float *)entry->data)[1]);
  nm_girw_close(package);
}

/* The C writer produces exactly the bytes GraphIR's builder wrote. */
void test_written_image_is_byte_identical_to_the_graphir_reference(void) {
  float weights[24];
  for (int i = 0; i < 24; ++i)
    weights[i] = (float)i * 0.25f - 3.0f;
  const float bias[3] = {0.5f, -1.5f, 2.0f};
  const NmGirwSource sources[2] = {
      {"layer.0.weight", NM_GIRW_FLOAT32, 24, weights},
      {"layer.0.bias", NM_GIRW_FLOAT32, 3, bias},
  };

  size_t reference_size = 0;
  uint8_t *reference = read_all(FIXTURE, &reference_size);
  size_t size = nm_girw_image_size(sources, 2);
  TEST_ASSERT_EQUAL_size_t(reference_size, size);
  uint8_t *image = aligned_alloc(64, (size + 63) / 64 * 64);
  TEST_ASSERT_NOT_NULL(image);
  TEST_ASSERT_EQUAL(NM_STATUS_OK, nm_girw_write_image(sources, 2, image, size));
  TEST_ASSERT_EQUAL_MEMORY(reference, image, size);

  TEST_ASSERT_EQUAL(NM_STATUS_INVALID_ARGUMENT,
                    nm_girw_write_image(sources, 2, image, size - 1));
  free(image);
  free(reference);
}

void test_memory_package_is_read_without_copy_and_verified(void) {
  size_t size = 0;
  uint8_t *file = read_all(FIXTURE, &size);
  uint8_t *image = aligned_alloc(64, (size + 63) / 64 * 64);
  TEST_ASSERT_NOT_NULL(image);
  memcpy(image, file, size);
  free(file);

  NmGirwPackage *package = NULL;
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_girw_open_memory(image, size, NM_GIRW_VERIFY_ALL, &package));
  const NmGirwEntry *bias = nm_girw_entry(package, 1);
  TEST_ASSERT_EQUAL_PTR(image + bias->offset, bias->data); /* points into `image` */
  TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)(uintptr_t)bias->data % 64);
  const size_t bias_offset = (size_t)bias->offset;
  nm_girw_close(package);

  image[bias_offset] ^= 0x01; /* corrupt a blob */
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_girw_open_memory(image, size, NM_GIRW_VERIFY_ALL, &package));
  /* A trusted transport may skip blob checksums; the index is still verified. */
  TEST_ASSERT_EQUAL(NM_STATUS_OK,
                    nm_girw_open_memory(image, size, NM_GIRW_VERIFY_INDEX_ONLY, &package));
  nm_girw_close(package);
  image[33] ^= 0x01;
  TEST_ASSERT_EQUAL(NM_STATUS_FORMAT_ERROR,
                    nm_girw_open_memory(image, size, NM_GIRW_VERIFY_INDEX_ONLY, &package));
  free(image);
}
