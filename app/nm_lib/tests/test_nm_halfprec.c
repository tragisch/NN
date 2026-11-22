/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* 16-bit float conversions: every pattern round-trips, rounding is to nearest
 * even, and the known values match GraphIR's halfprec.rs tests. */
#include "nm_halfprec.h"
#include <math.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_bf16_every_pattern_round_trips(void) {
  for (uint32_t bits = 0; bits <= 0xffffU; ++bits) {
    float value = nm_bf16_to_f32((uint16_t)bits);
    if (isnan(value))
      continue;
    TEST_ASSERT_EQUAL_HEX16((uint16_t)bits, nm_f32_to_bf16(value));
  }
}

void test_f16_every_pattern_round_trips(void) {
  for (uint32_t bits = 0; bits <= 0xffffU; ++bits) {
    float value = nm_f16_to_f32((uint16_t)bits);
    if (isnan(value)) {
      TEST_ASSERT_TRUE(isnan(nm_f16_to_f32(nm_f32_to_f16(value))));
      continue;
    }
    TEST_ASSERT_EQUAL_HEX16((uint16_t)bits, nm_f32_to_f16(value));
  }
}

void test_rounding_goes_to_nearest_even(void) {
  TEST_ASSERT_EQUAL_HEX16(0x3f80, nm_f32_to_bf16(1.0f + ldexpf(1.0f, -8)));
  TEST_ASSERT_EQUAL_HEX16(0x3f82, nm_f32_to_bf16(1.0f + 3.0f * ldexpf(1.0f, -8)));
  TEST_ASSERT_EQUAL_HEX16(0x7f80, nm_f32_to_bf16(3.4028235e38f)); /* overflow */
  TEST_ASSERT_EQUAL_HEX16(0x3c00, nm_f32_to_f16(1.0f));
  TEST_ASSERT_EQUAL_HEX16(0xc000, nm_f32_to_f16(-2.0f));
  TEST_ASSERT_EQUAL_HEX16(0x7bff, nm_f32_to_f16(65504.0f));
  TEST_ASSERT_EQUAL_HEX16(0x7c00, nm_f32_to_f16(65520.0f));
  TEST_ASSERT_EQUAL_HEX16(0x0001, nm_f32_to_f16(ldexpf(1.0f, -24)));
  TEST_ASSERT_EQUAL_HEX16(0x0000, nm_f32_to_f16(ldexpf(1.0f, -25)));
  TEST_ASSERT_EQUAL_HEX16(0x3c00, nm_f32_to_f16(1.0f + ldexpf(1.0f, -11)));
  TEST_ASSERT_EQUAL_HEX16(0x3c02, nm_f32_to_f16(1.0f + 3.0f * ldexpf(1.0f, -11)));
}
