/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* f32 <-> bfloat16 / IEEE binary16, rounding to nearest even. Same algorithms
 * as GraphIR's halfprec.rs, so both sides produce identical bits. */
#ifndef NM_HALFPREC_H
#define NM_HALFPREC_H

#include <stdint.h>
#include <string.h>

static inline uint32_t nm_bits_of(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline float nm_float_of(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline uint16_t nm_f32_to_bf16(float value) {
  uint32_t bits = nm_bits_of(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) /* NaN: keep it quiet */
    return (uint16_t)((bits >> 16) | 0x0040U);
  return (uint16_t)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

static inline float nm_bf16_to_f32(uint16_t bits) {
  return nm_float_of((uint32_t)bits << 16);
}

static inline uint16_t nm_f32_to_f16(float value) {
  uint32_t bits = nm_bits_of(value);
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000U);
  int32_t exponent = (int32_t)((bits >> 23) & 0xffU);
  uint32_t mantissa = bits & 0x007fffffU;
  if (exponent == 0xff)
    return (uint16_t)(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
  int32_t half_exponent = exponent - 127 + 15;
  if (half_exponent >= 0x1f)
    return (uint16_t)(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    uint32_t full = mantissa | 0x00800000U;
    uint32_t shift = (uint32_t)(14 - half_exponent);
    uint32_t half = full >> shift;
    uint32_t remainder = full & ((1U << shift) - 1U);
    uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U)))
      half += 1U;
    return (uint16_t)(sign | half);
  }
  uint32_t half = ((uint32_t)half_exponent << 10) | (mantissa >> 13);
  uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U)))
    half += 1U;
  return (uint16_t)(sign | half);
}

static inline float nm_f16_to_f32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000U) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fU;
  uint32_t mantissa = bits & 0x03ffU;
  uint32_t out;
  if (exponent == 0 && mantissa == 0) {
    out = sign;
  } else if (exponent == 0) {
    int32_t e = 127 - 15 + 1;
    while (!(mantissa & 0x0400U)) {
      mantissa <<= 1;
      --e;
    }
    out = sign | ((uint32_t)e << 23) | ((mantissa & 0x03ffU) << 13);
  } else if (exponent == 0x1f) {
    out = sign | 0x7f800000U | (mantissa ? 0x00400000U | (mantissa << 13) : 0U);
  } else {
    out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return nm_float_of(out);
}

#endif /* NM_HALFPREC_H */
