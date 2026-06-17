#pragma once

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "dist_header.h"

namespace utils
{

static inline int64_t L2SqrInt8ScalarInt64(const int8_t* pVec1,
                                           const int8_t* pVec2,
                                           std::size_t dim)
{
  int64_t res = 0;
  for (std::size_t idx = 0; idx < dim; ++idx)
  {
    const int64_t diff = static_cast<int64_t>(pVec1[idx]) -
                         static_cast<int64_t>(pVec2[idx]);
    res += diff * diff;
  }
  return res;
}

static inline float L2SqrInt8Scalar(const int8_t* pVec1, const int8_t* pVec2,
                                    std::size_t dim)
{
  return static_cast<float>(L2SqrInt8ScalarInt64(pVec1, pVec2, dim));
}

#if defined(__AVX512F__)
// Hardware reduce: single vpdpbusd-style tree-reduce; avoids store-to-load
// forwarding stall and the 16-add scalar dependency chain. Safe in int32:
// for L2 the per-lane accumulator is bounded by dim * 4 * 127 * 128 (see
// dpbusd contract), so int32 overflow requires dim > ~33M elements.
static inline int64_t HsumInt32x16AsInt64(__m512i acc)
{
  return static_cast<int64_t>(_mm512_reduce_add_epi32(acc));
}

static inline int64_t HsumInt32x16WidenToInt64(__m512i acc)
{
  const __m256i lo32 = _mm512_castsi512_si256(acc);
  const __m256i hi32 = _mm512_extracti64x4_epi64(acc, 1);
  const __m512i lo64 = _mm512_cvtepi32_epi64(lo32);
  const __m512i hi64 = _mm512_cvtepi32_epi64(hi32);
  return _mm512_reduce_add_epi64(lo64) + _mm512_reduce_add_epi64(hi64);
}
#endif

#if defined(__AVX512VNNI__)
static inline int64_t L2SqrInt8VNNIInt64(const int8_t* pVec1,
                                         const int8_t* pVec2,
                                         std::size_t dim,
                                         int32_t norm_sq_q,
                                         int32_t sum_b,
                                         int32_t norm_sq_b)
{
  __m512i acc = _mm512_setzero_si512();
  const __m512i sign_flip =
      _mm512_set1_epi32(static_cast<int>(0x80808080u));
  std::size_t idx = 0;
  for (; idx + 64 <= dim; idx += 64)
  {
    __m512i q = _mm512_loadu_si512(
        reinterpret_cast<const void*>(pVec1 + idx));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const void*>(pVec2 + idx));
    q = _mm512_xor_si512(q, sign_flip);
    acc = _mm512_dpbusd_epi32(acc, q, b);
  }
  if (idx < dim)
  {
    // Masked tail load: avoids zero-init+memcpy round-trip via stack.
    // tail in [1, 63]; bits >= tail are 0, so those byte lanes are zeroed.
    // After XOR with 0x80 in q, the zero padding becomes 128 unsigned, but
    // the corresponding b lanes are 0 — dpbusd contribution 128*0 = 0, so
    // the dot product is unaffected.
    const std::size_t tail = dim - idx;
    const __mmask64 mask =
        (static_cast<__mmask64>(1) << tail) - static_cast<__mmask64>(1);
    __m512i q = _mm512_maskz_loadu_epi8(mask, pVec1 + idx);
    const __m512i b = _mm512_maskz_loadu_epi8(mask, pVec2 + idx);
    q = _mm512_xor_si512(q, sign_flip);
    acc = _mm512_dpbusd_epi32(acc, q, b);
  }

  const int64_t unsigned_dot = HsumInt32x16AsInt64(acc);
  const int64_t dot = unsigned_dot - 128LL * static_cast<int64_t>(sum_b);
  return static_cast<int64_t>(norm_sq_q) + static_cast<int64_t>(norm_sq_b) -
         2LL * dot;
}

static inline float L2SqrInt8AVX512VNNI(const int8_t* pVec1,
                                        const int8_t* pVec2,
                                        std::size_t dim, int32_t norm_sq_q,
                                        int32_t sum_b, int32_t norm_sq_b)
{
  return static_cast<float>(L2SqrInt8VNNIInt64(
      pVec1, pVec2, dim, norm_sq_q, sum_b, norm_sq_b));
}
#endif

#if defined(__AVX512BW__)
static inline int64_t L2SqrInt8AVX512FallbackInt64(const int8_t* pVec1,
                                                   const int8_t* pVec2,
                                                   std::size_t dim)
{
  __m512i acc = _mm512_setzero_si512();
  std::size_t idx = 0;
  for (; idx + 32 <= dim; idx += 32)
  {
    const __m256i q8 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(pVec1 + idx));
    const __m256i b8 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(pVec2 + idx));
    const __m512i q16 = _mm512_cvtepi8_epi16(q8);
    const __m512i b16 = _mm512_cvtepi8_epi16(b8);
    const __m512i diff = _mm512_sub_epi16(q16, b16);
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff, diff));
  }

  int64_t res = HsumInt32x16AsInt64(acc);
  for (; idx < dim; ++idx)
  {
    const int64_t diff = static_cast<int64_t>(pVec1[idx]) -
                         static_cast<int64_t>(pVec2[idx]);
    res += diff * diff;
  }
  return res;
}

static inline float L2SqrInt8AVX512Fallback(const int8_t* pVec1,
                                            const int8_t* pVec2,
                                            std::size_t dim)
{
  return static_cast<float>(
      L2SqrInt8AVX512FallbackInt64(pVec1, pVec2, dim));
}
#endif

static inline float L2SqrInt8SIMD(const int8_t* pVec1, const int8_t* pVec2,
                                  std::size_t dim, int32_t norm_sq_q,
                                  int32_t sum_b, int32_t norm_sq_b)
{
#if defined(__AVX512VNNI__)
  return L2SqrInt8AVX512VNNI(pVec1, pVec2, dim, norm_sq_q, sum_b, norm_sq_b);
#elif defined(__AVX512BW__)
  (void)norm_sq_q;
  (void)sum_b;
  (void)norm_sq_b;
  return L2SqrInt8AVX512Fallback(pVec1, pVec2, dim);
#else
  (void)norm_sq_q;
  (void)sum_b;
  (void)norm_sq_b;
  return L2SqrInt8Scalar(pVec1, pVec2, dim);
#endif
}

static inline int64_t L2SqrUint8ScalarInt64(const uint8_t* pVec1,
                                            const uint8_t* pVec2,
                                            std::size_t dim)
{
  int64_t res = 0;
  for (std::size_t idx = 0; idx < dim; ++idx)
  {
    const int64_t diff = static_cast<int64_t>(pVec1[idx]) -
                         static_cast<int64_t>(pVec2[idx]);
    res += diff * diff;
  }
  return res;
}

static inline float L2SqrUint8Scalar(const uint8_t* pVec1,
                                     const uint8_t* pVec2,
                                     std::size_t dim)
{
  return static_cast<float>(L2SqrUint8ScalarInt64(pVec1, pVec2, dim));
}

#if defined(__AVX512VNNI__)
static inline int64_t L2SqrUint8VNNIInt64(const uint8_t* pVec1,
                                          const uint8_t* pVec2,
                                          std::size_t dim,
                                          int32_t norm_sq_q,
                                          int32_t sum_b,
                                          int32_t norm_sq_b)
{
  __m512i acc = _mm512_setzero_si512();
  const __m512i sign_flip =
      _mm512_set1_epi32(static_cast<int>(0x80808080u));
  std::size_t idx = 0;
  for (; idx + 64 <= dim; idx += 64)
  {
    __m512i q = _mm512_loadu_si512(
        reinterpret_cast<const void*>(pVec1 + idx));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const void*>(pVec2 + idx));
    q = _mm512_xor_si512(q, sign_flip);
    acc = _mm512_dpbusd_epi32(acc, b, q);
  }
  if (idx < dim)
  {
    const std::size_t tail = dim - idx;
    const __mmask64 mask =
        (static_cast<__mmask64>(1) << tail) - static_cast<__mmask64>(1);
    __m512i q = _mm512_maskz_loadu_epi8(mask, pVec1 + idx);
    const __m512i b = _mm512_maskz_loadu_epi8(mask, pVec2 + idx);
    q = _mm512_xor_si512(q, sign_flip);
    acc = _mm512_dpbusd_epi32(acc, b, q);
  }

  const int64_t unsigned_dot = HsumInt32x16WidenToInt64(acc);
  const int64_t dot = unsigned_dot + 128LL * static_cast<int64_t>(sum_b);
  return static_cast<int64_t>(norm_sq_q) + static_cast<int64_t>(norm_sq_b) -
         2LL * dot;
}

static inline float L2SqrUint8AVX512VNNI(const uint8_t* pVec1,
                                         const uint8_t* pVec2,
                                         std::size_t dim, int32_t norm_sq_q,
                                         int32_t sum_b, int32_t norm_sq_b)
{
  return static_cast<float>(L2SqrUint8VNNIInt64(
      pVec1, pVec2, dim, norm_sq_q, sum_b, norm_sq_b));
}
#endif

#if defined(__AVX512BW__)
static inline int64_t L2SqrUint8AVX512FallbackInt64(const uint8_t* pVec1,
                                                    const uint8_t* pVec2,
                                                    std::size_t dim)
{
  __m512i acc = _mm512_setzero_si512();
  std::size_t idx = 0;
  for (; idx + 32 <= dim; idx += 32)
  {
    const __m256i q8 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(pVec1 + idx));
    const __m256i b8 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(pVec2 + idx));
    const __m512i q16 = _mm512_cvtepu8_epi16(q8);
    const __m512i b16 = _mm512_cvtepu8_epi16(b8);
    const __m512i diff = _mm512_sub_epi16(q16, b16);
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff, diff));
  }

  int64_t res = HsumInt32x16WidenToInt64(acc);
  for (; idx < dim; ++idx)
  {
    const int64_t diff = static_cast<int64_t>(pVec1[idx]) -
                         static_cast<int64_t>(pVec2[idx]);
    res += diff * diff;
  }
  return res;
}

static inline float L2SqrUint8AVX512Fallback(const uint8_t* pVec1,
                                             const uint8_t* pVec2,
                                             std::size_t dim)
{
  return static_cast<float>(
      L2SqrUint8AVX512FallbackInt64(pVec1, pVec2, dim));
}
#endif

static inline int64_t L2SqrUint8SIMDInt64(const uint8_t* pVec1,
                                          const uint8_t* pVec2,
                                          std::size_t dim,
                                          int32_t norm_sq_q, int32_t sum_b,
                                          int32_t norm_sq_b)
{
#if defined(__AVX512VNNI__)
  return L2SqrUint8VNNIInt64(pVec1, pVec2, dim, norm_sq_q, sum_b, norm_sq_b);
#elif defined(__AVX512BW__)
  (void)norm_sq_q;
  (void)sum_b;
  (void)norm_sq_b;
  return L2SqrUint8AVX512FallbackInt64(pVec1, pVec2, dim);
#else
  (void)norm_sq_q;
  (void)sum_b;
  (void)norm_sq_b;
  return L2SqrUint8ScalarInt64(pVec1, pVec2, dim);
#endif
}

static inline float L2SqrUint8SIMD(const uint8_t* pVec1, const uint8_t* pVec2,
                                   std::size_t dim, int32_t norm_sq_q,
                                   int32_t sum_b, int32_t norm_sq_b)
{
  return static_cast<float>(
      L2SqrUint8SIMDInt64(pVec1, pVec2, dim, norm_sq_q, sum_b, norm_sq_b));
}

static inline std::size_t align_up_64(const std::size_t value)
{
  constexpr std::size_t kAlignment = 64;
  return ((value + kAlignment - 1) / kAlignment) * kAlignment;
}

static inline float cell_lb_batch(
    const float* cell_lo, const float* cell_hi, float res_lo, float res_hi,
    const float* q_p, float q_r_norm, std::size_t effective_rank)
{
  float box_term_sq = 0.0f;
  for (std::size_t axis = 0; axis < effective_rank; ++axis)
  {
    const float below = std::max(cell_lo[axis] - q_p[axis], 0.0f);
    const float above = std::max(q_p[axis] - cell_hi[axis], 0.0f);
    const float delta = std::max(below, above);
    box_term_sq += delta * delta;
  }
  const float res_below = std::max(res_lo - q_r_norm, 0.0f);
  const float res_above = std::max(q_r_norm - res_hi, 0.0f);
  const float res_delta = std::max(res_below, res_above);
  return std::sqrt(box_term_sq + res_delta * res_delta);
}

#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)

// Adapted from
// https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
static inline __m128 MaskedReadFloat(const std::size_t dim, const float *data)
{
  assert(0 <= dim && dim < 4);
  ALIGNED(16) float buf[4] = {0, 0, 0, 0};
  switch (dim)
  {
    case 3:
      buf[2] = data[2];
    case 2:
      buf[1] = data[1];
    case 1:
      buf[0] = data[0];
  }
  return _mm_load_ps(buf);
}

static inline __m128i MaskedReadInt(const std::size_t dim, const int *data)
{
  assert(0 <= dim && dim < 4);
  ALIGNED(16) int buf[4] = {0, 0, 0, 0};
  switch (dim)
  {
    case 3:
      buf[2] = data[2];
    case 2:
      buf[1] = data[1];
    case 1:
      buf[0] = data[0];
  }
  return _mm_load_si128((__m128i *)buf);
}

// Adapted from
// https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av
static float HsumFloat128(__m128 x)
{
  //    __m128 h64 = _mm_unpackhi_ps(x, x);
  __m128 h64 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
  __m128 sum64 = _mm_add_ps(h64, x);
  __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(0, 1, 2, 3));
  __m128 sum32 = _mm_add_ps(sum64, h32);
  return _mm_cvtss_f32(sum32);
}

static int HsumInt128(__m128i x)
{
  __m128i hi64 = _mm_unpackhi_epi64(x, x);
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);
}

// Adapted from
// https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
#if defined(USE_AVX512)
static float L2SqrFloatAVX512(const void *pVec1v, const void *pVec2v,
                              const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m512 mx512, my512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16)
  {
    mx512 = _mm512_loadu_ps(pVec1);
    pVec1 += 16;
    my512 = _mm512_loadu_ps(pVec2);
    pVec2 += 16;
    diff512 = _mm512_sub_ps(mx512, my512);
    sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8)
  {
    __m256 mx256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 my256 = _mm256_loadu_ps(pVec2);
    pVec2 += 8;
    __m256 diff256 = _mm256_sub_ps(mx256, my256);
    sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 mx128, my128, diff128;

  if (dim >= 4)
  {
    mx128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    my128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    mx128 = MaskedReadFloat(dim, pVec1);
    my128 = MaskedReadFloat(dim, pVec2);
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
  }
  return HsumFloat128(sum128);
}
#endif  // USE_AVX512

#if defined(USE_AVX)
static float L2SqrFloatAVX(const void *pVec1v, const void *pVec2v,
                           const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m256 sum256 = _mm256_setzero_ps();

  while (dim >= 8)
  {
    __m256 mx256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 my256 = _mm256_loadu_ps(pVec2);
    pVec2 += 8;
    __m256 diff256 = _mm256_sub_ps(mx256, my256);
    sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 mx128, my128, diff128;

  if (dim >= 4)
  {
    mx128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    my128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    mx128 = MaskedReadFloat(dim, pVec1);
    my128 = MaskedReadFloat(dim, pVec2);
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
  }
  return HsumFloat128(sum128);
}
#endif  // USE_AVX

#if defined(USE_SSE)
static float L2SqrFloatSSE(const void *pVec1v, const void *pVec2v,
                           const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m128 sum128 = _mm_setzero_ps();
  __m128 mx128, my128, diff128;

  while (dim >= 4)
  {
    mx128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    my128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    mx128 = MaskedReadFloat(dim, pVec1);
    my128 = MaskedReadFloat(dim, pVec2);
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
  }
  return HsumFloat128(sum128);
}
#endif  // USE_SSE

#if defined(USE_AVX512)
static float InnerProductFloatAVX512(const void *pVec1v, const void *pVec2v,
                                     const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  // std::size_t dim = *((std::size_t *) dim_ptr);
  std::size_t dim = 200;

  __m512 x512, y512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16)
  {
    x512 = _mm512_loadu_ps(pVec1);
    pVec1 += 16;
    y512 = _mm512_loadu_ps(pVec2);
    pVec2 += 16;
    sum512 = _mm512_fmadd_ps(x512, y512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 y256 = _mm256_loadu_ps(pVec2);
    pVec2 += 8;
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128, y128;

  if (dim >= 4)
  {
    x128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    y128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pVec1);
    y128 = MaskedReadFloat(dim, pVec2);
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return -HsumFloat128(sum128);
}

static float InnerProductFloatAVX512Dim20(const void *pVec1v,
                                          const void *pVec2v,
                                          const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  // std::size_t dim = *((std::size_t *) dim_ptr);
  std::size_t dim = 20;

  __m512 x512, y512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16)
  {
    x512 = _mm512_loadu_ps(pVec1);
    pVec1 += 16;
    y512 = _mm512_loadu_ps(pVec2);
    pVec2 += 16;
    sum512 = _mm512_fmadd_ps(x512, y512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 y256 = _mm256_loadu_ps(pVec2);
    pVec2 += 8;
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128, y128;

  if (dim >= 4)
  {
    x128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    y128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pVec1);
    y128 = MaskedReadFloat(dim, pVec2);
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return -HsumFloat128(sum128);
}

static float InnerProductFloatAVX512Hp(const void *pVec1v, const void *pVec2v,
                                       const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  uint8_t *pVec2 = (uint8_t *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m512 x512, y512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16)
  {
    x512 = _mm512_loadu_ps(pVec1);
    pVec1 += 16;
    y512 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)pVec2));
    pVec2 += 32;
    sum512 = _mm512_fmadd_ps(x512, y512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 y256 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)pVec2));
    pVec2 += 16;
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128, y128;

  // if (dim >= 4) {
  //     x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
  //     y128 = _mm_cvtph_ps(_mm_loadu_si((const __m128i*)pVec2)); pVec2 += 8;
  //     sum128 = _mm_fmadd_ps(x128, y128, sum128);
  //     dim -= 4;
  // }

  // if (dim > 0) {
  //     x128 = MaskedReadFloat(dim, pVec1);
  //     y128 = MaskedReadFloat(dim, pVec2);
  //     sum128 = _mm_fmadd_ps(x128, y128, sum128);
  // }
  return -HsumFloat128(sum128);
}

static float InnerProductFloatAVX512HpDim200(const void *pVec1v,
                                             const void *pVec2v,
                                             const void *dim_ptr)
{
  float *x = (float *)pVec1v;
  uint8_t *code = (uint8_t *)pVec2v;
  __m512 sum512 = _mm512_setzero_ps();
  for (size_t i = 0; i < 192; i += 16)
  {
    // __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2*i));
    __m256i codei = _mm256_lddqu_si256((const __m256i *)(code + 2 * i));
    __m512 code512 = _mm512_cvtph_ps(codei);
    __m512 q512 = _mm512_loadu_ps(x + i);
    sum512 = _mm512_fmadd_ps(code512, q512, sum512);
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));
  __m128i c128i = _mm_loadu_si128((const __m128i *)(code + 384));
  __m256 c256 = _mm256_cvtph_ps(c128i);
  __m256 q256 = _mm256_loadu_ps(x + 192);
  sum256 = _mm256_fmadd_ps(c256, q256, sum256);
  sum256 = _mm256_hadd_ps(sum256, sum256);
  sum256 = _mm256_hadd_ps(sum256, sum256);
  return -_mm_cvtss_f32(_mm256_castps256_ps128(sum256)) -
         _mm_cvtss_f32(_mm256_extractf128_ps(sum256, 1));
}
#endif  // USE_AVX512

#if defined(USE_AVX)
static float InnerProductFloatAVX(const void *pVec1v, const void *pVec2v,
                                  const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m256 sum256 = _mm256_setzero_ps();

  while (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pVec1);
    pVec1 += 8;
    __m256 y256 = _mm256_loadu_ps(pVec2);
    pVec2 += 8;
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128, y128;

  if (dim >= 4)
  {
    x128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    y128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pVec1);
    y128 = MaskedReadFloat(dim, pVec2);
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return HsumFloat128(sum128);
}
#endif  // USE_AVX

#if defined(USE_SSE)
static float InnerProductFloatSSE(const void *pVec1v, const void *pVec2v,
                                  const void *dim_ptr)
{
  float *pVec1 = (float *)pVec1v;
  float *pVec2 = (float *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m128 sum128 = _mm_setzero_ps();
  __m128 x128, y128;

  while (dim >= 4)
  {
    x128 = _mm_loadu_ps(pVec1);
    pVec1 += 4;
    y128 = _mm_loadu_ps(pVec2);
    pVec2 += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pVec1);
    y128 = MaskedReadFloat(dim, pVec2);
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return HsumFloat128(sum128);
}
#endif  // USE_SSE

#if defined(USE_AVX512)
static float NormSqrFloatAVX512(const void *pVec, const void *dim_ptr)
{
  float *pV = (float *)pVec;
  std::size_t dim = *((std::size_t *)dim_ptr);
  __m512 x512;
  __m512 res512 = _mm512_setzero_ps();

  while (dim >= 16)
  {
    x512 = _mm512_loadu_ps(pV);
    pV += 16;
    res512 = _mm512_fmadd_ps(x512, x512, res512);
    dim -= 16;
  }
  __m256 res256 = _mm256_add_ps(_mm512_castps512_ps256(res512),
                                _mm512_extractf32x8_ps(res512, 1));

  if (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pV);
    pV += 8;
    res256 = _mm256_fmadd_ps(x256, x256, res256);
    dim -= 8;
  }
  __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256),
                             _mm256_extractf128_ps(res256, 1));
  __m128 x128;

  if (dim >= 4)
  {
    x128 = _mm_loadu_ps(pV);
    pV += 4;
    res128 = _mm_fmadd_ps(x128, x128, res128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pV);
    res128 = _mm_fmadd_ps(x128, x128, res128);
  }
  return HsumFloat128(res128);
}
#endif  // USE_AVX512

#if defined(USE_AVX)
static float NormSqrFloatAVX(const void *pVec, const void *dim_ptr)
{
  float *pV = (float *)pVec;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m256 res256 = _mm256_setzero_ps();
  while (dim >= 8)
  {
    __m256 x256 = _mm256_loadu_ps(pV);
    pV += 8;
    res256 = _mm256_fmadd_ps(x256, x256, res256);
    dim -= 8;
  }
  __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256),
                             _mm256_extractf128_ps(res256, 1));
  __m128 x128;

  if (dim >= 4)
  {
    x128 = _mm_loadu_ps(pV);
    pV += 4;
    res128 = _mm_fmadd_ps(x128, x128, res128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pV);
    res128 = _mm_fmadd_ps(x128, x128, res128);
  }
  return HsumFloat128(res128);
}
#endif  // USE_AVX

#if defined(USE_SSE)
static float NormSqrFloatSSE(const void *pVec, const void *dim_ptr)
{
  float *pV = (float *)pVec;
  std::size_t dim = *((std::size_t *)dim_ptr);

  __m128 res128 = _mm_setzero_ps();
  __m128 x128;

  while (dim >= 4)
  {
    x128 = _mm_loadu_ps(pV);
    pV += 4;
    res128 = _mm_fmadd_ps(x128, x128, res128);
    dim -= 4;
  }

  if (dim > 0)
  {
    x128 = MaskedReadFloat(dim, pV);
    res128 = _mm_fmadd_ps(x128, x128, res128);
  }
  return HsumFloat128(res128);
}
#endif  // USE_SSE

#if defined(USE_AVX512)
static float L2SqrWithNormAVX512(const void *pVec1, const void *pVec2,
                                 const void *dim_ptr, const void *norm_ptr)
{
  return *((float *)norm_ptr) -
         2 * InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
}
#endif  // USE_AVX512

#if defined(USE_AVX)
static float L2SqrWithNormAVX(const void *pVec1, const void *pVec2,
                              const void *dim_ptr, const void *norm_ptr)
{
  return *((float *)norm_ptr) - 2 * InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
}
#endif  // USE_AVX

#if defined(USE_SSE)
static float L2SqrWithNormSSE(const void *pVec1, const void *pVec2,
                              const void *dim_ptr, const void *norm_ptr)
{
  return *((float *)norm_ptr) - 2 * InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
}
#endif  // use_SSE

static float L2SqrSIMD(const void *pVec1, const void *pVec2,
                       const void *dim_ptr)
{
#if defined(USE_AVX512)
  return L2SqrFloatAVX512(pVec1, pVec2, dim_ptr);
#elif defined(USE_AVX)
  return L2SqrFloatAVX(pVec1, pVec2, dim_ptr);
#elif defined(USE_SSE)
  return L2SqrFloatSSE(pVec1, pVec2, dim_ptr);
#endif
}

static float InverseL2SqrSIMD(const void *pVec1, const void *pVec2,
                              const void *dim_ptr)
{
#if defined(USE_AVX512)
  return -L2SqrFloatAVX512(pVec1, pVec2, dim_ptr);
#elif defined(USE_AVX)
  return -L2SqrFloatAVX(pVec1, pVec2, dim_ptr);
#elif defined(USE_SSE)
  return -L2SqrFloatSSE(pVec1, pVec2, dim_ptr);
#endif
}

static float InnerProductSIMD(const void *pVec1, const void *pVec2,
                              const void *dim_ptr)
{
#if defined(USE_AVX512)
  return InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
#elif defined(USE_AVX)
  return InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
#elif defined(USE_SSE)
  return InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
#endif
}

static float InverseInnerProductSIMD(const void *pVec1, const void *pVec2,
                                     const void *dim_ptr)
{
#if defined(USE_AVX512)
  return -InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
#elif defined(USE_AVX)
  return -InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
#elif defined(USE_SSE)
  return -InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
#endif
}

static float AbsInnerProductSIMD(const void *pVec1, const void *pVec2,
                                 const void *dim_ptr)
{
#if defined(USE_AVX512)
  return fabs(InnerProductFloatAVX512(pVec1, pVec2, dim_ptr));
#elif defined(USE_AVX)
  return fabs(InnerProductFloatAVX(pVec1, pVec2, dim_ptr));
#elif defined(USE_SSE)
  return fabs(InnerProductFloatSSE(pVec1, pVec2, dim_ptr));
#endif
}

#endif  // defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)

template <typename T>
static T L2SqrNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr)
{
  T *pVec1 = (T *)pVec1v;
  T *pVec2 = (T *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  T diff, res = 0;
#pragma omp simd
  for (auto idx = 0; idx < dim; ++idx)
  {
    diff = pVec1[idx] - pVec2[idx];
    res += diff * diff;
  }
  return res;
}

template <typename T>
static float IPNaive(const void *pVec1v, const void *pVec2v,
                     const void *dim_ptr)
{
  T *pVec1 = (T *)pVec1v;
  T *pVec2 = (T *)pVec2v;
  std::size_t dim = *((std::size_t *)dim_ptr);

  T res = 0;
#pragma omp simd
  for (auto idx = 0; idx < dim; ++idx)
  {
    res += pVec1[idx] * pVec2[idx];
  }
  return res;
}

template <typename T>
static T NormSqr(const void *pVec, const void *dim_ptr)
{
  T *pV = (T *)pVec;
  std::size_t dim = *((std::size_t *)dim_ptr);

  T res = 0;
#pragma omp simd
  for (auto idx = 0; idx < dim; ++idx)
  {
    res += pV[idx] * pV[idx];
  }
  return res;
}

template <typename T>
static T NormSqrT(const void *pVec, const void *dim_ptr)
{
  T *pV = (T *)pVec;
  std::size_t dim = *((std::size_t *)dim_ptr);

  T res = 0;
#pragma omp simd
  for (auto idx = 0; idx < dim; ++idx)
  {
    res += pV[idx] * pV[idx];
  }
  return sqrt(res);
}

template <typename T>
static T P2HNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr)
{
  return fabs(IPNaive<T>(pVec1v, pVec2v, dim_ptr));
}

static float InnerProduct(const void *pVec1v, const void *pVec2v,
                          const void *dim_ptr)
{
  float res = 0;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
  res = InnerProductSIMD(pVec1v, pVec2v, dim_ptr);
#elif
  std::size_t dim = *((std::size_t *)dim_ptr);
  for (std::size_t i = 0; i < dim; ++i)
  {
    res += ((float *)pVec1v)[i] * ((float *)pVec2v)[i];
  }
#endif
  return res;
}

static float InverseInnerProduct(const void *pVec1v, const void *pVec2v,
                                 const void *dim_ptr)
{
  return -InnerProduct(pVec1v, pVec2v, dim_ptr);
}

static float AbsInnerProduct(const void *pVec1v, const void *pVec2v,
                             const void *dim_ptr)
{
  return fabs(InnerProduct(pVec1v, pVec2v, dim_ptr));
}

static float InverseAbsInnerProduct(const void *pVec1v, const void *pVec2v,
                                    const void *dim_ptr)
{
  return -fabs(InnerProduct(pVec1v, pVec2v, dim_ptr));
}

static float L2Sqr(const void *pVec1v, const void *pVec2v, const void *dim_ptr)
{
  float res = 0;
#if defined(USE_AVX512) || defined(USE_AVX512) || defined(USE_SSE)
  res = L2SqrSIMD(pVec1v, pVec2v, dim_ptr);
#elif
  std::size_t dim = *((std::size_t *)dim_ptr);

  float diff = 0;
  for (std::size_t i = 0; i < dim; ++i)
  {
    diff = ((float *)pVec1v)[i] - ((float *)pVec2v)[i];
    res += diff * diff;
  }
#endif

  return res;
}

static float InverseL2Sqr(const void *pVec1v, const void *pVec2v,
                          const void *dim_ptr)
{
  return -L2Sqr(pVec1v, pVec2v, dim_ptr);
}

}  // namespace utils
