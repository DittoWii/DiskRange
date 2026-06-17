#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace utils
{

static inline void cast_int8_to_float(const int8_t* src, size_t count,
                                      float* dst)
{
  for (size_t i = 0; i < count; ++i)
  {
    dst[i] = static_cast<float>(src[i]);
  }
}

static inline void cast_uint8_to_float(const uint8_t* src, size_t count,
                                       float* dst)
{
  for (size_t i = 0; i < count; ++i)
  {
    dst[i] = static_cast<float>(src[i]);
  }
}

static inline void cast_byte_payload_to_float(uint32_t vec_dtype,
                                              const uint8_t* src,
                                              size_t count, float* dst)
{
  if (vec_dtype == 1u)
  {
    cast_int8_to_float(reinterpret_cast<const int8_t*>(src), count, dst);
    return;
  }
  if (vec_dtype == 2u)
  {
    cast_uint8_to_float(src, count, dst);
    return;
  }
  throw std::runtime_error("cast_byte_payload_to_float requires byte dtype; got " +
                           std::to_string(vec_dtype));
}

}  // namespace utils
