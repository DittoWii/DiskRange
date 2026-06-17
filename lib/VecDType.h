#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace range_search_config
{

constexpr uint32_t kVecDTypeFloat32 = 0u;
constexpr uint32_t kVecDTypeInt8 = 1u;
constexpr uint32_t kVecDTypeUint8Reserved = 2u;
constexpr uint32_t kVecDTypeFp16Reserved = 3u;
constexpr size_t kInt8DimMetadataLimit = 131071;
constexpr size_t kUint8DimMetadataLimit = 33025;

inline std::string vec_dtype_to_string(uint32_t dtype)
{
  switch (dtype)
  {
    case kVecDTypeFloat32:
      return "float32";
    case kVecDTypeInt8:
      return "int8";
    case kVecDTypeUint8Reserved:
      return "uint8";
    case kVecDTypeFp16Reserved:
      return "fp16";
    default:
      return "unknown(" + std::to_string(dtype) + ")";
  }
}

inline bool is_fbin_dtype_enum_value(uint32_t dtype)
{
  return dtype <= kVecDTypeFp16Reserved;
}

inline bool is_supported_vec_dtype(uint32_t dtype)
{
  return dtype == kVecDTypeFloat32 || dtype == kVecDTypeInt8 ||
         dtype == kVecDTypeUint8Reserved;
}

inline bool is_byte_dtype(uint32_t dtype)
{
  return dtype == kVecDTypeInt8 || dtype == kVecDTypeUint8Reserved;
}

inline size_t bytes_per_element_for_fbin_dtype(uint32_t dtype)
{
  switch (dtype)
  {
    case kVecDTypeFloat32:
      return sizeof(float);
    case kVecDTypeInt8:
      return sizeof(int8_t);
    case kVecDTypeUint8Reserved:
      return sizeof(uint8_t);
    case kVecDTypeFp16Reserved:
      return 2;
    default:
      throw std::runtime_error("invalid fbin dtype: " + std::to_string(dtype));
  }
}

inline uint32_t parse_vec_dtype_string(const std::string& name)
{
  if (name == "float32") return kVecDTypeFloat32;
  if (name == "int8")    return kVecDTypeInt8;
  if (name == "uint8")   return kVecDTypeUint8Reserved;
  if (name == "fp16")    return kVecDTypeFp16Reserved;
  throw std::runtime_error("unrecognized vec_dtype string: " + name);
}

inline uint32_t parse_vec_dtype_token_strict(const std::string& token)
{
  if (token == "float32") return kVecDTypeFloat32;
  if (token == "int8") return kVecDTypeInt8;
  if (token == "uint8") return kVecDTypeUint8Reserved;
  if (!token.empty() && token[0] == '-')
  {
    throw std::runtime_error(
        "vec_dtype must be one of 0/1/2 or float32/int8/uint8: " + token);
  }
  try
  {
    size_t idx = 0;
    const unsigned long raw = std::stoul(token, &idx);
    if (idx != token.size() || raw > kVecDTypeUint8Reserved)
    {
      throw std::runtime_error(
          "vec_dtype must be one of 0/1/2 or float32/int8/uint8: " + token);
    }
    return static_cast<uint32_t>(raw);
  }
  catch (const std::invalid_argument&)
  {
    throw std::runtime_error(
        "vec_dtype must be one of 0/1/2 or float32/int8/uint8: " + token);
  }
  catch (const std::out_of_range&)
  {
    throw std::runtime_error(
        "vec_dtype must be one of 0/1/2 or float32/int8/uint8: " + token);
  }
}

}  // namespace range_search_config
