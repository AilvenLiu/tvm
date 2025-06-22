/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ptx.cc
 */

#include "ptx.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

// PTX related data structures and functions.
namespace ptx {

/*!
 * \brief PTX data type.
 * \note
 * PTX fundamental data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types
 * PTX matrix data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-data-types
 */
enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat4_e2m1fn = 10,
  kFloat6_e2m3fn = 11,
  kFloat6_e3m2fn = 12,
  kFloat8_e4m3fn = 13,
  kFloat8_e4m3fnuz = 14,
  kFloat8_e5m2 = 15,
  kFloat8_e8m0fnu = 16,
  kFloat16 = 17,
  kBFloat16 = 18,
  kFloat16x2 = 19,
  kFloat32 = 20,
  kTensorFloat32 = 21,
  kFloat64 = 22,
  kBit1 = 23,
  kBit8 = 24,
  kBit16 = 25,
  kBit32 = 26,
  kBit64 = 27,
};

static const char* dtype_str[] = {".s4",    ".u4",   ".s8",    ".u8",   ".s16",  ".u16",   ".s32",
                                  ".u32",   ".s64",  ".u64",   ".e2m1", ".e2m3", ".e3m2",  ".e4m3",
                                  ".ue4m3", ".e5m2", ".ue8m0", ".f16",  ".bf16", ".f16x2", ".f32",
                                  ".tf32",  ".f64",  ".b1",    ".b8",   ".b16",  ".b32",   ".b64"};
static const uint32_t num_bits[] = {4, 4, 8, 8,  16, 16, 32, 32, 64, 64, 4, 6,  6,  8,
                                    7, 8, 8, 16, 16, 32, 32, 32, 64, 1,  8, 16, 32, 64};

/*!
 * \brief Create PTX data type from string.
 */
inline DataType DTypeFromString(const std::string str) {
  if (str == "int4" || str == ".s4") {
    return DataType::kInt4;
  } else if (str == "uint4" || str == ".u4") {
    return DataType::kUInt4;
  } else if (str == "int8" || str == ".s8") {
    return DataType::kInt8;
  } else if (str == "uint8" || str == ".u8") {
    return DataType::kUInt8;
  } else if (str == "int16" || str == ".s16") {
    return DataType::kInt16;
  } else if (str == "uint16" || str == ".u16") {
    return DataType::kUInt16;
  } else if (str == "int32" || str == ".s32") {
    return DataType::kInt32;
  } else if (str == "uint32" || str == ".u32") {
    return DataType::kUInt32;
  } else if (str == "int64" || str == ".s64") {
    return DataType::kInt64;
  } else if (str == "uint64" || str == ".u64") {
    return DataType::kUInt64;
  } else if (str == "e2m1" || str == ".e2m1" || str == "float4_e2m1fn") {
    return DataType::kFloat4_e2m1fn;
  } else if (str == "e2m3" || str == ".e2m3" || str == "float6_e2m3fn") {
    return DataType::kFloat6_e2m3fn;
  } else if (str == "e3m2" || str == ".e3m2" || str == "float6_e3m2fn") {
    return DataType::kFloat6_e3m2fn;
  } else if (str == "e4m3" || str == ".e4m3" || str == "float8_e4m3fn") {
    return DataType::kFloat8_e4m3fn;
  } else if (str == "float8_e4m3fnuz" || str == "float8_e4m3b11fnuz") {
    return DataType::kFloat8_e4m3fnuz;
  } else if (str == "e5m2" || str == ".e5m2" || str == "float8_e5m2" || str == "float8_e5m2fn" ||
             str == "float8_e5m2fnuz") {
    return DataType::kFloat8_e5m2;
  } else if (str == "ue8m0" || str == ".ue8m0" || str == "float8_e8m0fnu") {
    return DataType::kFloat8_e8m0fnu;
  } else if (str == "float16" || str == "fp16" || str == ".f16") {
    return DataType::kFloat16;
  } else if (str == "bfloat16" || str == "bf16") {
    return DataType::kBFloat16;
  } else if (str == ".f16x2") {
    return DataType::kFloat16x2;
  } else if (str == "float32" || str == "fp32" || str == ".f32") {
    return DataType::kFloat32;
  } else if (str == "tf32") {
    return DataType::kTensorFloat32;
  } else if (str == "float64" || str == "fp64" || str == ".f64") {
    return DataType::kFloat64;
  } else if (str == "int1" || str == ".b1") {
    return DataType::kBit1;
  } else if (str == ".b8") {
    return DataType::kBit8;
  } else if (str == ".b16") {
    return DataType::kBit16;
  } else if (str == ".b32") {
    return DataType::kBit32;
  } else if (str == ".b64") {
    return DataType::kBit64;
  } else {
    TVM_FFI_THROW(InternalError) << "Unrecognized PTX data type " << str;
  }
}

/*!
 * \brief Get the string representation of given PTX data type.
 */
inline std::string DTypeToString(DataType dtype) { return dtype_str[static_cast<int>(dtype)]; }

/*!
 * \brief Get the number of bits of given PTX data type.
 */
inline uint32_t DTypeBits(DataType dtype) { return num_bits[static_cast<int>(dtype)]; }

/*!
 * \brief Extract the value m, n, k from string m*n*k*
 */
inline std::tuple<int, int, int> ParseMMAShape(const std::string& str) {
  size_t pos_m = str.find("m"), pos_n = str.find("n"), pos_k = str.find("k");
  TVM_FFI_CHECK(pos_m != str.npos && pos_n != str.npos && pos_k != str.npos)
      << "Cannot parse MMA shape " << str;
  int m = std::stoi(str.substr(pos_m + 1, pos_n - pos_m - 1)),
      n = std::stoi(str.substr(pos_n + 1, pos_k - pos_n - 1)), k = std::stoi(str.substr(pos_k + 1));
  return std::make_tuple(m, n, k);
}

/*!
 * \brief Layout Type
 */
enum class LayoutType : int { kRowMajor = 0, kColumnMajor = 1 };

/*!
 * \brief Parse layout type
 */
LayoutType LayoutTypeFromString(const std::string& str) {
  if (str == "row") {
    return LayoutType::kRowMajor;
  } else if (str == "col") {
    return LayoutType::kColumnMajor;
  } else {
    TVM_FFI_THROW(InternalError) << "Unrecognized layout type " << str;
  }
}

static const char* layout_type_str[] = {"row", "col"};

/*!
 * \brief Convert layout type to string.
 */
inline std::string LayoutTypeToString(LayoutType layout) {
  return layout_type_str[static_cast<int>(layout)];
}

/*!
 * \brief MMA Configurations, used to determine validity.
 */
struct MMAConfig {
  explicit MMAConfig(int m, int n, int k, DataType dtype_mul, bool use_bit_op, bool sparse)
      : m(m), n(n), k(k), dtype_mul(dtype_mul), use_bit_op(use_bit_op), sparse(sparse) {}
  int m, n, k;
  DataType dtype_mul;
  bool use_bit_op;
  bool sparse;
  inline bool operator==(const MMAConfig& other) {
    return m == other.m && n == other.n && k == other.k && dtype_mul == other.dtype_mul &&
           use_bit_op == other.use_bit_op && sparse == other.sparse;
  }
};

/*!
 * \brief Valid MMA configurations
 * \note Reference:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
 */
const MMAConfig valid_mma_configs[] = {
    MMAConfig(8, 8, 4, DataType::kFloat64, false, false),
    MMAConfig(8, 8, 4, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 8, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 16, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 8, DataType::kBFloat16, false, false),
    MMAConfig(16, 8, 16, DataType::kBFloat16, false, false),
    MMAConfig(16, 8, 4, DataType::kTensorFloat32, false, false),
    MMAConfig(16, 8, 8, DataType::kTensorFloat32, false, false),
    MMAConfig(8, 8, 16, DataType::kInt8, false, false),
    MMAConfig(16, 8, 16, DataType::kInt8, false, false),
    MMAConfig(16, 8, 32, DataType::kInt8, false, false),
    MMAConfig(8, 8, 16, DataType::kUInt8, false, false),
    MMAConfig(16, 8, 16, DataType::kUInt8, false, false),
    MMAConfig(16, 8, 32, DataType::kUInt8, false, false),
    MMAConfig(8, 8, 32, DataType::kInt4, false, false),
    MMAConfig(16, 8, 32, DataType::kInt4, false, false),
    MMAConfig(16, 8, 64, DataType::kInt4, false, false),
    MMAConfig(8, 8, 32, DataType::kUInt4, false, false),
    MMAConfig(16, 8, 32, DataType::kUInt4, false, false),
    MMAConfig(16, 8, 64, DataType::kUInt4, false, false),
    MMAConfig(8, 8, 128, DataType::kBit1, true, false),
    MMAConfig(16, 8, 128, DataType::kBit1, true, false),
    MMAConfig(16, 8, 256, DataType::kBit1, true, false),
    MMAConfig(16, 8, 16, DataType::kFloat16, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat16, false, true),
    MMAConfig(16, 8, 16, DataType::kBFloat16, false, true),
    MMAConfig(16, 8, 32, DataType::kBFloat16, false, true),
    MMAConfig(16, 8, 8, DataType::kTensorFloat32, false, true),
    MMAConfig(16, 8, 16, DataType::kTensorFloat32, false, true),
    MMAConfig(16, 8, 32, DataType::kInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kInt8, false, true),
    MMAConfig(16, 8, 32, DataType::kUInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kUInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kInt4, false, true),
    MMAConfig(16, 8, 128, DataType::kInt4, false, true),
    MMAConfig(16, 8, 64, DataType::kUInt4, false, true),
    MMAConfig(16, 8, 128, DataType::kUInt4, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat8_e4m3fn, false, false),
    MMAConfig(16, 8, 64, DataType::kFloat8_e4m3fn, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat8_e5m2, false, false),
    MMAConfig(16, 8, 64, DataType::kFloat8_e5m2, false, true),
};

/*!
 * \brief Check whether the multiplicand data type and accumulator data type is valid for MMA
 * computation.
 * \param dtype_a The data type of multiplicand a.
 * \param dtype_b The data type of multiplicand b.
 * \param dtype_c The data type of accumulator c.
 * \note Reference:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-data-types
 */
void CheckMMADTypeCompatible(DataType dtype_a, DataType dtype_b, DataType dtype_c) {
  std::string ab_not_match_err_str = "The multiplicands' data type " + DTypeToString(dtype_a) +
                                     DTypeToString(dtype_b) + " do not match.";
  // check a and b
  switch (dtype_a) {
    case DataType::kBit1:
    case DataType::kFloat16:
    case DataType::kBFloat16:
    case DataType::kTensorFloat32:
    case DataType::kFloat64:
      TVM_FFI_CHECK(dtype_a == dtype_b) << ab_not_match_err_str;
      break;
    case DataType::kInt4:
    case DataType::kUInt4:
      TVM_FFI_CHECK(dtype_b == DataType::kInt4 || dtype_b == DataType::kUInt4) << ab_not_match_err_str;
      break;
    case DataType::kInt8:
    case DataType::kUInt8:
      TVM_FFI_CHECK(dtype_b == DataType::kInt8 || dtype_b == DataType::kUInt8) << ab_not_match_err_str;
      break;
    case DataType::kFloat8_e4m3fn:
    case DataType::kFloat8_e5m2:
      TVM_FFI_CHECK(dtype_b == DataType::kFloat8_e4m3fn || dtype_b == DataType::kFloat8_e5m2)
          << ab_not_match_err_str;
      break;
    default:
      TVM_FFI_CHECK(false) << "Invalid multiplicand data types: " << DTypeToString(dtype_a)
                   << DTypeToString(dtype_b);
  }
  // check a,b and c
  switch (dtype_a) {
    case DataType::kBit1:
    case DataType::kInt4:
    case DataType::kUInt4:
    case DataType::kInt8:
    case DataType::kUInt8:
      TVM_FFI_CHECK(dtype_c == DataType::kInt32)
          << "For multiplicand data type " << DTypeToString(dtype_a) << DTypeToString(dtype_b)
          << ", accumulator data type should be s32.";
      break;
    case DataType::kFloat16:
      TVM_FFI_CHECK(dtype_c == DataType::kFloat16 || dtype_c == DataType::kFloat32)
          << "For multiplicand data type f16, accumulator data type should be f16/f32.";
      break;
    case DataType::kBFloat16:
    case DataType::kTensorFloat32:
      TVM_FFI_CHECK(dtype_c == DataType::kFloat32)
          << "For multiplicand data type bf16/tf32, accumulator data type can only be f32.";
      break;
    case DataType::kFloat64:
      TVM_FFI_CHECK(dtype_c == DataType::kFloat64)
          << "For multiplicand data type f64, accumulator data type can only be f64.";
      break;
    case DataType::kFloat8_e4m3fn:
    case DataType::kFloat8_e5m2:
      TVM_FFI_CHECK(dtype_c == DataType::kFloat32)
          << "For multiplicand data type e4m3/e5m2, accumulator data type can only be f32.";
      break;
    default:
      TVM_FFI_CHECK(false) << "Invalid multiplicand/accumulator data types: " << DTypeToString(dtype_a)
                   << DTypeToString(dtype_b) << DTypeToString(dtype_c) << ".";
  }
}

/*!
 * \brief Check whether the given configuration is valid for MMA computation.
 * \param m The M in mMnNkK of MMA instructions.
 * \param n The N in mMnNkK of MMA instructions.
 * \param k The K in mMnNkK of MMA instructions.
 * \param layout_a The layout of multiplicand A (row/col).
 * \param layout_b The layout of multiplicand B (row/col).
 * \param dtype_a The data type of multiplicand A.
 * \param dtype_b The data type of multiplicand B.
 * \param dtype_c The data type of accumulator C.
 * \param bit_op The bit operator for 1-bit MMA computation, can be "xor"/"and" or ""(if it's not
 * 1-bit MMA).
 * \param sparse Whether it's Sparse MMA or not.
 * \param saturate Whether saturate output or not.
 */
void CheckMMAConfigValidity(int m, int n, int k, LayoutType layout_a, LayoutType layout_b,
                            DataType dtype_a, DataType dtype_b, DataType dtype_c,
                            const std::string& bit_op, bool sparse, bool saturate) {
  TVM_FFI_CHECK(bit_op == "xor" || bit_op == "and" || bit_op == "")
      << "Unrecognized 1-bit operation " << bit_op << " , can only be xor/and.";
  bool use_bit_op = !bit_op.empty();
  if (use_bit_op) {
    TVM_FFI_CHECK(dtype_a == DataType::kBit1) << "Bit operator is only compatible with 1-bit multiplicand.";
  }
  CheckMMADTypeCompatible(dtype_a, dtype_b, dtype_c);
  if (saturate) {
    TVM_FFI_CHECK(dtype_a == DataType::kInt4 || dtype_a == DataType::kUInt4 || dtype_a == DataType::kInt8 ||
          dtype_a == DataType::kUInt8)
        << "Output saturation only applicable to multiplicand type s4/u4/s8/u8.";
  }

  if (!(m == 8 && n == 8 && k == 4 && dtype_a == ptx::DataType::kFloat16)) {
    // Only MMA on m8n8k4 for fp16 supports customized layouts.
    TVM_FFI_CHECK(layout_a == LayoutType::kRowMajor && layout_b == LayoutType::kColumnMajor)
        << "Invalid layout combination " << LayoutTypeToString(layout_a) << ","
        << LayoutTypeToString(layout_b) << ".";
  }

  MMAConfig config(m, n, k, dtype_a, use_bit_op, sparse);
  bool match = false;
  for (const MMAConfig& valid_config : valid_mma_configs) {
    if (config == valid_config) {
      match = true;
      break;
    }
  }
  TVM_FFI_CHECK(match) << "Cannot find matched MMA configurations.";
}

/*!
 * \brief Fragment attributes
 */
class FragAttrs {
 public:
  explicit FragAttrs(char reg_type, uint32_t size, std::string ptr_type)
      : reg_type(reg_type), size(size), ptr_type(ptr_type) {}
  /*! \brief PTX register type */
  char reg_type;
  /*! \brief Fragment size */
  uint32_t size;
  /*! \brief Fragment pointer type */
  std::string ptr_type;
};

/*!
 * \brief Fragment attributes of given data type.
 */
inline FragAttrs GetFragAttrs(DataType dtype) {
  switch (dtype) {
    case DataType::kBit1:
    case DataType::kInt4:
    case DataType::kUInt4:
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kFloat8_e4m3fn:
    case DataType::kFloat8_e5m2:
    case DataType::kBit16:
    case DataType::kFloat16:  // .f16x2 register
    case DataType::kBFloat16:
    case DataType::kTensorFloat32:
      return FragAttrs('r', 32, "(unsigned *)");
    case DataType::kInt32:
      return FragAttrs('r', 32, "(int *)");
    case DataType::kFloat32:
      return FragAttrs('f', 32, "(float *)");
    case DataType::kFloat64:
      return FragAttrs('d', 64, "(double *)");
    default:
      TVM_FFI_ICHECK(false) << DTypeToString(dtype) << " is not matrix data type in MMA.";
      return FragAttrs('\0', 0, "");
  }
}

};  // namespace ptx

/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
 public:
  void register_rule(const std::string& pattern, const std::string& replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto&& rule : _rules) {
      auto [pattern, replacement] = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

 private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

/*!
 * \brief Get the number of MMA computations for given shape and datatype.
 */
inline uint32_t GetNumMMAComputations(int m, int n, int k, ptx::DataType dtype) {
  if (m == 8 && n == 8 && k == 4 && dtype == ptx::DataType::kFloat16) {
    // MMA for m8n8k4 on fp16 would launch 4 MMA computations instead of one.
    return 4;
  } else {
    return 1;
  }
}

/*!
 * \brief Return template string, input operands string and output operands string.
 * \param m The M in mMnNkK of MMA instructions.
 * \param n The N in mMnNkK of MMA instructions.
 * \param k The K in mMnNkK of MMA instructions.
 * \param dtype_a The data type of multiplicand a.
 * \param dtype_b The data type of multiplicand b.
 * \param dtype_c The data type of accumulator c.
 * \param sparse Whether it's Sparse MMA or not.
 */
inline std::tuple<std::string, std::string, std::string> GetMMAOperands(int m, int n, int k,
                                                                        ptx::DataType dtype_a,
                                                                        ptx::DataType dtype_b,
                                                                        ptx::DataType dtype_c,
                                                                        bool sparse) {
  std::stringstream templates, inputs, outputs;
  const ptx::FragAttrs frag_attr_a = ptx::GetFragAttrs(dtype_a),
                       frag_attr_b = ptx::GetFragAttrs(dtype_b),
                       frag_attr_c = ptx::GetFragAttrs(dtype_c);
  constexpr uint32_t warp_size = 32;
  const uint32_t threads = warp_size / GetNumMMAComputations(m, n, k, dtype_a);
  const int num_operands_a =
                (m * k) * ptx::DTypeBits(dtype_a) / frag_attr_a.size / threads / (sparse ? 2 : 1),
            num_operands_b = (k * n) * ptx::DTypeBits(dtype_b) / frag_attr_b.size / threads,
            num_operands_c = (m * n) * ptx::DTypeBits(dtype_c) / frag_attr_c.size / threads;

  // generate templates;
  int arg_counter = 0;
  templates << "{"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_a; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_b; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}";
  // templates of metadata and sparse selector for sparse mma.
  if (sparse) {
    templates << ", %" << (arg_counter++) << ", F";
  }

  // generate inputs
  for (int i = 0; i < num_operands_a; ++i) {
    if (i != 0) {
      inputs << ", ";
    }
    inputs << "\"" << frag_attr_a.reg_type << "\"((" << frag_attr_a.ptr_type << "(A))[" << i
           << "])";
  }
  for (int i = 0; i < num_operands_b; ++i) {
    inputs << ", \"" << frag_attr_b.reg_type << "\"((" << frag_attr_b.ptr_type << "(B))[" << i
           << "])";
  }
  for (int i = 0; i < num_operands_c; ++i) {
    inputs << ", \"" << frag_attr_c.reg_type << "\"((" << frag_attr_c.ptr_type << "(C))[" << i
           << "])";
  }
  // input of metadata for sparse mma.
  if (sparse) {
    inputs << ", \"r\"(((unsigned *)(E))[0])";
  }

  // generate outputs
  for (int i = 0; i < num_operands_c; ++i) {
    if (i != 0) {
      outputs << ",";
    }
    outputs << " \"=" << frag_attr_c.reg_type << "\"((" << frag_attr_c.ptr_type << "(D))[" << i
            << "])";
  }
  return std::make_tuple(templates.str(), inputs.str(), outputs.str());
}

std::string PrintMMAAssembly(const std::string& shape, const std::string& A_layout,
                             const std::string& B_layout, const std::string& A_dtype,
                             const std::string& B_dtype, const std::string& C_dtype,
                             const std::string& a_ptr, const std::string& a_elem_offset,
                             const std::string& b_ptr, const std::string& b_elem_offset,
                             const std::string& c_ptr, const std::string& c_elem_offset,
                             const std::string& metadata, const std::string& metadata_offset,
                             const std::string& sparsity_selector, const std::string& bit_op,
                             bool sparse, bool saturate) {
  ptx::DataType dtype_a = ptx::DTypeFromString(A_dtype), dtype_b = ptx::DTypeFromString(B_dtype),
                dtype_c = ptx::DTypeFromString(C_dtype);
  ptx::LayoutType layout_a = ptx::LayoutTypeFromString(A_layout),
                  layout_b = ptx::LayoutTypeFromString(B_layout);
  auto [m, n, k] = ptx::ParseMMAShape(shape);
  CheckMMAConfigValidity(m, n, k, layout_a, layout_b, dtype_a, dtype_b, dtype_c, bit_op, sparse,
                         saturate);
  std::string asm_code = R"(
  {
    __asm__ __volatile__(
      "mma{.sparse}.sync.aligned{.shape}{.alayout}{.blayout}{.saturate}{.dtype}{.atype}{.btype}{.ctype}{.bitop}"
      "{templates};\n"
      : {outputs}
      : {inputs});
  }
)";
  auto [templates_str, inputs_str, outputs_str] =
      GetMMAOperands(m, n, k, dtype_a, dtype_b, dtype_c, sparse);

  // replace patterns
  Replacer replacer;
  replacer.register_rule("{.sparse}", sparse ? ".sp" : "");
  replacer.register_rule("{.shape}", "." + shape);
  replacer.register_rule("{.saturate}", saturate ? ".satfinite" : "");
  replacer.register_rule("{.alayout}", "." + A_layout);
  replacer.register_rule("{.blayout}", "." + B_layout);
  replacer.register_rule("{.atype}", ptx::DTypeToString(dtype_a));
  replacer.register_rule("{.btype}", ptx::DTypeToString(dtype_b));
  replacer.register_rule("{.ctype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{.dtype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{.bitop}", bit_op.empty() ? "" : "." + bit_op + ".popc");
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  replacer.register_rule("{inputs}", inputs_str);
  asm_code = replacer.rewrite(asm_code);
  replacer.empty_rules();
  replacer.register_rule("A", a_ptr + " + " + a_elem_offset);
  replacer.register_rule("B", b_ptr + " + " + b_elem_offset);
  replacer.register_rule("C", c_ptr + " + " + c_elem_offset);
  replacer.register_rule("D", c_ptr + " + " + c_elem_offset);
  replacer.register_rule("E", metadata + " + " + metadata_offset);
  replacer.register_rule("F", sparsity_selector);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

inline std::tuple<std::string, std::string> GetLoadMatrixOperands(
    int num, const std::string& local_ptr, const std::string& local_elem_offset) {
  std::stringstream templates, outputs;
  int arg_counter = 0;
  // generate templates
  templates << "{%" << arg_counter++;
  for (int i = 1; i < num; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, [%" << arg_counter++ << "]";
  // generate outputs
  std::string ptr_type = "(unsigned *)";
  for (int i = 0; i < num; ++i) {
    if (i != 0) {
      outputs << ", ";
    }
    outputs << "\"=r\"((" << ptr_type << "(" << local_ptr << " + " << local_elem_offset << "))["
            << i << "])";
  }
  return std::make_tuple(templates.str(), outputs.str());
}

std::string PrintLoadMatrixAssembly(bool trans, int num, const std::string& type,
                                    const std::string& local_ptr,
                                    const std::string& local_elem_offset,
                                    const std::string& smem_ptr,
                                    const std::string& smem_elem_offset) {
  TVM_FFI_CHECK(num == 1 || num == 2 || num == 4) << "ldmatrix only accept loading 1/2/4 matrices.";
  ptx::DataType data_type = ptx::DTypeFromString(type);
  TVM_FFI_CHECK(data_type == ptx::DataType::kBit16) << "ldmatrix only accept matrix with type .b16.";
  std::string asm_code = R"(
  {
    unsigned int addr = __cvta_generic_to_shared({smem_addr});
    __asm__ __volatile__(
      "ldmatrix.sync.aligned{.shape}{.num}{.trans}{.ss}{.type}"
      "{templates};\n"
      : {outputs}
      : "r"(addr)
    );
  }
)";
  auto [templates_str, outputs_str] = GetLoadMatrixOperands(num, local_ptr, local_elem_offset);

  Replacer replacer;
  replacer.register_rule("{.shape}", ".m8n8");
  replacer.register_rule("{.num}", ".x" + std::to_string(num));
  replacer.register_rule("{.trans}", trans ? ".trans" : "");
  replacer.register_rule("{.ss}", ".shared");
  replacer.register_rule("{.type}", ptx::DTypeToString(data_type));
  replacer.register_rule("{smem_addr}", smem_ptr + " + " + smem_elem_offset);
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintCpAsyncAssembly(const std::string& shared_ptr,
                                 const std::string& shared_elem_offset,
                                 const std::string& global_ptr,
                                 const std::string& global_elem_offset, const std::string& bytes) {
  std::string asm_code = R"(// T.ptx_cp_async()
{
  unsigned int addr = __cvta_generic_to_shared({smem_addr});
  __asm__ __volatile__(
    #if TVM_ENABLE_L2_PREFETCH
      "cp.async.{cg_or_ca}.shared.global.L2::128B [%0], [%1], %2;"
    #else
      "cp.async.{cg_or_ca}.shared.global [%0], [%1], %2;"
    #endif
      :: "r"(addr), "l"((void*)({global_ptr})), "n"({bytes})
  );
}
)";
  Replacer replacer;
  replacer.register_rule("{smem_addr}", shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}", global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{cg_or_ca}", bytes == "16" ? "cg" : "ca");
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintPredicatedCpAsyncAssembly(const std::string& shared_ptr,
                                           const std::string& shared_elem_offset,
                                           const std::string& global_ptr,
                                           const std::string& global_elem_offset,
                                           const std::string& bytes,
                                           const std::string& predicate_value) {
  TVM_FFI_CHECK(bytes == "16" || bytes == "12" || bytes == "8" || bytes == "4" || bytes == "2" ||
        bytes == "1")
      << "Only support 16, 12, 8, 4, 2, 1 bytes for predicated cp.async";
  std::string predicated_asm_code = R"(
  {
    unsigned int addr = __cvta_generic_to_shared({smem_addr});
    int pred_guard = (int){pred_guard};
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.{cg_or_ca}.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.{cg_or_ca}.shared.global [%1], [%2], %3;"
      #endif
      "  @!p {store_shared};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)({global_ptr})), "n"({bytes}), {nopreg}
    );
  }
)";
  auto [store_shared, nopreg] = [](const std::string& bytes) {
    if (bytes == "16")
      return std::make_tuple("st.shared.v4.u32 [%1], {%4, %5, %6, %7}",
                             "\"r\"(0), \"r\"(0), \"r\"(0),\"r\"(0)");
    else if (bytes == "12")
      return std::make_tuple("st.shared.v3.u32 [%1], {%4, %5, %6}", "\"r\"(0), \"r\"(0), \"r\"(0)");
    else if (bytes == "8")
      return std::make_tuple("st.shared.v2.u32 [%1], {%4, %5}", "\"r\"(0), \"r\"(0)");
    else if (bytes == "4")
      return std::make_tuple("st.shared.u32 [%1], {%4}", "\"r\"(0)");
    else if (bytes == "2")
      return std::make_tuple("st.shared.u16 [%1], {%4}", "\"r\"(0)");
    else if (bytes == "1")
      return std::make_tuple("st.shared.u8 [%1], {%4}", "\"r\"(0)");
    else
      return std::make_tuple("", "");
  }(bytes);

  Replacer replacer;
  replacer.register_rule("{smem_addr}", shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}", global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{cg_or_ca}", bytes == "16" ? "cg" : "ca");
  replacer.register_rule("{store_shared}", store_shared);
  replacer.register_rule("{nopreg}", nopreg);
  replacer.register_rule("{pred_guard}", predicate_value);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintCpAsyncBulkAsm(const std::string& shared_ptr,
                                const std::string& shared_elem_offset,
                                const std::string& global_ptr,
                                const std::string& global_elem_offset, const std::string& bytes,
                                const std::string& barrier) {
  std::string asm_code = R"(
  {
    unsigned int smem_addr_int = __cvta_generic_to_shared({smem_addr});
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    __asm__ __volatile__(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_addr_int), "l"({global_ptr}), "r"({bytes}), "r"(barrier_addr_int)
      : "memory"
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{smem_addr}", shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}", global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{barrier}", "&" + barrier);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintCpAsyncBarrierAsm(const std::string& barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    __asm__ __volatile__(
      "cp.async.mbarrier.arrive.shared.b64 [%0];"
      :: "r" (barrier_addr_int)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintInitBarrierThreadCountAsm(const std::string& barrier,
                                           const std::string& thread_count) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    int thread_count = {thread_count};
    __asm__ __volatile__(
      "mbarrier.init.shared.b64 [%0], %1;"
      :: "r"(barrier_addr_int), "r"(thread_count)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  replacer.register_rule("{thread_count}", thread_count);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintArriveBarrierAsm(const std::string& barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    __asm__ __volatile__(
      "{ .reg .b64 state; mbarrier.arrive.shared.b64 state, [%0]; }"
      :: "r"(barrier_addr_int)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintArriveBarrierExpectTxAsm(const std::string& barrier,
                                          const std::string& byte_count) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    int byte_count = {byte_count};
    __asm__ __volatile__(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
      :: "r"(barrier_addr_int), "r"(byte_count)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  replacer.register_rule("{byte_count}", byte_count);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintWaitBarrierAsm(const std::string& barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = __cvta_generic_to_shared({barrier});
    constexpr int phase_bit = 0;
    __asm__ __volatile__(
      "{ .reg .pred P; WAIT: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @P bra.uni DONE; bra.uni WAIT; DONE: }"
      :: "r"(barrier_addr_int), "r"(phase_bit)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintCudaFenceProxyAsyncAssembly(CodeGenCUDA* cg, std::string scope) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  __asm__ __volatile__("fence.proxy.async.{scope};");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_cuda_fence_proxy_async_{scope}";
  {  // func name
    Replacer replacer;
    replacer.register_rule("{scope}", scope);
    func_name = replacer.rewrite(func_name);
  }
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    if (scope == "shared") {
      scope = "shared::cta";
    }
    replacer.register_rule("{scope}", scope);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintMbarrierInitAssembly(CodeGenCUDA* cg, const std::string& barrier,
                                      const std::string& thread_count) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier, int thread_count) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.init.shared.b64 [%0], %1;"
    :: "r"(barrier_addr_int), "r"(thread_count)
  );
}
)";
  std::string caller_code = "{func_name}({barrier}, {thread_count});\n";

  std::string func_name = "ptx_mbarrier_init";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{barrier}", barrier);
    replacer.register_rule("{thread_count}", thread_count);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintMbarrierArriveAssembly(codegen::CodeGenCUDA* cg, const std::string& barrier,
                                        bool remote, const std::string& cta_id,
                                        const std::string& pred) {
  if (!remote) {
    std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.shared.b64 _, [%0];"
    :: "r"(barrier_addr_int)
  );
}
)";
    std::string caller_code = "{func_name}({barrier});\n";

    std::string func_name = "ptx_mbarrier_arrive";
    {  // func code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      func_code = replacer.rewrite(func_code);
    }
    {  // caller code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      replacer.register_rule("{barrier}", barrier);
      caller_code = replacer.rewrite(caller_code);
    }
    cg->AddUtilFunction(func_name, func_code);
    return caller_code;
  } else {
    std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier, int cta_id, int pred) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      ".reg .b32 remAddr32;\n\t"
      "setp.eq.u32 p, %2, 1;\n\t"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
      "}"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred));
  }
)";
    std::string caller_code = "{func_name}({barrier}, {cta_id}, {pred});\n";

    std::string func_name = "ptx_mbarrier_arrive_remote";
    {  // func code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      func_code = replacer.rewrite(func_code);
    }
    {  // caller code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      replacer.register_rule("{barrier}", barrier);
      replacer.register_rule("{cta_id}", cta_id);
      replacer.register_rule("{pred}", pred);
      caller_code = replacer.rewrite(caller_code);
    }
    cg->AddUtilFunction(func_name, func_code);
    return caller_code;
  }
}

std::string PrintMbarrierArriveExpectTxAssembly(CodeGenCUDA* cg, const std::string& barrier,
                                                const std::string& byte_count, bool remote,
                                                const std::string& cta_id,
                                                const std::string& pred) {
  if (!remote) {
    std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
    :: "r"(barrier_addr_int), "r"(byte_count)
  );
}
)";
    std::string caller_code = "{func_name}({barrier}, {byte_count});\n";

    std::string func_name = "ptx_mbarrier_arrive_expect_tx";
    {  // func code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      func_code = replacer.rewrite(func_code);
    }
    {  // caller code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      replacer.register_rule("{barrier}", barrier);
      replacer.register_rule("{byte_count}", byte_count);
      caller_code = replacer.rewrite(caller_code);
    }
    cg->AddUtilFunction(func_name, func_code);
    return caller_code;
  } else {
    std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count, int cta_id, int pred) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      ".reg .b32 remAddr32;\n\t"
      "setp.eq.u32 p, %2, 1;\n\t"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
      "}"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred), "r"(byte_count));
  }
}
)";
    std::string caller_code = "{func_name}({barrier}, {byte_count}, {cta_id}, {pred});\n";

    std::string func_name = "ptx_mbarrier_arrive_expect_tx_remote";
    {  // func code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      func_code = replacer.rewrite(func_code);
    }
    {  // caller code
      Replacer replacer;
      replacer.register_rule("{func_name}", func_name);
      replacer.register_rule("{barrier}", barrier);
      replacer.register_rule("{byte_count}", byte_count);
      replacer.register_rule("{cta_id}", cta_id);
      replacer.register_rule("{pred}", pred);
      caller_code = replacer.rewrite(caller_code);
    }
    cg->AddUtilFunction(func_name, func_code);
    return caller_code;
  }
}

std::string PrintMbarrierWaitAssembly(codegen::CodeGenCUDA* cg, const std::string& barrier,
                                      const std::string& phase) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* barrier, int phase) {
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile (
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n"
      ::
      "r"(barrier_addr_int),
      "r"(phase)
  );
}
)";
  std::string caller_code = "{func_name}(reinterpret_cast<void*>({barrier}), {phase});\n";

  std::string func_name = "ptx_mbarrier_wait";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{barrier}", barrier);
    replacer.register_rule("{phase}", phase);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintNamedBarrierArriveAssembly(const std::string& name_bar_id,
                                            const std::string& thread_count) {
  std::string asm_code = R"(/* T.ptx_bar_arrive() */ {
  asm volatile("bar.arrive %0, %1;" : : "r"({name_bar_id}), "r"({thread_count}));
}
)";

  Replacer replacer;
  replacer.register_rule("{name_bar_id}", name_bar_id);
  replacer.register_rule("{thread_count}", thread_count);
  return replacer.rewrite(asm_code);
}

std::string PrintNamedBarrierSyncAssembly(const std::string& name_bar_id,
                                          const std::string& thread_count) {
  std::string asm_code = R"(/* T.ptx_bar_sync() */ {
  asm volatile("bar.sync %0, %1;" : : "r"({name_bar_id}), "r"({num_threads}));
}
)";

  Replacer replacer;
  replacer.register_rule("{name_bar_id}", name_bar_id);
  replacer.register_rule("{num_threads}", thread_count);
  return replacer.rewrite(asm_code);
}

std::string PrintCpAsyncBulkTensorGlobalToClusterAssembly(
    CodeGenCUDA* cg, int dim, const std::string& dst, const std::string& bar,
    const std::string& tensormap, int cta_mask, int cta_group, std::vector<std::string> coords) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap* tensormap, int cta_mask_, {coord_arg_list}) {
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(tensormap);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_);
  if (cta_mask != 0) {
    // multicast
    __asm__ __volatile__(
      "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster{cta_group}"
      " [%0], [%1, {arg_template_multicast}], [%2], %3;"
      :
      : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr), "h"(cta_mask),
        {coord_list}
      : "memory"
    );
  } else {
    // unicast
    __asm__ __volatile__(
      "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes{cta_group}"
      " [%0], [%1, {arg_template_unicast}], [%2];"
      :
      : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr),
        {coord_list}
      : "memory"
    );
  }
}
)";
  std::string caller_code =
      "{func_name}({dst}, {bar}, &({tensormap}), {cta_mask}, {coord_list});\n";

  std::string func_name = "ptx_cp_async_bulk_tensor_global_to_cluster_{dim}d";
  {
    // func name
    Replacer replacer;
    replacer.register_rule("{dim}", std::to_string(dim));
    func_name = replacer.rewrite(func_name);
  }
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    // get coord arg list
    std::string coord_arg_list = "int coord0";
    for (int i = 1; i < dim; ++i) {
      coord_arg_list += ", int coord" + std::to_string(i);
    }
    replacer.register_rule("{coord_arg_list}", coord_arg_list);
    // get arg template for multicast
    std::string arg_template_multicast = "{%" + std::to_string(4);
    for (int i = 1; i < dim; ++i) {
      arg_template_multicast += ", %" + std::to_string(4 + i);
    }
    arg_template_multicast += "}";
    replacer.register_rule("{arg_template_multicast}", arg_template_multicast);
    // get arg template for unicast
    std::string arg_template_unicast = "{%" + std::to_string(3);
    for (int i = 1; i < dim; ++i) {
      arg_template_unicast += ", %" + std::to_string(3 + i);
    }
    arg_template_unicast += "}";
    replacer.register_rule("{arg_template_unicast}", arg_template_unicast);
    // get coord list
    std::string coord_list = "\"r\"(coord0)";
    for (int i = 1; i < dim; ++i) {
      coord_list += ", \"r\"(coord" + std::to_string(i) + ")";
    }
    replacer.register_rule("{coord_list}", coord_list);
    replacer.register_rule("{dim}", std::to_string(dim));
    if (cta_group == -1) {
      replacer.register_rule("{cta_group}", "");
    } else {
      replacer.register_rule("{cta_group}", ".cta_group::" + std::to_string(cta_group));
    }
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{dst}", dst);
    replacer.register_rule("{bar}", bar);
    replacer.register_rule("{tensormap}", tensormap);
    replacer.register_rule("{cta_mask}", std::to_string(cta_mask));
    // get coord list
    std::string coord_list = coords[0];
    for (int i = 1; i < dim; ++i) {
      coord_list += ", " + coords[i];
    }
    replacer.register_rule("{coord_list}", coord_list);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintCpAsyncBulkTensorSharedToGlobalAssembly(codegen::CodeGenCUDA* cg, int dim,
                                                         const std::string& src,
                                                         const std::string& tensormap,
                                                         std::vector<std::string> coords) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* src, const CUtensorMap* tensormap, {coord_arg_list}) {
  unsigned int src_addr = __cvta_generic_to_shared(src);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"
    "[%0, {arg_template}], [%1];"
    :
    : "l"(tensormap_addr), "r"(src_addr),
      {coord_list}
    : "memory"
  );
}
)";
  std::string caller_code = "{func_name}({src}, &({tensormap}), {coord_list});\n";

  std::string func_name = "ptx_cp_async_bulk_tensor_shared_to_global_{dim}d";
  {
    // func name
    Replacer replacer;
    replacer.register_rule("{dim}", std::to_string(dim));
    func_name = replacer.rewrite(func_name);
  }
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    // get coord arg list
    std::string coord_arg_list = "int coord0";
    for (int i = 1; i < dim; ++i) {
      coord_arg_list += ", int coord" + std::to_string(i);
    }
    replacer.register_rule("{coord_arg_list}", coord_arg_list);
    // get arg template
    std::string arg_template = "{%" + std::to_string(2);
    for (int i = 1; i < dim; ++i) {
      arg_template += ", %" + std::to_string(2 + i);
    }
    arg_template += "}";
    replacer.register_rule("{arg_template}", arg_template);
    // get coord list
    std::string coord_list = "\"r\"(coord0)";
    for (int i = 1; i < dim; ++i) {
      coord_list += ", \"r\"(coord" + std::to_string(i) + ")";
    }
    replacer.register_rule("{coord_list}", coord_list);
    replacer.register_rule("{dim}", std::to_string(dim));
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{src}", src);
    replacer.register_rule("{tensormap}", tensormap);
    // get coord list
    std::string coord_list = coords[0];
    for (int i = 1; i < dim; ++i) {
      coord_list += ", " + coords[i];
    }
    replacer.register_rule("{coord_list}", coord_list);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintCpAsyncBulkTensorCommitGroupAssembly(codegen::CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("cp.async.bulk.commit_group;");
}
)";
  std::string func_name = "ptx_cp_async_bulk_tensor_commit_group";
  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  func_code = replacer.rewrite(func_code);
  cg->AddUtilFunction(func_name, func_code);
  return func_name + "();\n";
}

std::string PrintCpAsyncBulkTensorWaitGroupAssembly(codegen::CodeGenCUDA* cg, const std::string& N,
                                                    bool read) {
  std::string func_code = R"(
template <int N, bool read>
__forceinline__ __device__ void {func_name}() {
  if (read) {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N): "memory");
  } else {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N): "memory");
  }
}
)";
  std::string func_name = "ptx_cp_async_bulk_tensor_wait_group";
  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  func_code = replacer.rewrite(func_code);
  cg->AddUtilFunction(func_name, func_code);
  return func_name + "<" + N + ", " + std::to_string(read) + ">();\n";
}

std::string PrintPtxFetchRegisterAssembly(codegen::CodeGenCUDA* cg, int bits,
                                          const std::string& reg) {
  std::string func_code = R"(
__forceinline__ __device__ int{bits}_t {func_name}() {
  uint{bits}_t x;
  asm volatile("mov.u{bits} %0, %{reg};\n" : "=r"(x) : );
  return (int{bits}_t)x;
}
)";
  TVM_FFI_CHECK(bits == 32 || bits == 64) << "Only support 32/64 bits for ptx_fetch_register.";

  std::string reg_rp = reg;
  std::replace(reg_rp.begin(), reg_rp.end(), '.', '_');
  std::string func_name = "ptx_" + reg_rp;

  Replacer replacer;
  replacer.register_rule("{bits}", std::to_string(bits));
  replacer.register_rule("{func_name}", func_name);
  replacer.register_rule("{reg}", reg);
  func_code = replacer.rewrite(func_code);

  cg->AddUtilFunction(func_name, func_code);
  return func_name + "()";
}

std::string PrintWGMMAFenceOpearandAssembly(CodeGenCUDA* cg, const std::string& reg,
                                            tvm::DataType dtype) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}({dtype} reg) {
  asm volatile("" : "+{format}"(reg)::"memory");
}
)";
  std::string func_name = "ptx_wgmma_fence_{dtype}";
  std::string format, dtype_str;
  if (dtype == DataType::UInt(32)) {
    format = "r";
    dtype_str = "uint32_t";
  } else if (dtype == DataType::Float(32)) {
    format = "f";
    dtype_str = "float";
  } else {
    TVM_FFI_THROW(InternalError) << "Only support uint32/float32 for wgmma_fence.";
  }
  {
    // func name
    Replacer replacer;
    replacer.register_rule("{dtype}", dtype_str);
    func_name = replacer.rewrite(func_name);
  }
  {
    // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{format}", format);
    replacer.register_rule("{dtype}", dtype_str);
    func_code = replacer.rewrite(func_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return func_name + "(" + reg + ");\n";
}

std::string PrintWGMMAasyncSSAssembly(int M, int N, int K, const std::string& in_dtype,
                                      const std::string& out_dtype, bool transA, bool transB,
                                      float scaleA, float scaleB, const std::string& scaleD,
                                      const std::string& descA, const std::string& descB,
                                      const std::vector<std::string>& accums) {
  std::string asm_code = R"(/* T.ptx_wgmma_mma_async_ss() */ {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, {scaleD_r}, 0;\n"
    "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype}{itype}{itype} "
    "{{accum_r_list}},"
    "{descA_r}, {descB_r},"
    "p, {scaleA_r}, {scaleB_r}{transpose_r_code};\n"
    "}\n"
    : {accum_list}
    : "l"({descA}), "l"({descB}), "r"(int32_t({scaleD})), "n"(int32_t({scaleA})),
      "n"(int32_t({scaleB})){transpose_code}
  );
}
)";
  std::string transpose_r_code = ", {transA_r}, {transB_r}";
  std::string transpose_code = ", \"n\"(int32_t({transA})), \"n\"(int32_t({transB}))";
  Replacer replacer;
  auto itype = ptx::DTypeToString(ptx::DTypeFromString(in_dtype));
  auto otype = ptx::DTypeToString(ptx::DTypeFromString(out_dtype));
  std::string accum_r_list = "%" + std::to_string(0);
  for (size_t i = 1; i < accums.size(); ++i) {
    accum_r_list += ", %" + std::to_string(i);
  }
  bool allow_transpose = in_dtype == "float16" || in_dtype == "bfloat16";
  if (!allow_transpose) {
    TVM_FFI_CHECK(transA == false && transB == false)
        << "Matrices A and B are stored in row-major and column-major format respectively. The "
           "transpose operation is only supported for the wgmma.mma_async variants with .f16/ "
           ".bf16 types on matrices accessed from shared memory using matrix descriptors.";
  }
  TVM_FFI_CHECK(out_dtype == "float32")
      << "ValueError: codegen only support float32 as output dtype for WGMMMA.";
  std::string accum_list = "\"+f\"(" + accums[0] + ")";
  for (size_t i = 1; i < accums.size(); ++i) {
    accum_list += ", \"+f\"(" + accums[i] + ")";
  }
  std::string descA_r = "%" + std::to_string(accums.size());
  std::string descB_r = "%" + std::to_string(accums.size() + 1);
  std::string scaleD_r = "%" + std::to_string(accums.size() + 2);
  std::string scaleA_r = "%" + std::to_string(accums.size() + 3);
  std::string scaleB_r = "%" + std::to_string(accums.size() + 4);
  std::string transA_r = "%" + std::to_string(accums.size() + 5);
  std::string transB_r = "%" + std::to_string(accums.size() + 6);

  replacer.register_rule("{M}", std::to_string(M));
  replacer.register_rule("{N}", std::to_string(N));
  replacer.register_rule("{K}", std::to_string(K));
  replacer.register_rule("{itype}", itype);
  replacer.register_rule("{otype}", otype);
  replacer.register_rule("{descA_r}", descA_r);
  replacer.register_rule("{descB_r}", descB_r);
  replacer.register_rule("{scaleA_r}", scaleA_r);
  replacer.register_rule("{scaleB_r}", scaleB_r);
  replacer.register_rule("{scaleD_r}", scaleD_r);
  replacer.register_rule("{transA_r}", transA_r);
  replacer.register_rule("{transB_r}", transB_r);
  replacer.register_rule("{descA}", descA);
  replacer.register_rule("{descB}", descB);
  replacer.register_rule("{scaleA}", std::to_string(scaleA));
  replacer.register_rule("{scaleB}", std::to_string(scaleB));
  replacer.register_rule("{scaleD}", scaleD);
  replacer.register_rule("{transA}", transA ? "1" : "0");
  replacer.register_rule("{transB}", transB ? "1" : "0");
  replacer.register_rule("{accum_r_list}", accum_r_list);
  replacer.register_rule("{accum_list}", accum_list);

  transpose_r_code = allow_transpose ? replacer.rewrite(transpose_r_code) : "";
  transpose_code = allow_transpose ? replacer.rewrite(transpose_code) : "";
  replacer.register_rule("{transpose_r_code}", transpose_r_code);
  replacer.register_rule("{transpose_code}", transpose_code);

  return replacer.rewrite(asm_code);
}

std::string PrintWGMMAasyncRSAssembly(int M, int N, int K, const std::string& in_dtype,
                                      const std::string& out_dtype, bool transA, bool transB,
                                      float scaleA, float scaleB, const std::string& scaleD,
                                      const std::vector<std::string>& A_regs,
                                      const std::string& descB,
                                      const std::vector<std::string>& accums) {
  std::string asm_code = R"(/* T.ptx_wgmma_mma_async_rs() */ {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, {scaleD_r}, 0;\n"
    "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype}{itype}{itype} "
    "{{accum_r_list}},"
    "{{A_reg_r_list}}, {descB_r},"
    "p, {scaleA_r}, {scaleB_r}{transpose_r_code};\n"
    "}\n"
    : {accum_list}
    : {A_reg_list}, 
      "l"({descB}), "r"(int32_t({scaleD})), "n"(int32_t({scaleA})),
      "n"(int32_t({scaleB})){transpose_code}
  );
}
)";
  std::string transpose_r_code = ", {transB_r}";
  std::string transpose_code = ", \"n\"(int32_t({transB}))";
  Replacer replacer;
  auto itype = ptx::DTypeToString(ptx::DTypeFromString(in_dtype));
  auto otype = ptx::DTypeToString(ptx::DTypeFromString(out_dtype));
  std::string accum_r_list = "%" + std::to_string(0);
  for (size_t i = 1; i < accums.size(); ++i) {
    accum_r_list += ", %" + std::to_string(i);
  }
  std::string A_reg_r_list = "%" + std::to_string(accums.size());
  for (size_t i = 1; i < A_regs.size(); ++i) {
    A_reg_r_list += ", %" + std::to_string(accums.size() + i);
  }
  bool allow_transpose = in_dtype == "float16" || in_dtype == "bfloat16";
  if (!allow_transpose) {
    TVM_FFI_CHECK(transA == false && transB == false)
        << "Matrices A and B are stored in row-major and column-major format respectively. The "
           "transpose operation is only supported for the wgmma.mma_async variants with .f16/ "
           ".bf16 types on matrices accessed from shared memory using matrix descriptors.";
  }
  TVM_FFI_CHECK(out_dtype == "float32")
      << "ValueError: codegen only support float32 as output dtype for WGMMMA.";
  std::string accum_list = "\"+f\"(" + accums[0] + ")";
  for (size_t i = 1; i < accums.size(); ++i) {
    accum_list += ", \"+f\"(" + accums[i] + ")";
  }
  std::string A_reg_list = "\"r\"(" + A_regs[0] + ")";
  for (size_t i = 1; i < A_regs.size(); ++i) {
    A_reg_list += ", \"r\"(" + A_regs[i] + ")";
  }
  std::string descB_r = "%" + std::to_string(accums.size() + A_regs.size());
  std::string scaleD_r = "%" + std::to_string(accums.size() + A_regs.size() + 1);
  std::string scaleA_r = "%" + std::to_string(accums.size() + A_regs.size() + 2);
  std::string scaleB_r = "%" + std::to_string(accums.size() + A_regs.size() + 3);
  std::string transB_r = "%" + std::to_string(accums.size() + A_regs.size() + 4);

  replacer.register_rule("{M}", std::to_string(M));
  replacer.register_rule("{N}", std::to_string(N));
  replacer.register_rule("{K}", std::to_string(K));
  replacer.register_rule("{itype}", itype);
  replacer.register_rule("{otype}", otype);
  replacer.register_rule("{A_reg_r_list}", A_reg_r_list);
  replacer.register_rule("{descB_r}", descB_r);
  replacer.register_rule("{scaleA_r}", scaleA_r);
  replacer.register_rule("{scaleB_r}", scaleB_r);
  replacer.register_rule("{scaleD_r}", scaleD_r);
  replacer.register_rule("{transB_r}", transB_r);
  replacer.register_rule("{A_reg_list}", A_reg_list);
  replacer.register_rule("{descB}", descB);
  replacer.register_rule("{scaleA}", std::to_string(scaleA));
  replacer.register_rule("{scaleB}", std::to_string(scaleB));
  replacer.register_rule("{scaleD}", scaleD);
  replacer.register_rule("{transB}", transB ? "1" : "0");
  replacer.register_rule("{accum_r_list}", accum_r_list);
  replacer.register_rule("{accum_list}", accum_list);

  transpose_r_code = allow_transpose ? replacer.rewrite(transpose_r_code) : "";
  transpose_code = allow_transpose ? replacer.rewrite(transpose_code) : "";
  replacer.register_rule("{transpose_r_code}", transpose_r_code);
  replacer.register_rule("{transpose_code}", transpose_code);

  return replacer.rewrite(asm_code);
}

std::string PrintWGMMAArriveAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_wgmma_fence";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintWGMMACommitGroupAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_wgmma_commit_group";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintWGMMAWaitGroupAssembly(CodeGenCUDA* cg, const std::string& N) {
  std::string func_code = R"(
template<int N>
__forceinline__ __device__ void {func_name}() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}
)";
  std::string caller_code = "{func_name}<{N}>();\n";

  std::string func_name = "ptx_wgmma_wait_group";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{N}", N);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintEncodeWgmmaMatrixDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                             const std::string& addr, const std::string& ldo,
                                             const std::string& sdo, int swizzle) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {
  GmmaDescriptor _desc;

  switch (swizzle) {
    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle
    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle
    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle
  }

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);
  
  constexpr uint8_t base_offset = 0;
  _desc.bitfield.base_offset_ = base_offset;

  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}
)";
  std::string caller_code = "{func_name}({desc}, {addr}, {ldo}, {sdo}, {swizzle});\n";

  std::string func_name = "ptx_wgmma_encode_matrix_descriptor";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{desc}", desc);
    replacer.register_rule("{addr}", addr);
    replacer.register_rule("{ldo}", ldo);
    replacer.register_rule("{sdo}", sdo);
    replacer.register_rule("{swizzle}", std::to_string(swizzle));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintBarrierClusterArriveAssembly(const std::string& sem, bool aligned) {
  std::string asm_code = R"(/* T.ptx_barrier_cluster_arrive() */ {
  asm volatile("barrier.cluster.arrive{sem}{aligned};\n" : :);
}
)";
  Replacer replacer;
  replacer.register_rule("{sem}", sem.empty() ? "" : "." + sem);
  replacer.register_rule("{aligned}", aligned ? ".aligned" : "");
  return replacer.rewrite(asm_code);
}

std::string PrintBarrierClusterWaitAssembly(bool acquire, bool aligned) {
  std::string asm_code = R"(/* T.ptx_barrier_cluster_wait() */ {
  asm volatile("barrier.cluster.wait{acquire}{aligned};\n" : :);
}
)";
  Replacer replacer;
  replacer.register_rule("{acquire}", acquire ? ".acquire" : "");
  replacer.register_rule("{aligned}", aligned ? ".aligned" : "");
  return replacer.rewrite(asm_code);
}

std::string PrintElectSyncAssembly(CodeGenCUDA* cg, uint32_t mask) {
  std::string func_code = R"(
__forceinline__ __device__ uint32_t {func_name}(uint32_t mask) {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %rx;\n"
      ".reg .pred %px;\n"
      "     elect.sync %rx|%px, %2;\n"
      "@%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(mask));
  return pred;
}  
)";
  std::string func_name = "elect_one_sync";

  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  func_code = replacer.rewrite(func_code);

  cg->AddUtilFunction(func_name, func_code);
  return func_name + "(" + std::to_string(mask) + ")";
}

std::string PrintFenceMbarrierInitReleaseClusterAssembly(codegen::CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}" ::);
}
)";
  std::string func_name = "ptx_fence_mbarrier_init_release_cluster";

  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  func_code = replacer.rewrite(func_code);

  cg->AddUtilFunction(func_name, func_code);
  return func_name + "();\n";
}

std::string PrintStmatrixSyncAlignedAssembly(int num, bool trans, const std::string& ptr,
                                             const std::vector<std::string>& vars) {
  std::string asm_code = R"(/* T.store_matrix_sync_aligned() */ {
  unsigned int addr = __cvta_generic_to_shared({ptr});
  half2 half_pairs[{num}];
{assign_half_code}
  asm volatile("stmatrix.sync.aligned.m8n8.x{num}{trans}.shared.b16 [%0], {{half_pairs_r}};\n" : : "r"(addr){half_pairs});
}
)";
  Replacer replacer;
  replacer.register_rule("{ptr}", ptr);
  replacer.register_rule("{num}", std::to_string(num));
  replacer.register_rule("{trans}", trans ? ".trans" : "");

  std::string assign_half_code;
  TVM_FFI_ICHECK_EQ(vars.size(), 2 * num);
  for (int i = 0; i < num; ++i) {
    assign_half_code += "   half_pairs[" + std::to_string(i) + "] = {" + vars[2 * i] + ", " +
                        vars[2 * i + 1] + "};\n";
  }
  replacer.register_rule("{assign_half_code}", assign_half_code);
  std::string half_pairs_r = "%1";
  for (int i = 1; i < num; ++i) {
    half_pairs_r += ", %" + std::to_string(i + 1);
  }
  replacer.register_rule("{half_pairs_r}", half_pairs_r);
  std::string half_pairs;
  for (int i = 0; i < num; ++i) {
    half_pairs += ", \"r\"(*(uint32_t *)&half_pairs[" + std::to_string(i) + "])";
  }
  replacer.register_rule("{half_pairs}", half_pairs);
  return replacer.rewrite(asm_code);
}

std::string PrintSetMaxNRegAssembly(bool inc, int reg_count) {
  std::string asm_code = R"(/* T.set_maxnreg() */ {
   asm volatile( "setmaxnreg{action}.sync.aligned.u32 %0;\n" : : "n"({reg_count}) );
}
)";
  Replacer replacer;
  replacer.register_rule("{action}", inc ? ".inc" : ".dec");
  replacer.register_rule("{reg_count}", std::to_string(reg_count));
  return replacer.rewrite(asm_code);
}

std::string PrintTcgen05AllocAssembly(CodeGenCUDA* cg, const std::string& dst_shared_ptr,
                                      const std::string& n_cols, int n_cta_group) {
  std::string func_code = R"(
template <int cta_group>
__forceinline__ __device__ void {func_name}(void* dst, int nCols) {
  unsigned int smem_addr = __cvta_generic_to_shared(dst);
  if (cta_group == 1) {
    __asm__ __volatile__(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(smem_addr), "r"(nCols)
      : "memory"
    );
  } else {
    __asm__ __volatile__(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(smem_addr), "r"(nCols)
      : "memory"
    );
  }
}
)";
  std::string func_name = "ptx_tcgen05_alloc";
  std::string caller_code = "{func_name}<{cta_group}>({dst}, {nCols});\n";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{dst}", dst_shared_ptr);
    replacer.register_rule("{nCols}", n_cols);
    replacer.register_rule("{cta_group}", std::to_string(n_cta_group));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05DeallocAssembly(CodeGenCUDA* cg, const std::string& taddr,
                                        const std::string& n_cols, int n_cta_group) {
  std::string func_code = R"(
template <int cta_group>
__forceinline__ __device__ void {func_name}(uint32_t taddr, int nCols) {
  if (cta_group == 1) {
    __asm__ __volatile__(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
      :: "r"(taddr), "r"(nCols)
      : "memory"
    );
  } else {
    __asm__ __volatile__(
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
      :: "r"(taddr), "r"(nCols)
      : "memory"
    );
  }
}
)";
  std::string func_name = "ptx_tcgen05_dealloc";
  std::string caller_code = "{func_name}<{cta_group}>({taddr}, {nCols});\n";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{taddr}", taddr);
    replacer.register_rule("{nCols}", n_cols);
    replacer.register_rule("{cta_group}", std::to_string(n_cta_group));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05RelinquishAllocPermitAssembly(CodeGenCUDA* cg, int n_cta_group) {
  std::string func_code = R"(
template <int cta_group>
__forceinline__ __device__ void {func_name}() {
  if (cta_group == 1) {
    __asm__ __volatile__(
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
      ::: "memory"
    );
  } else {
    __asm__ __volatile__(
      "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
      ::: "memory"
    );
  }
}
)";
  std::string func_name = "ptx_tcgen05_relinquish_alloc_permit";
  std::string caller_code = "{func_name}<{cta_group}>();\n";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{cta_group}", std::to_string(n_cta_group));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05FenceBeforeThreadSyncAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_tcgen05_fence_before_thread_sync";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05FenceAfterThreadSyncAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_tcgen05_fence_after_thread_sync";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05LoadAssembly(CodeGenCUDA* cg, const std::string& src_addr,
                                     const std::string& row_offset, const std::string& col_offset,
                                     const std::vector<std::string>& regs, const std::string& shape,
                                     int num, bool pack) {
  std::string asm_template = R"(
{
    /* T.ptx_tcgen05_ld() */
    asm volatile(
        "tcgen05.ld.sync.aligned.${shape}.x${num}${pack}.b32 "
        "{${regs_placeholder}}, "
        "[%${src_placeholder}]${imm_arg};\n"
        :  ${reg_operands}
        :  "r"(get_tmem_addr(${src_addr}, ${row_offset}, ${col_offset}))
    );
}
)";

  std::string regs_placeholder = "";
  for (size_t i = 0; i < regs.size(); ++i) {
    regs_placeholder += "%" + std::to_string(i);
    if (i != regs.size() - 1) {
      regs_placeholder += ", ";
    }
  }
  std::string src_placeholder = std::to_string(regs.size());
  std::string imm_arg = "";
  if (shape == "16x32bx2") {
    int imm = pack ? 2 * num : num;
    imm_arg = ", " + std::to_string(imm);
  }

  std::string reg_operands = "";
  for (size_t i = 0; i < regs.size(); ++i) {
    reg_operands += "\"=r\"(*(uint32_t*)&" + regs[i] + ")";
    if (i != regs.size() - 1) {
      reg_operands += ", ";
    }
  }

  Replacer replacer;
  replacer.register_rule("${shape}", shape);
  replacer.register_rule("${num}", std::to_string(num));
  replacer.register_rule("${pack}", pack ? ".pack::16b" : "");
  replacer.register_rule("${regs_placeholder}", regs_placeholder);
  replacer.register_rule("${src_placeholder}", src_placeholder);
  replacer.register_rule("${imm_arg}", imm_arg);
  replacer.register_rule("${reg_operands}", reg_operands);
  replacer.register_rule("${src_addr}", src_addr);
  replacer.register_rule("${row_offset}", row_offset);
  replacer.register_rule("${col_offset}", col_offset);

  return replacer.rewrite(asm_template);
}

std::string PrintTcgen05StoreAssembly(CodeGenCUDA* cg, const std::string& dst_addr,
                                      const std::string& row_offset, const std::string& col_offset,
                                      const std::vector<std::string>& regs,
                                      const std::string& shape, int num, bool unpack) {
  std::string asm_template = R"(
{
    /* T.ptx_tcgen05_st() */
    asm volatile(
        "tcgen05.st.sync.aligned.${shape}.x${num}${unpack}.b32 "
        "[%0]${imm_arg}, "
        "{${regs_placeholder}};\n"
        :
        :  "r"(get_tmem_addr(${dst_addr}, ${row_offset}, ${col_offset})), ${reg_operands}
    );
}
)";

  std::string regs_placeholder = "";
  for (size_t i = 0; i < regs.size(); ++i) {
    regs_placeholder += "%" + std::to_string(1 + i);
    if (i != regs.size() - 1) {
      regs_placeholder += ", ";
    }
  }
  std::string imm_arg = "";
  if (shape == "16x32bx2") {
    int imm = unpack ? 2 * num : num;
    imm_arg = ", " + std::to_string(imm);
  }

  std::string reg_operands = "";
  for (size_t i = 0; i < regs.size(); ++i) {
    reg_operands += "\"r\"(*(uint32_t*)&" + regs[i] + ")";
    if (i != regs.size() - 1) {
      reg_operands += ", ";
    }
  }

  Replacer replacer;
  replacer.register_rule("${shape}", shape);
  replacer.register_rule("${num}", std::to_string(num));
  replacer.register_rule("${unpack}", unpack ? ".unpack::16b" : "");
  replacer.register_rule("${imm_arg}", imm_arg);
  replacer.register_rule("${regs_placeholder}", regs_placeholder);
  replacer.register_rule("${dst_addr}", dst_addr);
  replacer.register_rule("${row_offset}", row_offset);
  replacer.register_rule("${col_offset}", col_offset);
  replacer.register_rule("${reg_operands}", reg_operands);

  return replacer.rewrite(asm_template);
}

std::string PrintTcgen05WaitLdSyncAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_tcgen05_wait_ld";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05WaitStSyncAssembly(CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}
)";
  std::string caller_code = "{func_name}();\n";

  std::string func_name = "ptx_tcgen05_wait_st";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintEncodeTcgen05MatrixDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                               const std::string& addr, const std::string& ldo,
                                               const std::string& sdo, int swizzle) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {
  SmemDescriptor _desc;

  _desc.version_ = 1;
  _desc.lbo_mode_ = 0;

  switch (swizzle) {
    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle
    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle
    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle
    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle
  }

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.base_offset_ = base_offset;

  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}
)";
  std::string caller_code = "{func_name}({desc}, {addr}, {ldo}, {sdo}, {swizzle});\n";

  std::string func_name = "ptx_tcgen05_encode_matrix_descriptor";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{desc}", desc);
    replacer.register_rule("{addr}", addr);
    replacer.register_rule("{ldo}", ldo);
    replacer.register_rule("{sdo}", sdo);
    replacer.register_rule("{swizzle}", std::to_string(swizzle));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string GetTcgen05MMAKind(const std::string& d_dtype, const std::string& a_dtype,
                              const std::string& b_dtype, const std::string& sfa_dtype = "",
                              const std::string& sfb_dtype = "") {
  // ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape

  using namespace ptx;
  std::string kind;
  DataType dtype = DTypeFromString(d_dtype);
  DataType atype = DTypeFromString(a_dtype);
  DataType btype = DTypeFromString(b_dtype);

  if (atype == DataType::kFloat16 && btype == DataType::kFloat16 && dtype == DataType::kFloat16) {
    kind = "f16";
  } else if ((atype == DataType::kBFloat16 || atype == DataType::kFloat16) &&
             (btype == DataType::kBFloat16 || btype == DataType::kFloat16) &&
             dtype == DataType::kFloat32) {
    kind = "f16";
  } else if (atype == DataType::kTensorFloat32 && btype == DataType::kTensorFloat32 &&
             dtype == DataType::kFloat32) {
    kind = "tf32";
  } else if (atype == DataType::kFloat4_e2m1fn && btype == DataType::kFloat4_e2m1fn &&
             !sfa_dtype.empty() && !sfb_dtype.empty() && dtype == DataType::kFloat32) {
    if (DTypeFromString(sfa_dtype) == DataType::kFloat8_e8m0fnu &&
        DTypeFromString(sfb_dtype) == DataType::kFloat8_e8m0fnu) {
      kind = "mxf4";
    } else if ((DTypeFromString(sfa_dtype) == DataType::kFloat8_e4m3fn ||
                DTypeFromString(sfa_dtype) == DataType::kFloat8_e4m3fnuz) &&
               (DTypeFromString(sfb_dtype) == DataType::kFloat8_e4m3fn ||
                DTypeFromString(sfb_dtype) == DataType::kFloat8_e4m3fnuz)) {
      kind = "mxf4nvf4";
    }
  } else if ((atype == DataType::kFloat8_e4m3fn || atype == DataType::kFloat8_e4m3fnuz ||
              atype == DataType::kFloat8_e5m2 || atype == DataType::kFloat6_e2m3fn ||
              atype == DataType::kFloat6_e3m2fn || atype == DataType::kFloat4_e2m1fn) &&
             (btype == DataType::kFloat8_e4m3fn || btype == DataType::kFloat8_e4m3fnuz ||
              btype == DataType::kFloat8_e5m2 || btype == DataType::kFloat6_e2m3fn ||
              btype == DataType::kFloat6_e3m2fn || btype == DataType::kFloat4_e2m1fn)) {
    if (sfa_dtype.empty() && sfb_dtype.empty() &&
        (dtype == DataType::kFloat32 || dtype == DataType::kFloat16)) {
      kind = "f8f6f4";
    } else if (!sfa_dtype.empty() && !sfb_dtype.empty() &&
               DTypeFromString(sfa_dtype) == DataType::kFloat8_e8m0fnu &&
               DTypeFromString(sfb_dtype) == DataType::kFloat8_e8m0fnu &&
               dtype == DataType::kFloat32) {
      kind = "mxf8f6f4";
    }
  } else if ((atype == DataType::kInt8 || atype == DataType::kUInt8) &&
             (btype == DataType::kInt8 || btype == DataType::kUInt8) && dtype == DataType::kInt32) {
    kind = "i8";
  }

  CHECK(!kind.empty()) << "Invalid multiplicand data types for Tcgen05 MMA, check failed for d: "
                       << d_dtype << ", a: " << a_dtype << ", b: " << b_dtype
                       << ", scale_a: " << sfa_dtype << ", scale_b: " << sfb_dtype;
  return kind;
}

bool CheckTcgen05MMAMatrixShape(const std::string& kind, int cta_group, int M, int N, int K,
                                bool is_sparse) {
  // ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape

  std::string err = "Invalid matrix shape for Tcgen05 MMA, check failed for kind: " + kind +
                    ", is_sparse: " + std::to_string(is_sparse) +
                    ", cta_group: " + std::to_string(cta_group) + ", M: " + std::to_string(M) +
                    ", N: " + std::to_string(N) + ", K: " + std::to_string(K);
  // check for M/N
  if (kind == "f16" || kind == "tf32" || kind == "f8f6f4") {
    if (cta_group == 1) {
      if (M == 64) {
        CHECK(8 <= N && N <= 256 && N % 8 == 0) << err;
      } else if (M == 128) {
        CHECK(16 <= N && N <= 256 && N % 16 == 0) << err;
      } else {
        CHECK(false) << err;
      }
    } else if (cta_group == 2) {
      CHECK(M == 128 || M == 256) << err;
      CHECK(32 <= N && N <= 256 && N % 32 == 0) << err;
    }
  } else if (kind == "i8") {
    if (cta_group == 1) {
      CHECK(M == 64 || M == 128) << err;
      CHECK(N == 8 || N == 24 || (16 <= N && N <= 256 && N % 16 == 0)) << err;
    } else if (cta_group == 2) {
      CHECK(M == 128 || M == 256) << err;
      CHECK(32 <= N && N <= 256 && N % 32 == 0) << err;
    }
  } else if (kind == "mxf8f6f4" || kind == "mxf4" || kind == "mxf4nvf4") {
    if (cta_group == 1) {
      CHECK(M == 128) << err;
      CHECK(8 <= N && N <= 256 && N % 8 == 0) << err;
    } else if (cta_group == 2) {
      if (is_sparse) {
        CHECK(M == 256) << err;
      } else {
        CHECK(M == 128 || M == 256) << err;
      }
      CHECK(16 <= N && N <= 256 && N % 16 == 0) << err;
    }
  } else {
    CHECK(false) << err;
  }
  // check for K
  if (kind == "f16") {
    CHECK(K == (is_sparse ? 32 : 16)) << err;
  } else if (kind == "tf32") {
    CHECK(K == (is_sparse ? 16 : 8)) << err;
  } else if (kind == "f8f6f4" || kind == "i8" || kind == "mxf8f6f4") {
    CHECK(K == (is_sparse ? 64 : 32)) << err;
  } else if (kind == "mxf4" || kind == "mxf4nvf4") {
    CHECK(K == (is_sparse ? 128 : 64)) << err;
  } else {
    CHECK(false) << err;
  }

  return true;
}

std::string PrintEncodeTcgen05InstrDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                              const std::string& d_dtype,
                                              const std::string& a_dtype,
                                              const std::string& b_dtype, int M, int N, int K,
                                              bool trans_a, bool trans_b, int n_cta_group,
                                              bool neg_a, bool neg_b, bool sat_d, bool is_sparse) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int d_format,
                                            int a_format, int b_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool sat_d, bool is_sparse) {
  InstrDescriptor _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.c_format_ = uint8_t(d_format);

  _desc.m_dim_ = (M >> 4);
  _desc.n_dim_ = (N >> 3);

  _desc.a_major_ = static_cast<uint8_t>(trans_a);
  _desc.b_major_ = static_cast<uint8_t>(trans_b);

  _desc.a_negate_ = static_cast<uint8_t>(neg_a);
  _desc.b_negate_ = static_cast<uint8_t>(neg_b);
  _desc.saturate_ = static_cast<uint8_t>(sat_d);

  _desc.sparse_flag_ = is_sparse;
  _desc.sparse_id2_  = 0;                          // should modify in sparse case

  _desc.max_shift_ = uint8_t(0);                   // WS not used

  *desc = (uint32_t)_desc;
}
)";

  std::string kind = GetTcgen05MMAKind(d_dtype, a_dtype, b_dtype);
  CHECK(kind == "f16" || kind == "tf32" || kind == "f8f6f4" || kind == "i8")
      << "Check failed for Data Type for tcgen05 instruction descriptor: d_dtype: " << d_dtype
      << ", a_dtype: " << a_dtype << ", b_dtype: " << b_dtype;
  CHECK(CheckTcgen05MMAMatrixShape(kind, n_cta_group, M, N, K, is_sparse));

  using namespace ptx;
  std::unordered_map<DataType, int> format_map = {{DataType::kFloat16, 0},
                                                  {DataType::kBFloat16, 1},
                                                  {DataType::kTensorFloat32, 2},
                                                  {DataType::kFloat8_e4m3fn, 0},
                                                  {DataType::kFloat8_e4m3fnuz, 0},
                                                  {DataType::kFloat8_e5m2, 1},
                                                  {DataType::kFloat6_e2m3fn, 3},
                                                  {DataType::kFloat6_e3m2fn, 4},
                                                  {DataType::kFloat4_e2m1fn, 5},
                                                  {DataType::kUInt8, 0},
                                                  {DataType::kInt8, 1},
                                                  {DataType::kFloat32, 1},
                                                  {DataType::kInt32, 2}};
  DataType dtype = DTypeFromString(d_dtype);
  DataType atype = DTypeFromString(a_dtype);
  DataType btype = DTypeFromString(b_dtype);
  int d_format = format_map[dtype];
  int a_format = format_map[atype];
  int b_format = format_map[btype];

  std::vector<DataType> valid_dtypes_for_trans = {
      DataType::kFloat8_e4m3fn, DataType::kFloat8_e4m3fnuz, DataType::kFloat8_e5m2,
      DataType::kInt8,          DataType::kUInt8,           DataType::kFloat16,
      DataType::kBFloat16,      DataType::kTensorFloat32,
  };
  if (trans_a) {
    CHECK(std::find(valid_dtypes_for_trans.begin(), valid_dtypes_for_trans.end(), atype) !=
          valid_dtypes_for_trans.end())
        << "Invalid a_dtype for transpose in instruction descriptor, a_dtype: " << a_dtype;
  }
  if (trans_b) {
    CHECK(std::find(valid_dtypes_for_trans.begin(), valid_dtypes_for_trans.end(), btype) !=
          valid_dtypes_for_trans.end())
        << "Invalid b_dtype for transpose in instruction descriptor, b_dtype: " << b_dtype;
  }
  if (neg_a || neg_b) {
    CHECK(kind == "f16" || kind == "tf32" || kind == "f8f6f4")
        << "Invalid kind for negate in instruction descriptor, kind: " << kind;
  }
  if (sat_d) {
    CHECK(kind == "i8") << "Invalid kind for saturate in instruction descriptor, kind: " << kind;
  }

  std::string caller_code =
      "{func_name}({desc}, {M}, {N}, {d_format}, {a_format}, {b_format}, {trans_a}, {trans_b}, "
      "{neg_a}, {neg_b}, {sat_d}, {is_sparse});\n";
  std::string func_name = "ptx_tcgen05_encode_instr_descriptor";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{desc}", desc);
    replacer.register_rule("{M}", std::to_string(M));
    replacer.register_rule("{N}", std::to_string(N));
    replacer.register_rule("{d_format}", std::to_string(d_format));
    replacer.register_rule("{a_format}", std::to_string(a_format));
    replacer.register_rule("{b_format}", std::to_string(b_format));
    replacer.register_rule("{trans_a}", trans_a ? "true" : "false");
    replacer.register_rule("{trans_b}", trans_b ? "true" : "false");
    replacer.register_rule("{neg_a}", neg_a ? "true" : "false");
    replacer.register_rule("{neg_b}", neg_b ? "true" : "false");
    replacer.register_rule("{sat_d}", sat_d ? "true" : "false");
    replacer.register_rule("{is_sparse}", is_sparse ? "true" : "false");
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintEncodeTcgen05InstrDescriptorBlockScaled(
    codegen::CodeGenCUDA* cg, const std::string& desc, const std::string& d_dtype,
    const std::string& a_dtype, const std::string& b_dtype, const std::string& sfa_dtype,
    const std::string& sfb_dtype, const std::string& sfa_tmem_addr,
    const std::string& sfb_tmem_addr, int M, int N, int K, bool trans_a, bool trans_b,
    int n_cta_group, bool neg_a, bool neg_b, bool is_sparse) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int a_format,
                                            int b_format, int s_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool is_sparse,
                                            uint32_t sfa_tmem_addr, uint32_t sfb_tmem_addr) {
  InstrDescriptorBlockScaled _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.scale_format_ = uint8_t(s_format);

  _desc.a_sf_id_ = (sfa_tmem_addr & 0xC0000000) >> 30;
  _desc.b_sf_id_ = (sfb_tmem_addr & 0xC0000000) >> 30;

  _desc.m_dim_ = (M >> 4);
  _desc.n_dim_ = (N >> 3);

  _desc.a_major_ = static_cast<uint8_t>(trans_a);
  _desc.b_major_ = static_cast<uint8_t>(trans_b);

  _desc.a_negate_ = static_cast<uint8_t>(neg_a);
  _desc.b_negate_ = static_cast<uint8_t>(neg_b);

  _desc.sparse_flag_ = is_sparse;
  _desc.sparse_id2_  = 0;                          // should modify in sparse case

  *desc = (uint32_t)_desc;
}
)";

  std::string kind = GetTcgen05MMAKind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype);
  CHECK(kind == "mxf8f6f4" || kind == "mxf4" || kind == "mxf4nvf4")
      << "Check failed for Data Type for tcgen05 instruction descriptor: d_dtype: " << d_dtype
      << ", a_dtype: " << a_dtype << ", b_dtype: " << b_dtype << ", scale_a: " << sfa_dtype
      << ", scale_b: " << sfb_dtype;
  CHECK(CheckTcgen05MMAMatrixShape(kind, n_cta_group, M, N, K, is_sparse));

  using namespace ptx;
  int a_format, b_format, s_format;
  std::unordered_map<DataType, int> format_map = {
      {DataType::kFloat8_e4m3fn, 0}, {DataType::kFloat8_e4m3fnuz, 0}, {DataType::kFloat8_e5m2, 1},
      {DataType::kFloat6_e2m3fn, 3}, {DataType::kFloat6_e3m2fn, 4},   {DataType::kFloat4_e2m1fn, 5},
  };
  std::unordered_map<DataType, int> format_map_sf = {
      {DataType::kFloat8_e4m3fn, 0},
      {DataType::kFloat8_e4m3fnuz, 0},
      {DataType::kFloat8_e8m0fnu, 1},
  };

  DataType atype = DTypeFromString(a_dtype);
  DataType btype = DTypeFromString(b_dtype);
  DataType stype = DTypeFromString(sfa_dtype);
  if (kind == "mxf8f6f4") {
    a_format = format_map[atype];
    b_format = format_map[btype];
  } else {
    a_format = 1;  // E2M1
    b_format = 1;  // E2M1
  }
  s_format = format_map_sf[stype];

  std::vector<DataType> valid_dtypes_for_trans = {
      DataType::kFloat8_e4m3fn,
      DataType::kFloat8_e4m3fnuz,
      DataType::kFloat8_e5m2,

  };
  if (trans_a) {
    CHECK(std::find(valid_dtypes_for_trans.begin(), valid_dtypes_for_trans.end(), atype) !=
          valid_dtypes_for_trans.end())
        << "Invalid a_dtype for transpose in instruction descriptor, a_dtype: " << a_dtype;
  }
  if (trans_b) {
    CHECK(std::find(valid_dtypes_for_trans.begin(), valid_dtypes_for_trans.end(), btype) !=
          valid_dtypes_for_trans.end())
        << "Invalid b_dtype for transpose in instruction descriptor, b_dtype: " << b_dtype;
  }

  std::string caller_code =
      "{func_name}({desc}, {M}, {N}, {a_format}, {b_format}, {s_format}, {trans_a}, "
      "{trans_b}, {neg_a}, {neg_b}, {is_sparse}, {sfa_tmem_addr}, "
      "{sfb_tmem_addr});\n";
  std::string func_name = "ptx_tcgen05_encode_instr_descriptor_block_scaled";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{desc}", desc);
    replacer.register_rule("{M}", std::to_string(M));
    replacer.register_rule("{N}", std::to_string(N));
    replacer.register_rule("{a_format}", std::to_string(a_format));
    replacer.register_rule("{b_format}", std::to_string(b_format));
    replacer.register_rule("{s_format}", std::to_string(s_format));
    replacer.register_rule("{trans_a}", trans_a ? "true" : "false");
    replacer.register_rule("{trans_b}", trans_b ? "true" : "false");
    replacer.register_rule("{neg_a}", neg_a ? "true" : "false");
    replacer.register_rule("{neg_b}", neg_b ? "true" : "false");
    replacer.register_rule("{is_sparse}", is_sparse ? "true" : "false");
    replacer.register_rule("{sfa_tmem_addr}", sfa_tmem_addr);
    replacer.register_rule("{sfb_tmem_addr}", sfb_tmem_addr);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

int GetTcgen05MMAScaleVecSize(const std::string& kind, const std::string& scale_dtype) {
  // ref:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#valid-combinations-of-scale-vec-size-with-types-and-mma-kind

  using namespace ptx;
  int scale_vec_size = 0;
  DataType stype = DTypeFromString(scale_dtype);
  if (kind == "mxf8f6f4" && stype == DataType::kFloat8_e8m0fnu) {
    scale_vec_size = 1;
  } else if (kind == "mxf4" && stype == DataType::kFloat8_e8m0fnu) {
    scale_vec_size = 2;
  } else if (kind == "mxf4nvf4" && stype == DataType::kFloat8_e8m0fnu) {
    scale_vec_size = 2;
  } else if (kind == "mxf4nvf4" &&
             (stype == DataType::kFloat8_e4m3fn || stype == DataType::kFloat8_e4m3fnuz)) {
    scale_vec_size = 4;
  }
  CHECK_GT(scale_vec_size, 0)
      << "Invalid scale vector size for Tcgen05 MMA, check failed for kind::" << kind
      << ", scale_dtype: " << scale_dtype;
  return scale_vec_size;
}

std::string PrintTcgen05MMAAssembly(CodeGenCUDA* cg, const std::string& d_dtype,
                                    const std::string& a_dtype, const std::string& b_dtype,
                                    const std::string& d_tmem_addr, const std::string& a_operand,
                                    const std::string& b_desc, const std::string& i_desc,
                                    bool use_a_tmem, int cta_group,
                                    const std::vector<std::string>& disable_output_lane,
                                    bool enable_input_d, int scale_input_d, bool sparse,
                                    const std::string& sp_tmem_addr) {
  std::string asm_template = R"(
{
    /* T.ptx_tcgen05_mma${sparse_func}() */
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, ${p_operand}, 0;\n"
        "tcgen05.mma${sparse}.cta_group::${cta_group}.kind::${kind} [%0], ${a_operand}, %2, ${i_sp_operand} "
        "{${mask_list}}, p${scale_placeholder};\n"
        "}\n"
        :
        : ${input_operands}
    );
}
)";

  std::string sparse_func = sparse ? "_sp" : "";
  std::string kind = GetTcgen05MMAKind(d_dtype, a_dtype, b_dtype);
  CHECK(kind == "f16" || kind == "tf32" || kind == "f8f6f4" || kind == "i8")
      << "Check failed for Data Type for tcgen05 MMA operation: d_dtype: " << d_dtype
      << ", a_dtype: " << a_dtype << ", b_dtype: " << b_dtype;
  std::string p_operand_str = sparse ? "%5" : "%4";
  std::string sparse_str = sparse ? ".sp" : "";
  std::string a_operand_str = use_a_tmem ? "[%1]" : "%1";
  std::string i_sp_operand_str = sparse ? "[%3], %4," : "%3,";
  std::string mask_placeholders;
  size_t start = sparse ? 6 : 5;
  for (size_t i = 0; i < disable_output_lane.size(); ++i) {
    mask_placeholders += "%" + std::to_string(start + i);
    if (i != disable_output_lane.size() - 1) {
      mask_placeholders += ", ";
    }
  }
  if (scale_input_d > 0) {
    CHECK(kind == "f16" || kind == "tf32")
        << "scale_input_d is only valid for kind::f16 or kind::tf32, not valid for kind::" << kind;
  }
  std::string scale_placeholder;
  if (enable_input_d && scale_input_d > 0) {
    scale_placeholder = ", %" + std::to_string(start + disable_output_lane.size());
  }

  std::string a_constraint = use_a_tmem ? "\"r\"" : "\"l\"";
  std::string enable_input_d_str = enable_input_d ? "1" : "0";
  std::string input_operands = "\"r\"(" + d_tmem_addr + "), "            // %0
                               + a_constraint + "(" + a_operand + "), "  // %1
                               + "\"l\"(" + b_desc + "), ";              // %2
  if (sparse) {
    input_operands += "\"r\"(" + sp_tmem_addr + "), ";  // %3 (sparse)
  }
  input_operands += "\"r\"(" + i_desc + "), "                 // %3 or %4 (sparse)
                    + "\"r\"(" + enable_input_d_str + "), ";  // %4 or %5 (sparse)
  for (size_t i = 0; i < disable_output_lane.size(); ++i) {
    input_operands += "\"r\"(" + disable_output_lane[i] + ")";
    if (i != disable_output_lane.size() - 1) {
      input_operands += ", ";
    }
  }
  if (enable_input_d && scale_input_d > 0) {
    input_operands += ", \"n\"(" + std::to_string(scale_input_d) + ")";
  }

  Replacer replacer;
  replacer.register_rule("${sparse_func}", sparse_func);
  replacer.register_rule("${p_operand}", p_operand_str);
  replacer.register_rule("${sparse}", sparse_str);
  replacer.register_rule("${cta_group}", std::to_string(cta_group));
  replacer.register_rule("${kind}", kind);
  replacer.register_rule("${a_operand}", a_operand_str);
  replacer.register_rule("${i_sp_operand}", i_sp_operand_str);
  replacer.register_rule("${mask_list}", mask_placeholders);
  replacer.register_rule("${scale_placeholder}", scale_placeholder);
  replacer.register_rule("${input_operands}", input_operands);

  return replacer.rewrite(asm_template);
}

std::string PrintTcgen05MMABlockScaleAssembly(
    CodeGenCUDA* cg, const std::string& d_dtype, const std::string& a_dtype,
    const std::string& b_dtype, const std::string& sfa_dtype, const std::string& sfb_dtype,
    const std::string& d_tmem_addr, const std::string& a_operand, const std::string& b_desc,
    const std::string& sfa_tmem_addr, const std::string& sfb_tmem_addr, const std::string& i_desc,
    bool use_a_tmem, int cta_group, bool enable_input_d, bool sparse,
    const std::string& sp_tmem_addr) {
  std::string asm_template = R"(
{
    /* T.ptx_tcgen05_mma${sparse_func}_block_scale() */
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma${sparse}.cta_group::${cta_group}.kind::${kind}.block_scale.scale_vec::${scale_vec_size}X "
        "[%0], ${a_operand_placeholder}, %2, ${sparse_placeholder}%3, [%5], [%6], p;\n"
        "}\n"
        :
        : "r"(${d_tmem}), ${a_constraint}(${a_operand}), "l"(${b_desc}), "r"(${i_desc}),
          "r"(${enable_input_d}), "r"(${sfa_tmem_addr}), "r"(${sfb_tmem_addr})${sp_tmem_addr}
    );
}
)";

  std::string sparse_func = sparse ? "_sp" : "";
  std::string kind = GetTcgen05MMAKind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype);
  int scale_vec_size = GetTcgen05MMAScaleVecSize(kind, sfa_dtype);
  CHECK(kind == "mxf8f6f4" || kind == "mxf4" || kind == "mxf4nvf4")
      << "Check failed for Data Type for tcgen05 MMA operation with block scale: d_dtype: "
      << d_dtype << ", a_dtype: " << a_dtype << ", b_dtype: " << b_dtype
      << ", sfa_dtype: " << sfa_dtype << ", sfb_dtype: " << sfb_dtype;
  std::string sparse_str = sparse ? ".sp" : "";
  std::string a_operand_placeholder = use_a_tmem ? "[%1]" : "%1";
  std::string sparse_placeholder = sparse ? "[%7], " : "";
  std::string a_constraint = use_a_tmem ? "\"r\"" : "\"l\"";
  std::string enable_input_d_str = enable_input_d ? "1" : "0";
  std::string sp_tmem_addr_str = sparse ? ", \"r\"(" + sp_tmem_addr + ")" : "";

  Replacer replacer;
  replacer.register_rule("${sparse_func}", sparse_func);
  replacer.register_rule("${sparse}", sparse_str);
  replacer.register_rule("${cta_group}", std::to_string(cta_group));
  replacer.register_rule("${kind}", kind);
  replacer.register_rule("${scale_vec_size}", std::to_string(scale_vec_size));
  replacer.register_rule("${a_operand_placeholder}", a_operand_placeholder);
  replacer.register_rule("${sparse_placeholder}", sparse_placeholder);
  replacer.register_rule("${d_tmem}", d_tmem_addr);
  replacer.register_rule("${a_constraint}", a_constraint);
  replacer.register_rule("${a_operand}", a_operand);
  replacer.register_rule("${b_desc}", b_desc);
  replacer.register_rule("${i_desc}", i_desc);
  replacer.register_rule("${enable_input_d}", enable_input_d_str);
  replacer.register_rule("${sfa_tmem_addr}", sfa_tmem_addr);
  replacer.register_rule("${sfb_tmem_addr}", sfb_tmem_addr);
  replacer.register_rule("${sp_tmem_addr}", sp_tmem_addr_str);

  return replacer.rewrite(asm_template);
}

std::string PrintTcgen05CommitAssembly(CodeGenCUDA* cg, const std::string& bar, int cta_group,
                                       int cta_mask) {
  std::string func_code = R"(
template <int cta_group>
__forceinline__ __device__ void {func_name}(void* bar, int cta_mask_) {
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_);
  if (cta_group == 1) {
    __asm__ __volatile__(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster{multicast}.b64 [%0]{mask_operand};"
      :
      :"r"(bar_addr){cta_mask_arg}
    );
  } else {
    __asm__ __volatile__(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster{multicast}.b64 [%0]{mask_operand};"
      :
      :"r"(bar_addr){cta_mask_arg}
    );
  }
}
)";

  std::string caller_code =
      "{func_name}<{cta_group}>(reinterpret_cast<void*>({bar}), {cta_mask});\n";
  std::string func_name = "ptx_tcgen05_commit{multicast_func}";
  {
    // func name
    Replacer replacer;
    replacer.register_rule("{multicast_func}", (cta_mask != 0) ? "_multicast" : "");
    func_name = replacer.rewrite(func_name);
  }
  {  // func code
    Replacer replacer;
    std::string multicast = (cta_mask != 0) ? ".multicast::cluster" : "";
    std::string mask_operand = (cta_mask != 0) ? ", %1" : "";
    std::string cta_mask_arg = (cta_mask != 0) ? ", \"h\"(cta_mask)" : "";

    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{multicast}", multicast);
    replacer.register_rule("{mask_operand}", mask_operand);
    replacer.register_rule("{cta_mask_arg}", cta_mask_arg);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{cta_group}", std::to_string(cta_group));
    replacer.register_rule("{bar}", bar);
    replacer.register_rule("{cta_mask}", std::to_string(cta_mask));
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintTcgen05CopyAssembly(CodeGenCUDA* cg, const std::string& dst_addr,
                                     const std::string& row_offset, const std::string& col_offset,
                                     const std::string& src_desc, const std::string& shape,
                                     const std::string& dst_dtype, const std::string& src_dtype,
                                     int cta_group, const std::string& multicast) {
  std::string asm_template = R"(
{
    /* T.ptx_tcgen05_cp() */
    asm volatile(
        "tcgen05.cp.cta_group::${cta_group}.${shape}${multicast}${dst_src_fmt} [%0], %1;"
        :
        : "r"(get_tmem_addr(${dst_addr}, ${row_offset}, ${col_offset}))  "l"(${src_addr})
    );
}
)";

  // check shape and multicast validity
  CHECK(shape == "128x256b" || shape == "4x256b" || shape == "128x128b" || shape == "64x128b" ||
        shape == "32x128b")
      << "Invalid shape for tcgen05 copy, check failed for shape: " << shape;
  std::string err_msg = "Invalid multicast for tcgen05 copy, check failed for shape: " + shape +
                        ", multicast: " + multicast;
  if (shape == "64x128b") {
    CHECK(multicast == "warpx2::02_13" || multicast == "warpx2::01_23") << err_msg;
  } else if (shape == "32x128b") {
    CHECK(multicast == "warpx4") << err_msg;
  } else {
    CHECK(multicast == "") << err_msg;
  }

  // check data decompression
  using namespace ptx;
  std::string dst_src_fmt = "";
  DataType dtype = DTypeFromString(dst_dtype);
  DataType stype = DTypeFromString(src_dtype);
  if (dtype == DataType::kFloat8_e4m3fn || dtype == DataType::kFloat8_e4m3fnuz ||
      dtype == DataType::kFloat8_e5m2 || dtype == DataType::kFloat8_e8m0fnu) {
    if (stype == DataType::kFloat4_e2m1fn) {
      dst_src_fmt = ".b8x16.b4x16_p64";
    } else if (stype == DataType::kFloat6_e2m3fn || stype == DataType::kFloat6_e3m2fn) {
      dst_src_fmt = ".b8x16.b6x16_p32";
    }
  }

  Replacer replacer;
  replacer.register_rule("${cta_group}", std::to_string(cta_group));
  replacer.register_rule("${shape}", shape);
  replacer.register_rule("${multicast}", multicast != "" ? "." + multicast : "");
  replacer.register_rule("${dst_src_fmt}", dst_src_fmt);
  replacer.register_rule("${dst_addr}", dst_addr);
  replacer.register_rule("${src_addr}", src_desc);
  replacer.register_rule("${row_offset}", row_offset);
  replacer.register_rule("${col_offset}", col_offset);

  return replacer.rewrite(asm_template);
}

std::string PrintTcgen05ShiftAssembly(CodeGenCUDA* cg, const std::string& taddr, int n_cta_group) {
  std::string func_code = R"(
template <int cta_group>
__forceinline__ __device__ void {func_name}(uint32_t taddr) {
  if (cta_group == 1) {
    __asm__ __volatile__(
      "tcgen05.shift.cta_group::1.down %0;"
      :: "r"(taddr)
    );
  } else {
    __asm__ __volatile__(
      "tcgen05.shift.cta_group::2.down %0;"
      :: "r"(taddr)
    );
  }
}
)";
  std::string func_name = "ptx_tcgen05_shift";
  std::string caller_code = "{func_name}<{cta_group}>({taddr});\n";
  {  // func code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    func_code = replacer.rewrite(func_code);
  }
  {  // caller code
    Replacer replacer;
    replacer.register_rule("{func_name}", func_name);
    replacer.register_rule("{cta_group}", std::to_string(n_cta_group));
    replacer.register_rule("{taddr}", taddr);
    caller_code = replacer.rewrite(caller_code);
  }
  cg->AddUtilFunction(func_name, func_code);
  return caller_code;
}

std::string PrintGetTimestampAssembly(codegen::CodeGenCUDA* cg) {
  std::string func_code = R"(
__forceinline__ __device__ uint32_t {func_name}() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}
)";
  std::string func_name = "ptx_get_timestamp";
  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  func_code = replacer.rewrite(func_code);
  cg->AddUtilFunction(func_name, func_code);
  return func_name + "()";
}

std::string PrintLdGlobalAcquireAssembly(codegen::CodeGenCUDA* cg, const std::string& res,
                                         const std::string& addr, DataType dtype) {
  std::string func_code = R"(
__forceinline__ __device__ {dtype} {func_name}({dtype}* addr) {
  {dtype} res;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile ("ld.global.acquire.gpu.{type} %0, [%1];\n" : "=r"(res) : "l"(addr));  
  #else
  asm volatile ("ld.global.cg.{type} %0, [%1];\n" : "=r"(res) : "l"(addr));  
  #endif
  return res;
}
)";
  std::string dtype_str;
  std::string type;
  if (dtype == DataType::UInt(32)) {
    dtype_str = "uint32_t";
    type = "b32";
  } else if (dtype == DataType::Int(32)) {
    dtype_str = "int32_t";
    type = "b32";
  } else if (dtype == DataType::UInt(64)) {
    dtype_str = "uint64_t";
    type = "b64";
  } else if (dtype == DataType::Int(64)) {
    dtype_str = "int64_t";
    type = "b64";
  } else {
    LOG(FATAL) << "Only support uint32/int32/uint64/int64 for ld.global.acquire.";
  }
  std::string func_name = "ptx_ld_global_acquire";
  Replacer replacer;
  replacer.register_rule("{func_name}", func_name);
  replacer.register_rule("{dtype}", dtype_str);
  replacer.register_rule("{type}", type);
  func_code = replacer.rewrite(func_code);
  cg->AddUtilFunction(func_name, func_code);
  return res + " = " + func_name + "(" + addr + ")";
}

}  // namespace codegen
}  // namespace tvm
