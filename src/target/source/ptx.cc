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
  kFloat8_e4m3 = 10,
  kFloat8_e5m2 = 11,
  kFloat16 = 12,
  kBFloat16 = 13,
  kFloat16x2 = 14,
  kFloat32 = 15,
  kTensorFloat32 = 16,
  kFloat64 = 17,
  kBit1 = 18,
  kBit8 = 19,
  kBit16 = 20,
  kBit32 = 21,
  kBit64 = 22
};

static const char* dtype_str[] = {".s4",  ".u4",   ".s8",    ".u8",  ".s16",  ".u16",
                                  ".s32", ".u32",  ".s64",   ".u64", ".e4m3", ".e5m2",
                                  ".f16", ".bf16", ".f16x2", ".f32", ".tf32", ".f64",
                                  ".b1",  ".b8",   ".b16",   ".b32", ".b64"};
static const uint32_t num_bits[] = {4,  4,  8,  8,  16, 16, 32, 32, 64, 64, 8, 8,
                                    16, 16, 32, 32, 32, 64, 1,  8,  16, 32, 64};

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
  } else if (str == "e4m3" || str == ".e4m3") {
    return DataType::kFloat8_e4m3;
  } else if (str == "e5m2" || str == ".e5m2") {
    return DataType::kFloat8_e5m2;
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
  } else if (str == "e4m3_float8") {
    return DataType::kE4M3;
  } else if (str == "e5m2_float8") {
    return DataType::kE5M2;
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
  TVM_FFI_ICHECK(pos_m != str.npos && pos_n != str.npos && pos_k != str.npos)
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
    MMAConfig(16, 8, 32, DataType::kFloat8_e4m3, false, false),
    MMAConfig(16, 8, 64, DataType::kFloat8_e4m3, false, true),
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
      TVM_FFI_ICHECK(dtype_a == dtype_b) << ab_not_match_err_str;
      break;
    case DataType::kInt4:
    case DataType::kUInt4:
      TVM_FFI_ICHECK(dtype_b == DataType::kInt4 || dtype_b == DataType::kUInt4)
          << ab_not_match_err_str;
      break;
    case DataType::kInt8:
    case DataType::kUInt8:
      TVM_FFI_ICHECK(dtype_b == DataType::kInt8 || dtype_b == DataType::kUInt8)
          << ab_not_match_err_str;
      break;
    case DataType::kFloat8_e4m3:
    case DataType::kFloat8_e5m2:
      TVM_FFI_ICHECK(dtype_b == DataType::kFloat8_e4m3 || dtype_b == DataType::kFloat8_e5m2)
          << ab_not_match_err_str;
      break;
    default:
      TVM_FFI_ICHECK(false) << "Invalid multiplicand data types: " << DTypeToString(dtype_a)
                            << DTypeToString(dtype_b);
  }
  // check a,b and c
  switch (dtype_a) {
    case DataType::kBit1:
    case DataType::kInt4:
    case DataType::kUInt4:
    case DataType::kInt8:
    case DataType::kUInt8:
      TVM_FFI_ICHECK(dtype_c == DataType::kInt32)
          << "For multiplicand data type " << DTypeToString(dtype_a) << DTypeToString(dtype_b)
          << ", accumulator data type should be s32.";
      break;
    case DataType::kFloat16:
      TVM_FFI_ICHECK(dtype_c == DataType::kFloat16 || dtype_c == DataType::kFloat32)
          << "For multiplicand data type f16, accumulator data type should be f16/f32.";
      break;
    case DataType::kBFloat16:
    case DataType::kTensorFloat32:
      TVM_FFI_ICHECK(dtype_c == DataType::kFloat32)
          << "For multiplicand data type bf16/tf32, accumulator data type can only be f32.";
      break;
    case DataType::kFloat64:
      TVM_FFI_ICHECK(dtype_c == DataType::kFloat64)
          << "For multiplicand data type f64, accumulator data type can only be f64.";
      break;
    case DataType::kFloat8_e4m3:
    case DataType::kFloat8_e5m2:
      TVM_FFI_ICHECK(dtype_c == DataType::kFloat32)
          << "For multiplicand data type e4m3/e5m2, accumulator data type can only be f32.";
      break;
    default:
      TVM_FFI_ICHECK(false) << "Invalid multiplicand/accumulator data types: "
                            << DTypeToString(dtype_a) << DTypeToString(dtype_b)
                            << DTypeToString(dtype_c) << ".";
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
  TVM_FFI_ICHECK(bit_op == "xor" || bit_op == "and" || bit_op == "")
      << "Unrecognized 1-bit operation " << bit_op << " , can only be xor/and.";
  bool use_bit_op = !bit_op.empty();
  if (use_bit_op) {
    TVM_FFI_ICHECK(dtype_a == DataType::kBit1)
        << "Bit operator is only compatible with 1-bit multiplicand.";
  }
  CheckMMADTypeCompatible(dtype_a, dtype_b, dtype_c);
  if (saturate) {
    TVM_FFI_ICHECK(dtype_a == DataType::kInt4 || dtype_a == DataType::kUInt4 ||
                   dtype_a == DataType::kInt8 || dtype_a == DataType::kUInt8)
        << "Output saturation only applicable to multiplicand type s4/u4/s8/u8.";
  }

  if (!(m == 8 && n == 8 && k == 4 && dtype_a == ptx::DataType::kFloat16)) {
    // Only MMA on m8n8k4 for fp16 supports customized layouts.
    TVM_FFI_ICHECK(layout_a == LayoutType::kRowMajor && layout_b == LayoutType::kColumnMajor)
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
  TVM_FFI_ICHECK(match) << "Cannot find matched MMA configurations.";
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
    case DataType::kFloat8_e4m3:
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
  TVM_FFI_ICHECK(num == 1 || num == 2 || num == 4)
      << "ldmatrix only accept loading 1/2/4 matrices.";
  ptx::DataType data_type = ptx::DTypeFromString(type);
  TVM_FFI_ICHECK(data_type == ptx::DataType::kBit16)
      << "ldmatrix only accept matrix with type .b16.";
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
  TVM_FFI_ICHECK(bytes == "16" || bytes == "12" || bytes == "8" || bytes == "4" || bytes == "2" ||
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
  std::string caller_code = "{func_name}({barrier}, {phase});\n";

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
  std::string asm_code = R"(/* T.named_barrier_arrive() */ {
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
  std::string asm_code = R"(/* T.named_barrier_sync() */ {
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
    const std::string& tensormap, int cta_mask, std::vector<std::string> coords) {
  std::string func_code = R"(
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap* tensormap, int cta_mask_, {coord_arg_list}) {
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(tensormap);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_);
  if (cta_mask != 0) {
    // multicast
    __asm__ __volatile__(
      "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {arg_template_multicast}], [%2], %3;"
      :
      : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr), "h"(cta_mask),
        {coord_list}
      : "memory"
    );
  } else {
    // unicast
    __asm__ __volatile__(
      "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes"
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

std::string PrintCpAsyncBulkTensorSharedToGlobalAssembly(codegen::CodeGenCUDA* cg, int dim, const std::string& src,
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

std::string PrintCpAsyncBulkTensorWaitGroupAssembly(codegen::CodeGenCUDA* cg, const std::string& N, bool read) {
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
  CHECK(bits == 32 || bits == 64) << "Only support 32/64 bits for ptx_fetch_register.";

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
    LOG(FATAL) << "Only support uint32/float32 for wgmma_fence.";
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
  std::string asm_code = R"(/* T.wgmma_mma_async_ss() */ {
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
    CHECK(transA == false && transB == false)
        << "Matrices A and B are stored in row-major and column-major format respectively. The "
           "transpose operation is only supported for the wgmma.mma_async variants with .f16/ "
           ".bf16 types on matrices accessed from shared memory using matrix descriptors.";
  }
  CHECK(out_dtype == "float32")
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
  std::string asm_code = R"(/* T.wgmma_mma_async_rs() */ {
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
    CHECK(transA == false && transB == false)
        << "Matrices A and B are stored in row-major and column-major format respectively. The "
           "transpose operation is only supported for the wgmma.mma_async variants with .f16/ "
           ".bf16 types on matrices accessed from shared memory using matrix descriptors.";
  }
  CHECK(out_dtype == "float32")
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

  std::string func_name = "ptx_wgmma_arrive";
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

std::string PrintEncodeMatrixDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
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

  std::string func_name = "encode_matrix_descriptor";
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
  std::string asm_code = R"(/* T.barrier_cluster_arrive() */ {
  asm volatile("barrier.cluster.arrive{sem}{aligned};\n" : :);
}
)";
  Replacer replacer;
  replacer.register_rule("{sem}", sem.empty() ? "" : "." + sem);
  replacer.register_rule("{aligned}", aligned ? ".aligned" : "");
  return replacer.rewrite(asm_code);
}

std::string PrintBarrierClusterWaitAssembly(bool acquire, bool aligned) {
  std::string asm_code = R"(/* T.barrier_cluster_wait() */ {
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
  std::string func_name = "fence_mbarrier_init_release_cluster";

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
  ICHECK_EQ(vars.size(), 2 * num);
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

}  // namespace codegen
}  // namespace tvm
