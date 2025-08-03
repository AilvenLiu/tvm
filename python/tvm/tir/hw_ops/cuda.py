# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""HW level ops for CUDA, along with its codegen ruls"""
# pylint: disable=missing-function-docstring
import functools
import enum

import tvm.ffi
from tvm import DataType
from tvm.tir.op import cuda_func_call


#########################
# HEADER GENERATOR FOR CUDA CODEGEN
# The header generator is used to generate the header for the CUDA code.
# It's controlled by the predefine tags.
# The tags are used to identify the utility functions/classes necessary for the codegen.
#########################

TAGS = {
    "cuda",
    "cuda/barrier",
    "cooperative_groups",
    "fp16",
    "bf16",
    "fp8",
    "fp6",
    "fp4",
    "int8",
    "math_constants",
    "mma",
    "warp_shuffle",
    "cast_smem_ptr_to_int",
    "get_tmem_addr",
    "gmma_descriptor",
    "smem_descriptor",
    "instr_descriptor",
    "instr_descriptor_block_scaled",
    "get_time_stamp",
    "nvshmem",
}


@tvm.register_func("tir.hw_ops.cuda.header_generator")
def header_generator(tags):
    """Generate the header for the CUDA code."""
    for tag in tags:
        if tag not in TAGS:
            raise ValueError(f"Invalid tag: {tag}")

    header = ""
    if "nvshmem" in tags:
        header += R"""
#include <nvshmem.h>
#include <nvshmemx.h>
"""

    if "cuda/barrier" in tags or "cooperative_groups" in tags:
        header += (
            R"""
#include <cuda/barrier>
#include <cooperative_groups.h>
"""
            + "\n"
        )

    header += """
#include <cuda.h>
"""

    if "fp16" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#endif // __CUDA_ARCH__ >= 530

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
"""

    if "bf16" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_bf16.h>
__device__ nv_bfloat16 max(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hgt(a, b) ? a : b;
}
__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hlt(a, b) ? a : b;
}
#endif // __CUDA_ARCH__ >= 800
// Pack two bfloat16 values.
static inline __device__ __host__ unsigned
__pack_nv_bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some bfp16 math functions are not supported in cuda_bfp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x, nv_bfloat16 y) {   \
  float tmp_x = __bfloat162float(x);                                      \
  float tmp_y = __bfloat162float(y);                                      \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2bfloat16(result);                                        \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x) {          \
  float tmp_x = __bfloat162float(x);                                     \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2bfloat16(result);                                       \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
"""

    if "fp8" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#include <cuda_fp8.h>
using fp8_e4_t = __nv_fp8_e4m3;
using fp8_e4x2_t = __nv_fp8x2_e4m3;
using fp8_e4x4_t = __nv_fp8x4_e4m3;
struct fp8_e4x8_t {
 fp8_e4_t data[8]; 
};
struct fp8_e4x16_t {
 fp8_e4_t data[16]; 
};
using fp8_e5_t = __nv_fp8_e5m2;
using fp8_e5x2_t = __nv_fp8x2_e5m2;
using fp8_e5x4_t = __nv_fp8x4_e5m2;
struct fp8_e5x8_t {
 fp8_e5_t data[8]; 
};
struct fp8_e5x16_t {
 fp8_e5_t data[16]; 
};
using fp8_e8_t = __nv_fp8_e8m0;
using fp8_e8x2_t = __nv_fp8x2_e8m0;
using fp8_e8x4_t = __nv_fp8x4_e8m0;
struct fp8_e8x8_t {
 fp8_e8_t data[8]; 
};
struct fp8_e8x16_t {
 fp8_e8_t data[16]; 
};
#endif // __CUDA_ARCH__ >= 890
"""

    if "fp6" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#include <cuda_fp6.h>
using fp6_e2_t = __nv_fp6_e2m3;
using fp6_e2x2_t = __nv_fp6x2_e2m3;
using fp6_e2x4_t = __nv_fp6x4_e2m3;
struct fp6_e2x8_t {
 fp6_e2_t data[8]; 
};
struct fp6_e2x16_t {
 fp6_e2_t data[16]; 
};
using fp6_e3_t = __nv_fp6_e3m2;
using fp6_e3x2_t = __nv_fp6x2_e3m2;
using fp6_e3x4_t = __nv_fp6x4_e3m2;
struct fp6_e3x8_t {
 fp6_e3_t data[8]; 
};
struct fp6_e3x16_t {
 fp6_e3_t data[16]; 
};
#endif // __CUDA_ARCH__ >= 1000
"""

    if "fp4" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_fp4.h>
using fp4_e2_t = __nv_fp4_e2m1;
using fp4_e2x2_t = __nv_fp4x2_e2m1;
using fp4_e2x4_t = __nv_fp4x4_e2m1;
struct fp4_e2x8_t {
 fp4_e2_t data[8]; 
};
struct fp4_e2x16_t {
 fp4_e2_t data[16]; 
};
#endif // __CUDA_ARCH__ >= 800
"""

    #########################################################
    # Vector type extensions
    #########################################################
    if "fp16" in tags or "bf16" in tags:
        header += R"""
#include <type_traits>
template <typename T, typename TVec2>
struct __align__(8) half4_bfloat164 {
  T x, y, z, w;
  __host__ __device__ half4_bfloat164() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
  __host__ __device__ half4_bfloat164(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
"""
        if "fp8" in tags:
            header += R"""
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e4m3& fp8x4) {
    if constexpr (std::is_same_v<T, __half>) {
      __nv_fp8x2_e4m3 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
    } else {
      __nv_fp8_storage_t elem0_raw = static_cast<__nv_fp8_storage_t>(fp8x4.__x & 0xFF);
      __nv_fp8_storage_t elem1_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 8) & 0xFF);
      __nv_fp8_storage_t elem2_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 16) & 0xFF);
      __nv_fp8_storage_t elem3_raw = static_cast<__nv_fp8_storage_t>((fp8x4.__x >> 24) & 0xFF);
      __nv_fp8_e4m3 elem0, elem1, elem2, elem3;
      elem0.__x = elem0_raw;
      elem1.__x = elem1_raw;
      elem2.__x = elem2_raw;
      elem3.__x = elem3_raw;
      x = T(elem0);
      y = T(elem1);
      z = T(elem2);
      w = T(elem3);
    }
  }
  __host__ __device__ explicit operator __nv_fp8x4_e4m3() const {
    __nv_fp8x4_e4m3 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e4m3 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e5m2& fp8x4) {
      __nv_fp8x2_e5m2 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e5m2() const {
    __nv_fp8x4_e5m2 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e5m2 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4_bfloat164(const __nv_fp8x4_e8m0& fp8x4) {
      __nv_fp8x2_e8m0 lo_part, hi_part;
      lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e8m0() const {
    __nv_fp8x4_e8m0 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __nv_fp8x2_e8m0 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
"""
        if "fp4" in tags:
            header += R"""
  __host__ __device__ explicit half4_bfloat164(const __nv_fp4x4_e2m1& fp4x4) {
    if constexpr (std::is_same_v<T, __half>) {
      __nv_fp4x2_storage_t lo_part = static_cast<__nv_fp4x2_storage_t>(fp4x4.__x & 0xFF);
      __nv_fp4x2_storage_t hi_part = static_cast<__nv_fp4x2_storage_t>((fp4x4.__x >> 8) & 0xFF);
      TVec2 lo_half2 = __half2(__nv_cvt_fp4x2_to_halfraw2(lo_part, __NV_E2M1));
      TVec2 hi_half2 = __half2(__nv_cvt_fp4x2_to_halfraw2(hi_part, __NV_E2M1));
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
    } else {
      __nv_fp4_e2m1 elem0, elem1, elem2, elem3;
      elem0.__x = static_cast<__nv_fp4_storage_t>(fp4x4.__x & 0xF);
      elem1.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 4) & 0xF);
      elem2.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 8) & 0xF);
      elem3.__x = static_cast<__nv_fp4_storage_t>((fp4x4.__x >> 12) & 0xF);
      x = T(elem0);
      y = T(elem1);
      z = T(elem2);
      w = T(elem3);
    }
  }
  __host__ __device__ explicit operator __nv_fp4x4_e2m1() const {
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    return __nv_fp4x4_e2m1(lo_half2, hi_half2);
  }
"""
        header += R"""
};
"""
    if "fp16" in tags:
        header += R"""
using half4 = half4_bfloat164<__half, __half2>;
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}
"""
    if "bf16" in tags:
        header += R"""
using nv_bfloat164 = half4_bfloat164<nv_bfloat16, nv_bfloat162>;
__host__ __device__ nv_bfloat164 make_nv_bfloat164(nv_bfloat16 x, nv_bfloat16 y, nv_bfloat16 z, nv_bfloat16 w) {
    return nv_bfloat164(x, y, z, w);
}
__host__ __device__ nv_bfloat162 make_nv_bfloat162(nv_bfloat16 x, nv_bfloat16 y) {
    return nv_bfloat162(x, y);
}
"""
        if "fp8" in tags:
            header += R"""
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e4m3& fp8x2) {
    __nv_fp8_e4m3 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e5m2& fp8x2) {
    __nv_fp8_e5m2 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp8x2_e8m0& fp8x2) {
    __nv_fp8_e8m0 elem0, elem1;
    elem0.__x = static_cast<__nv_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
    """
    if "fp8" in tags:
        header += R"""
__device__ __nv_fp8x2_e5m2 make___nv_fp8x2_e5m2(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y) {
    __nv_fp8x2_e5m2 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e5m2 make___nv_fp8x4_e5m2(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b, __nv_fp8_e5m2 c, __nv_fp8_e5m2 d) {
    __nv_fp8x4_e5m2 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e4m3 make___nv_fp8x2_e4m3(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y) {
    __nv_fp8x2_e4m3 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e4m3 make___nv_fp8x4_e4m3(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 c, __nv_fp8_e4m3 d) {
    __nv_fp8x4_e4m3 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e8m0 make___nv_fp8x2_e8m0(__nv_fp8_e8m0 x, __nv_fp8_e8m0 y) {
    __nv_fp8x2_e8m0 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e8m0 make___nv_fp8x4_e8m0(__nv_fp8_e8m0 a, __nv_fp8_e8m0 b, __nv_fp8_e8m0 c, __nv_fp8_e8m0 d) {
    __nv_fp8x4_e8m0 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
"""
    if "fp4" in tags:
        header += R"""
__host__ __device__ nv_bfloat162 cast_to_nv_bfloat162(const __nv_fp4x2_e2m1& fp4x2) {
    __nv_fp4_e2m1 elem0, elem1;
    elem0.__x = static_cast<__nv_fp4_storage_t>(fp4x2.__x & 0xFF);
    elem1.__x = static_cast<__nv_fp4_storage_t>((fp4x2.__x >> 8) & 0xFF);
    nv_bfloat16 x = nv_bfloat16(elem0);
    nv_bfloat16 y = nv_bfloat16(elem1);
    return nv_bfloat162(x, y);
}
"""

    if "int8" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>

#if defined(__CUDACC_RTC__)
#define __SM_61_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_61_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) __DEF_IF_HOST
__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) __DEF_IF_HOST

#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#undef __SM_61_INTRINSICS_DECL__

#endif // __CUDA_ARCH__ >= 610
"""
    if "math_constants" in tags:
        header += R"""
#include <math_constants.h>
"""
    if "mma" in tags:
        header += R"""
#include <mma.h>
"""

    if "warp_shuffle" in tags:
        header += R"""
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif
"""

    if "cast_smem_ptr_to_int" in tags:
        header += R"""
__forceinline__ __device__ unsigned int cast_smem_ptr_to_int(const void* const smem_ptr) {
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}
"""
    header += R"""
#include <cuda.h>
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef __CUDACC_RTC__
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #include <cstdint>
#endif
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
"""

    if "get_tmem_addr" in tags:
        header += R"""
__forceinline__ __device__ uint32_t get_tmem_addr(uint32_t idx, int row_offset, int col_offset) {
  int col_idx = idx & 0xFFFF;
  int row_idx = (idx >> 16) & 0xFFFF;
  col_idx += col_offset;
  row_idx += row_offset;
  col_idx = col_idx & 0xFFFF;
  row_idx = row_idx & 0xFFFF;

  uint32_t new_idx = (row_idx << 16) | col_idx;
  return new_idx;
}
"""

    if "get_time_stamp" in tags:
        header += R"""
__forceinline__ __device__ uint32_t tvm_builtin_get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}
"""

    if "gmma_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union GmmaDescriptor
{
  HOST_DEVICE constexpr
  GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
"""

    if "smem_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union SmemDescriptor
{
  uint64_t desc_ = 0;
  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;                     // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2;               // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, version_ : 2;       // 14 bits [0,14), 2 bits [14,16)
    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;     // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
    // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
    uint8_t : 5, layout_type_ : 3;                         // 6 bits unused, 3 bits [5,8)
  };
  // Seperate the field, as we may only update one part of desc
  struct {
    uint32_t lo;
    uint32_t hi;
  };

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
"""

    if "instr_descriptor" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union InstrDescriptor
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
             saturate_      : 1,  // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for S8
             c_format_      : 2,  // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
                            : 1,  //
             a_format_      : 3,  // bit [ 7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             b_format_      : 3,  // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
                            : 1,  //
             m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
                            : 1,  //
             max_shift_     : 2;  // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
  };

  // Decay to a uint32_t
  HOST_DEVICE constexpr explicit
  operator uint32_t() const noexcept { return desc_; }
};
"""

    if "instr_descriptor_block_scaled" in tags:
        header += R"""
#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union InstrDescriptorBlockScaled
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
                            : 1,  //
             b_sf_id_       : 2,  // bit [ 4, 6) : Matrix B Scale Factor ID
                            : 1,  //
             a_format_      : 3,  // bit [ 7, 9) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
             b_format_      : 3,  // bit [10,12) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
             scale_format_  : 1,  // bit [23,24) : 0=E4M3, 1=E8M0
             m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
             a_sf_id_       : 2,  // bit [29,31) : Matrix A Scale Factor ID
                            : 1;  //
  };

  // Decay to a uint32_t
  HOST_DEVICE constexpr
  operator uint32_t() const noexcept { return desc_; }
};
"""

    return header


#########################
# CODEGEN REGISTRY FOR CUDA HW OPS
#########################


CODEGEN_REGISTRY = {}


@tvm.ffi.register_func("tir.hw_ops.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda"):
    """register a codegen function for a given op
    The codegen function should return a cuda_func_call statement,
    and a list of tags that the codegen function needs.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(arg_list):
            res = func(*arg_list)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            else:
                return res, list()

        CODEGEN_REGISTRY["tir." + op] = wrapper
        return wrapper

    return decorator


from_string_func = tvm.ffi.get_global_func("tir.hw_ops.cuda.PTXDTypeFromString")
to_string_func = tvm.ffi.get_global_func("tir.hw_ops.cuda.PTXDTypeToString")


class PTXDataType(enum.Enum):
    """
    A Python equivalent of the provided C++ DataType enum class.

    Inherits from IntEnum so that members behave both as enum members
    and as integers, mirroring the C++ behavior.

    see also src/target/source/ptx.cc
    """

    INT4 = 0
    UINT4 = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    INT64 = 8
    UINT64 = 9
    FLOAT4_E2M1FN = 10
    FLOAT6_E2M3FN = 11
    FLOAT6_E3M2FN = 12
    FLOAT8_E4M3FN = 13
    FLOAT8_E4M3FNUZ = 14
    FLOAT8_E5M2 = 15
    FLOAT8_E8M0FNU = 16
    FLOAT16 = 17
    BFLOAT16 = 18
    FLOAT16X2 = 19
    FLOAT32 = 20
    TENSOR_FLOAT32 = 21
    FLOAT64 = 22
    BIT1 = 23
    BIT8 = 24
    BIT16 = 25
    BIT32 = 26
    BIT64 = 27

    @classmethod
    def from_string(cls, s_type: str) -> "PTXDataType":
        return PTXDataType(from_string_func(s_type))

    def to_string(self) -> str:
        return to_string_func(self.value)


########################################################
# PTX wgmma
########################################################


@register_codegen("ptx_wgmma_encode_matrix_descriptor")
def codegen_ptx_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    valid_swizzle_modes = {0, 1, 2, 3}
    if swizzle not in valid_swizzle_modes:
        raise ValueError(
            f"Invalid swizzle mode. Expected a value in {valid_swizzle_modes}, but got {swizzle}"
        )

    func_name = "ptx_wgmma_encode_matrix_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {{
  GmmaDescriptor _desc;

  switch (swizzle) {{
    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle
    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle
    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle
  }}

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);
  
  constexpr uint8_t base_offset = 0;
  _desc.bitfield.base_offset_ = base_offset;

  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}}"""
    return cuda_func_call(func_name, desc, addr, ldo, sdo, swizzle, source_code=source_code), [
        "gmma_descriptor"
    ]


@register_codegen("ptx_wgmma_noop_barrier")
def codegen_ptx_wgmma_noop_barrier(reg):
    dtype = str(reg.dtype)
    dtype_enum = PTXDataType.from_string(dtype)
    if dtype_enum == PTXDataType.UINT32:
        format_str, dtype_str = "r", "uint32_t"
    elif dtype_enum == PTXDataType.FLOAT32:
        format_str, dtype_str = "f", "float"
    else:
        raise ValueError(f"Only support uint32/float32 for wgmma_fence, but got {dtype}.")

    # 2. The function name is generated dynamically based on the data type.
    func_name = f"ptx_wgmma_fence_{dtype_str}"

    # 3. Populate the C++ template. The empty asm string "" with memory clobber
    #    and an input/output operand acts as a compiler optimization barrier.
    source_code = f"""
__forceinline__ __device__ void {func_name}({dtype_str} reg) {{
  asm volatile("" : "+{format_str}"(reg)::"memory");
}}"""
    return cuda_func_call(func_name, reg, source_code=source_code)


@register_codegen("ptx_wgmma_fence")
def codegen_ptx_wgmma_fence():
    func_name = "ptx_wgmma_fence"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_commit_group")
def codegen_ptx_wgmma_commit_group():
    func_name = "ptx_wgmma_commit_group"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_wait_group")
def codegen_ptx_wgmma_wait_group(n):
    n = int(n)
    func_name = f"ptx_wgmma_wait_group_{n}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"({n}) : "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_mma_async_ss")
def codegen_ptx_wgmma_mma_async_ss(
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descA, descB, *accums
):
    M = int(M)
    N = int(N)
    K = int(K)
    in_dtype = str(in_dtype)[1:-1]
    out_dtype = str(out_dtype)[1:-1]
    transA = bool(transA)
    transB = bool(transB)
    scaleA = float(scaleA)
    scaleB = float(scaleB)

    expected_accm_cnt = M * N // 128
    if len(accums) != expected_accm_cnt:
        raise ValueError(
            "The number of arguments is incorrect. Expected "
            f"{12 + expected_accm_cnt} total args (meaning {expected_accm_cnt} accumulator args), "
            f"but got {len(accums)}."
        )

    if out_dtype != "float32":
        raise ValueError("Codegen only supports float32 as output dtype for WGMMA.")

    allow_transpose = in_dtype in {"float16", "bfloat16"}
    if not allow_transpose and (transA or transB):
        raise ValueError("Transpose is only supported for .f16/.bf16 types in WGMMA.")

    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)

    num_accums = len(accums)
    descA_idx = num_accums
    descB_idx = num_accums + 1
    scaleD_idx = num_accums + 2
    scaleA_idx = num_accums + 3
    scaleB_idx = num_accums + 4
    transA_idx = num_accums + 5
    transB_idx = num_accums + 6

    if allow_transpose:
        transpose_r_code = f", %{transA_idx}, %{transB_idx}"
        transpose_constraints = f', "n"({1 if transA else 0}), "n"({1 if transB else 0})'
    else:
        transpose_r_code = ""
        transpose_constraints = ""

    descA_idx, descB_idx, scaleD_idx, scaleA_idx, scaleB_idx = (num_accums + i for i in range(5))
    transA_idx, transB_idx = (num_accums + 5, num_accums + 6)

    accum_params = ", ".join([f"float& p_acc{i}" for i in range(num_accums)])
    accum_r_list = ", ".join([f"%{i}" for i in range(num_accums)])
    accum_constraints = ", ".join([f'"+f"(p_acc{i})' for i in range(num_accums)])

    func_name = f"ptx_wgmma_mma_async_ss_{M}x{N}x{K}_{in_dtype.replace('.', '')}_{out_dtype.replace('.', '')}_{int(scaleA)}_{int(scaleB)}_{int(transA)}_{int(transB)}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({accum_params}, uint64_t p_descA, uint64_t p_descB, int p_scaleD) {{
    /* T.ptx_wgmma_mma_async_ss() */
    asm volatile(
      "{{    \\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype.to_string()}{itype.to_string()}{itype.to_string()} "
      "{{{accum_r_list}}},"
      "%{descA_idx}, %{descB_idx},"
      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"
      "}}\\n"
      : {accum_constraints}
      : "l"(p_descA), "l"(p_descB), "r"(p_scaleD), "n"({int(scaleA)}), "n"({int(scaleB)}){transpose_constraints}
    );
}}"""
    return cuda_func_call(func_name, *accums, descA, descB, scaleD, source_code=source_code)


@register_codegen("ptx_wgmma_mma_async_rs")
def codegen_ptx_wgmma_mma_async_rs(
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descB, *reg_list
):
    M = int(M)
    N = int(N)
    K = int(K)
    transA = bool(transA)
    transB = bool(transB)
    scaleA = float(scaleA)
    scaleB = float(scaleB)
    in_dtype = str(in_dtype)[1:-1]
    out_dtype = str(out_dtype)[1:-1]

    if out_dtype != "float32":
        raise ValueError("This generator only supports float32 as the output dtype for WGMMA.")

    in_dtype_bits = tvm.runtime.DataType(in_dtype).bits
    if in_dtype_bits is None:
        raise ValueError(f"Bit width not defined for input dtype: {in_dtype}")

    expected_A_cnt = M * K // 128 // (32 // in_dtype_bits)
    expected_accm_cnt = M * N // 128

    if len(reg_list) != expected_A_cnt + expected_accm_cnt:
        raise ValueError(
            f"Incorrect number of A registers. Expected {expected_A_cnt}, got {len(reg_list)}"
        )
    A_regs = reg_list[:expected_A_cnt]
    accums = reg_list[expected_A_cnt:]

    allow_transpose = in_dtype in {"float16", "bfloat16"}
    if not allow_transpose and (transA or transB):
        raise ValueError("Transpose is only supported for .f16/.bf16 types in WGMMA.")

    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)
    allow_transpose = in_dtype in {"float16", "bfloat16"}

    accum_params = ", ".join([f"float& p_acc{i}" for i in range(expected_accm_cnt)])
    A_reg_params = ", ".join([f"uint32_t& p_A{i}" for i in range(expected_A_cnt)])

    accum_r_list = ", ".join([f"%{i}" for i in range(expected_accm_cnt)])
    A_reg_r_list = ", ".join([f"%{expected_accm_cnt + i}" for i in range(expected_A_cnt)])

    base_idx = expected_accm_cnt + expected_A_cnt
    descB_idx, scaleD_idx, scaleA_idx, scaleB_idx, transB_idx = (base_idx + i for i in range(5))

    accum_constraints = ", ".join([f'"+f"(p_acc{i})' for i in range(expected_accm_cnt)])
    A_reg_constraints = ", ".join([f'"r"(p_A{i})' for i in range(expected_A_cnt)])

    if allow_transpose:
        transpose_r_code = f", %{transB_idx}"
        transpose_constraints = f', "n"({1 if transB else 0})'
    else:
        transpose_r_code, transpose_constraints = "", ""

    func_name = f"ptx_wgmma_mma_async_rs_{M}x{N}x{K}_{in_dtype.replace('.', '')}_{out_dtype.replace('.', '')}_{int(scaleA)}_{int(scaleB)}_{int(transA)}_{int(transB)}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({accum_params}, {A_reg_params}, uint64_t p_descB, int p_scaleD) {{
    /* T.ptx_wgmma_mma_async_rs() */
    asm volatile(
      "{{    \\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype.to_string()}{itype.to_string()}{itype.to_string()} "
      "{{{accum_r_list}}},"
      "{{{A_reg_r_list}}}, %{descB_idx},"
      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"
      "}}\\n"
      : {accum_constraints}
      : {A_reg_constraints}, "l"(p_descB), "r"(p_scaleD), "n"({int(scaleA)}), "n"({int(scaleB)}){transpose_constraints}
    );
}}"""
    return cuda_func_call(func_name, *accums, *A_regs, descB, scaleD, source_code=source_code)


########################################################
# PTX tcgen05
########################################################


@register_codegen("ptx_tcgen05_alloc")
def codegen_ptx_tcgen05_alloc(dst_shared_ptr, n_cols, n_cta_group):
    n_cols = int(n_cols)
    n_cta_group = int(n_cta_group)
    is_power_of_two = (n_cols > 0) and ((n_cols & (n_cols - 1)) == 0)
    if not (32 <= n_cols <= 512 and n_cols % 32 == 0 and is_power_of_two):
        raise ValueError(  # pylint: disable=raise-missing-from
            "The number of columns to allocate in Tensor Memory is invalid, "
            f"expect a value within range [32, 512] and be a multiple of 32 "
            f"and a power of 2, got {n_cols}"
        )

    if n_cta_group not in [1, 2]:
        raise ValueError(
            "The number of cta_group involved in allocating Tensor Memory is incorrect, "
            f"expected 1 or 2, got {n_cta_group}"
        )

    func_name = f"tvm_builtin_ptx_tcgen05_alloc_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, int nCols) {{
  unsigned int smem_addr = __cvta_generic_to_shared(dst);
    __asm__ __volatile__(
      "tcgen05.alloc.cta_group::{n_cta_group}.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(smem_addr), "r"(nCols)
      : "memory"
    );
}}
"""
    return cuda_func_call(func_name, dst_shared_ptr, n_cols, source_code=source_code)


@register_codegen("ptx_tcgen05_dealloc")
def codegen_ptx_tcgen05_dealloc(taddr, n_cols, n_cta_group):
    n_cols = int(n_cols)
    n_cta_group = int(n_cta_group)
    is_power_of_two = (n_cols > 0) and ((n_cols & (n_cols - 1)) == 0)
    if not (32 <= n_cols <= 512 and n_cols % 32 == 0 and is_power_of_two):
        raise ValueError(
            "The number of columns to deallocate in Tensor Memory is invalid, expect a value within"
            f"range [32, 512] and be a multiple of 32 and a power of 2, got {n_cols}"
        )

    if n_cta_group not in [1, 2]:
        raise ValueError(
            "The number of cta_group involved in deallocating Tensor Memory is incorrect, expected 1"
            f"or 2, got {n_cta_group}"
        )

    func_name = f"tvm_builtin_ptx_tcgen05_dealloc_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t taddr, int nCols) {{
    __asm__ __volatile__(
      "tcgen05.dealloc.cta_group::{n_cta_group}.sync.aligned.b32 %0, %1;"
      :: "r"(taddr), "r"(nCols)
      : "memory"
    );
}}
"""
    return cuda_func_call(func_name, taddr, n_cols, source_code=source_code)


@register_codegen("ptx_tcgen05_relinquish_alloc_permit")
def codegen_ptx_tcgen05_relinquish_alloc_permit(n_cta_group):
    n_cta_group = int(n_cta_group)
    if n_cta_group not in [1, 2]:
        raise ValueError(
            "The number of cta_group involved in relinquishing alloc permit is incorrect, expected 1"
            f"or 2, got {n_cta_group}"
        )

    func_name = f"tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __asm__ __volatile__(
        "tcgen05.relinquish_alloc_permit.cta_group::{n_cta_group}.sync.aligned;"
        ::: "memory"
    );
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_fence_before_thread_sync")
def codegen_ptx_tcgen05_fence_before_thread_sync():
    func_name = "tvm_builtin_ptx_tcgen05_fence_before_thread_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("tcgen05.fence::before_thread_sync;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_fence_after_thread_sync")
def codegen_ptx_tcgen05_fence_after_thread_sync():
    func_name = "tvm_builtin_ptx_tcgen05_fence_after_thread_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("tcgen05.fence::after_thread_sync;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_ld")
def codegen_ptx_tcgen05_ld(src_addr, row_offset, col_offset, shape, num, pack, *regs):
    shape = str(shape)[1:-1]
    num = int(num)
    pack = bool(pack)
    is_power_of_two = (num > 0) and ((num & (num - 1)) == 0)
    if not (1 <= num <= 128 and is_power_of_two):
        raise ValueError(
            "The repeat factor of ptx_tcgen05_ld is invalid, expect a value within range [1, 128] "
            f"and be a power of 2, got {num}"
        )

    if shape in ["16x32bx2", "16x64b", "32x32b"]:
        expected_n_regs = num
    elif shape == "16x128b":
        if num > 64:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_ld for shape 16x128b is invalid, "
                f"expect a value within range [1, 64], got {num}"
            )
        expected_n_regs = 2 * num
    elif shape == "16x256b":
        if num > 32:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_ld for shape 16x256b is invalid, "
                f"expect a value within range [1, 32], got {num}"
            )
        expected_n_regs = 4 * num
    else:
        raise ValueError(
            "The input shape of ptx_tcgen05_ld is invalid, expect one of [16x32bx2, 16x64b, "
            f"32x32b, 16x128b, 16x256b], got {shape}"
        )

    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_ld is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )

    reg_args = ", ".join([f"void* reg{i}" for i in range(len(regs))])
    regs_placeholder = ", ".join([f"%{i}" for i in range(len(regs))])
    src_placeholder = str(len(regs))

    imm_arg = ""
    if shape == "16x32bx2":
        imm = 2 * num if pack else num
        imm_arg = f", {imm}"

    reg_operands = ", ".join([f'"=r"(*(uint32_t*)reg{i})' for i in range(len(regs))])
    pack_str = ".pack::16b" if pack else ""

    func_name = "tvm_builtin_ptx_tcgen05_ld_" + shape + "_x" + str(num) + ("_pack" if pack else "")
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t src_addr, uint32_t row_offset, uint32_t col_offset, {reg_args}) {{
    asm volatile(
        "tcgen05.ld.sync.aligned.{shape}.x{num}{pack_str}.b32 "
        "{{{regs_placeholder}}}, "
        "[%{src_placeholder}]{imm_arg};\\n"
        :  {reg_operands}
        :  "r"(get_tmem_addr(src_addr, row_offset, col_offset))
    );
}}
"""
    regs = [tvm.tir.address_of(reg) for reg in regs]

    return cuda_func_call(
        func_name,
        src_addr,
        row_offset,
        col_offset,
        *regs,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_st")
def codegen_ptx_tcgen05_st(dst_addr, row_offset, col_offset, shape, num, unpack, *regs):
    shape = str(shape)[1:-1]
    num = int(num)
    unpack = bool(unpack)
    is_power_of_two = (num > 0) and ((num & (num - 1)) == 0)
    if not (1 <= num <= 128 and is_power_of_two):
        raise ValueError(
            "The repeat factor of ptx_tcgen05_st is invalid, expect a value within range [1, 128] "
            f"and be a power of 2, got {num}"
        )

    if shape in ["16x32bx2", "16x64b", "32x32b"]:
        expected_n_regs = num
    elif shape == "16x128b":
        if num > 64:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_st for shape 16x128b is invalid, "
                f"expect a value within range [1, 64], got {num}"
            )
        expected_n_regs = 2 * num
    elif shape == "16x256b":
        if num > 32:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_st for shape 16x256b is invalid, "
                f"expect a value within range [1, 32], got {num}"
            )
        expected_n_regs = 4 * num
    else:
        raise ValueError(
            "The input shape of ptx_tcgen05_st is invalid, expect one of [16x32bx2, 16x64b, "
            f"32x32b, 16x128b, 16x256b], got {shape}"
        )

    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_st is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )

    reg_args = ", ".join([f"void* reg{i}" for i in range(len(regs))])
    regs_placeholder = ", ".join([f"%{i + 1}" for i in range(len(regs))])

    imm_arg = ""
    if shape == "16x32bx2":
        imm = 2 * num if unpack else num
        imm_arg = f", {imm}"

    reg_operands = ", ".join([f'"r"(*(uint32_t*)reg{i})' for i in range(len(regs))])
    unpack_str = ".unpack::16b" if unpack else ""

    func_name = (
        "tvm_builtin_ptx_tcgen05_st_" + shape + "_x" + str(num) + ("_unpack" if unpack else "")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t dst_addr, uint32_t row_offset, uint32_t col_offset, {reg_args}) {{
    asm volatile(
        "tcgen05.st.sync.aligned.{shape}.x{num}{unpack_str}.b32 "
        "[%0]{imm_arg}, "
        "{{{regs_placeholder}}};\\n"
        :
        :  "r"(get_tmem_addr(dst_addr, row_offset, col_offset)), {reg_operands}
    );
}}
"""
    regs = [tvm.tir.address_of(reg) for reg in regs]
    return cuda_func_call(
        func_name,
        dst_addr,
        row_offset,
        col_offset,
        *regs,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_wait_ld")
def codegen_ptx_tcgen05_wait_ld():
    func_name = "tvm_builtin_ptx_tcgen05_wait_ld"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("tcgen05.wait::ld.sync.aligned;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_wait_st")
def codegen_ptx_tcgen05_wait_st():
    func_name = "tvm_builtin_ptx_tcgen05_wait_st"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("tcgen05.wait::st.sync.aligned;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_encode_matrix_descriptor")
def codegen_ptx_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    valid_swizzle_modes = [0, 1, 2, 3, 4]
    swizzle = int(swizzle)
    if swizzle not in valid_swizzle_modes:
        raise ValueError(
            f"Invalid swizzle mode. Expected a value in {valid_swizzle_modes}, but got {swizzle}"
        )

    func_name = "tvm_builtin_ptx_tcgen05_encode_matrix_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {{
  SmemDescriptor _desc;

  _desc.version_ = 1;
  _desc.lbo_mode_ = 0;

  switch (swizzle) {{
    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle
    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle
    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle
    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle
  }}

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.base_offset_ = base_offset;

  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}}
"""
    return cuda_func_call(func_name, desc, addr, ldo, sdo, swizzle, source_code=source_code), [
        "smem_descriptor"
    ]


def _get_tcgen05_mma_kind(
    d_dtype: str, a_dtype: str, b_dtype: str, sfa_dtype: str = "", sfb_dtype: str = ""
) -> str:
    kind = ""

    dtype = PTXDataType.from_string(d_dtype)
    atype = PTXDataType.from_string(a_dtype)
    btype = PTXDataType.from_string(b_dtype)
    kind = ""

    if (
        atype == PTXDataType.FLOAT16
        and btype == PTXDataType.FLOAT16
        and dtype == PTXDataType.FLOAT16
    ):
        kind = "f16"
    elif (
        (atype in {PTXDataType.BFLOAT16, PTXDataType.FLOAT16})
        and (btype in {PTXDataType.BFLOAT16, PTXDataType.FLOAT16})
        and dtype == PTXDataType.FLOAT32
    ):
        kind = "f16"
    elif (
        atype == PTXDataType.TENSOR_FLOAT32
        and btype == PTXDataType.TENSOR_FLOAT32
        and dtype == PTXDataType.FLOAT32
    ):
        kind = "tf32"
    elif (
        atype == PTXDataType.FLOAT4_E2M1FN
        and btype == PTXDataType.FLOAT4_E2M1FN
        and sfa_dtype
        and sfb_dtype
        and dtype == PTXDataType.FLOAT32
    ):
        sfa_dtype_enum = PTXDataType.from_string(sfa_dtype)
        sfb_dtype_enum = PTXDataType.from_string(sfb_dtype)
        if (
            sfa_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
            and sfb_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
        ):
            kind = "mxf4"
        elif (sfa_dtype_enum in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}) and (
            sfb_dtype_enum in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}
        ):
            kind = "mxf4nvf4"
    elif (
        atype
        in {
            PTXDataType.FLOAT8_E4M3FN,
            PTXDataType.FLOAT8_E4M3FNUZ,
            PTXDataType.FLOAT8_E5M2,
            PTXDataType.FLOAT6_E2M3FN,
            PTXDataType.FLOAT6_E3M2FN,
            PTXDataType.FLOAT4_E2M1FN,
        }
    ) and (
        btype
        in {
            PTXDataType.FLOAT8_E4M3FN,
            PTXDataType.FLOAT8_E4M3FNUZ,
            PTXDataType.FLOAT8_E5M2,
            PTXDataType.FLOAT6_E2M3FN,
            PTXDataType.FLOAT6_E3M2FN,
            PTXDataType.FLOAT4_E2M1FN,
        }
    ):
        if not sfa_dtype and not sfb_dtype:
            if dtype in {PTXDataType.FLOAT32, PTXDataType.FLOAT16}:
                kind = "f8f6f4"
        elif sfa_dtype and sfb_dtype:
            sfa_dtype_enum = PTXDataType.from_string(sfa_dtype)
            sfb_dtype_enum = PTXDataType.from_string(sfb_dtype)
            if (
                sfa_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
                and sfb_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
                and dtype == PTXDataType.FLOAT32
            ):
                kind = "mxf8f6f4"
    elif (
        (atype in {PTXDataType.INT8, PTXDataType.UINT8})
        and (btype in {PTXDataType.INT8, PTXDataType.UINT8})
        and dtype == PTXDataType.INT32
    ):
        kind = "i8"

    if not kind:
        raise ValueError(
            f"Invalid multiplicand data types for Tcgen05 MMA, check failed for d: {d_dtype}, "
            f"a: {a_dtype}, b: {b_dtype}, scale_a: {sfa_dtype}, scale_b: {sfb_dtype}"
        )
    return kind


def _check_tcgen05_mma_matrix_shape(
    kind: str, cta_group: int, m: int, n: int, k: int, is_sparse: bool
) -> bool:
    err = (
        f"Invalid matrix shape for Tcgen05 MMA, check failed for kind: {kind}, "
        f"is_sparse: {is_sparse}, cta_group: {cta_group}, M: {m}, N: {n}, K: {k}"
    )

    if kind in ["f16", "tf32", "f8f6f4"]:
        if cta_group == 1:
            if m == 64:
                if not (8 <= n <= 256 and n % 8 == 0):
                    raise ValueError(err)
            elif m == 128:
                if not (16 <= n <= 256 and n % 16 == 0):
                    raise ValueError(err)
            else:
                raise ValueError(err)
        elif cta_group == 2:
            if not (m in [128, 256]):
                raise ValueError(err)
            if not (32 <= n <= 256 and n % 32 == 0):
                raise ValueError(err)
    elif kind == "i8":
        if cta_group == 1:
            if not (m in [64, 128]):
                raise ValueError(err)
            is_n_valid = (n == 8) or (n == 24) or (16 <= n <= 256 and n % 16 == 0)
            if not is_n_valid:
                raise ValueError(err)
        elif cta_group == 2:
            if not (m in [128, 256]):
                raise ValueError(err)
            if not (32 <= n <= 256 and n % 32 == 0):
                raise ValueError(err)
    elif kind in ["mxf8f6f4", "mxf4", "mxf4nvf4"]:
        if cta_group == 1:
            if not (m == 128):
                raise ValueError(err)
            if not (8 <= n <= 256 and n % 8 == 0):
                raise ValueError(err)
        elif cta_group == 2:
            if is_sparse:
                if not (m == 256):
                    raise ValueError(err)
            else:  # dense
                if not (m in [128, 256]):
                    raise ValueError(err)
            if not (16 <= n <= 256 and n % 16 == 0):
                raise ValueError(err)
    else:
        raise ValueError(err)

    if kind == "f16":
        if not (k == (32 if is_sparse else 16)):
            raise ValueError(err)
    elif kind == "tf32":
        if not (k == (16 if is_sparse else 8)):
            raise ValueError(err)
    elif kind in ["f8f6f4", "i8", "mxf8f6f4"]:
        if not (k == (64 if is_sparse else 32)):
            raise ValueError(err)
    elif kind in ["mxf4", "mxf4nvf4"]:
        if not (k == (128 if is_sparse else 64)):
            raise ValueError(err)
    else:
        raise ValueError(err)

    return True


@register_codegen("ptx_tcgen05_encode_instr_descriptor")
def codegen_ptx_tcgen05_encode_instr_descriptor(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    sat_d,
    is_sparse,
):
    a_dtype = str(a_dtype)[1:-1]
    b_dtype = str(b_dtype)[1:-1]
    d_dtype = str(d_dtype)[1:-1]
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = int(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    sat_d = bool(sat_d)
    is_sparse = bool(is_sparse)

    if n_cta_group not in [1, 2]:
        raise ValueError(
            f"The number of cta_group involved is incorrect, expected 1 or 2, got {n_cta_group}"
        )

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype)
    if kind not in ["f16", "tf32", "f8f6f4", "i8"]:
        raise ValueError(
            f"Check failed for Data Type Kind. d_dtype: {d_dtype}, a_dtype: {a_dtype}, b_dtype: {b_dtype}"
        )

    if not _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse):
        raise ValueError(f"Invalid matrix shape ({M}, {N}, {K}) for kind '{kind}'")

    format_map = {
        PTXDataType.FLOAT16: 0,
        PTXDataType.BFLOAT16: 1,
        PTXDataType.TENSOR_FLOAT32: 2,
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
        PTXDataType.UINT8: 0,
        PTXDataType.INT8: 1,
        PTXDataType.FLOAT32: 1,
        PTXDataType.INT32: 2,
    }
    dtype = PTXDataType.from_string(d_dtype)
    atype = PTXDataType.from_string(a_dtype)
    btype = PTXDataType.from_string(b_dtype)

    d_format = format_map[dtype]
    a_format = format_map[atype]
    b_format = format_map[btype]

    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.INT8,
        PTXDataType.UINT8,
        PTXDataType.FLOAT16,
        PTXDataType.BFLOAT16,
        PTXDataType.TENSOR_FLOAT32,
    }
    if trans_a and atype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")
    if (neg_a or neg_b) and kind not in ["f16", "tf32", "f8f6f4"]:
        raise ValueError(f"Invalid kind for negate: {kind}")
    if sat_d and kind != "i8":
        raise ValueError(f"Invalid kind for saturate: {kind}")

    func_name = "ptx_tcgen05_encode_instr_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int d_format,
                                            int a_format, int b_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool sat_d, bool is_sparse) {{
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
}}
"""
    return cuda_func_call(
        func_name,
        desc,
        M,
        N,
        d_format,
        a_format,
        b_format,
        trans_a,
        trans_b,
        neg_a,
        neg_b,
        sat_d,
        is_sparse,
        source_code=source_code,
    ), ["instr_descriptor"]


@register_codegen("ptx_tcgen05_encode_instr_descriptor_block_scaled")
def codegen_ptx_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    sfa_tmem_addr,
    sfb_tmem_addr,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    is_sparse,
):
    a_dtype = str(a_dtype)[1:-1]
    b_dtype = str(b_dtype)[1:-1]
    d_dtype = str(d_dtype)[1:-1]
    sfa_dtype = str(sfa_dtype)[1:-1]
    sfb_dtype = str(sfb_dtype)[1:-1]
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = int(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    is_sparse = bool(is_sparse)

    if n_cta_group not in [1, 2]:
        raise ValueError(
            f"The number of cta_group involved is incorrect, expected 1 or 2, got {n_cta_group}"
        )

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype}, a:{a_dtype}, b:{b_dtype}, sfa:{sfa_dtype}, sfb:{sfb_dtype}"
        )

    _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse)

    # Phase 2: Map data types to integer format codes
    format_map = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
    }
    format_map_sf = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E8M0FNU: 1,
    }

    atype_enum = PTXDataType.from_string(a_dtype)
    btype_enum = PTXDataType.from_string(b_dtype)
    stype_enum = PTXDataType.from_string(sfa_dtype)

    if kind == "mxf8f6f4":
        a_format = format_map[atype_enum]
        b_format = format_map[btype_enum]
    else:  # mxf4 and mxf4nvf4
        a_format = 1  # Corresponds to E5M2 in the map, a specific hardware encoding choice
        b_format = 1

    s_format = format_map_sf[stype_enum]

    # Phase 3: Detailed conditional validation for transpose
    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
    }
    if trans_a and atype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")

    func_name = "ptx_tcgen05_encode_instr_descriptor_block_scaled"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int a_format,
                                            int b_format, int s_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool is_sparse,
                                            uint32_t sfa_tmem_addr, uint32_t sfb_tmem_addr) {{
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
}}
"""
    return cuda_func_call(
        func_name,
        desc,
        M,
        N,
        a_format,
        b_format,
        s_format,
        trans_a,
        trans_b,
        neg_a,
        neg_b,
        is_sparse,
        sfa_tmem_addr,
        sfb_tmem_addr,
        source_code=source_code,
    ), ["instr_descriptor_block_scaled"]


def _tcgen05_mma_common(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
    sparse=False,
    sp_tmem_addr=None,
):
    d_dtype = str(d_dtype)[1:-1]
    a_dtype = str(a_dtype)[1:-1]
    b_dtype = str(b_dtype)[1:-1]
    use_a_tmem = bool(use_a_tmem)
    cta_group = int(cta_group)
    enable_input_d = bool(enable_input_d)
    scale_input_d = int(scale_input_d)

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")
    if not (0 <= scale_input_d <= 15):
        raise ValueError(
            f"scale_input_d is incorrect, expected a value within [0, 15], got {scale_input_d}"
        )

    expected_vec_size = 8 if cta_group == 2 else 4
    if len(disable_output_lane) != expected_vec_size:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_mma is incorrect, expected "
            f"{11 + expected_vec_size} total args (meaning {expected_vec_size} lane mask args), "
            f"but got {len(disable_output_lane)}."
        )

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype)
    valid_kinds = {"f16", "tf32", "f8f6f4", "i8"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Got '{kind}', expected one of {valid_kinds}"
        )

    if scale_input_d > 0 and kind not in {"f16", "tf32"}:
        raise ValueError(f"scale_input_d is only valid for kind 'f16' or 'tf32', not '{kind}'")

    if sparse:
        p_operand_idx = 5
        sparse_instr_suffix = ".sp"
        i_sp_operand_str = "[%3], %4,"
        sp_tmem_addr_str = "uint32_t sp_tmem_addr, "
        mask_start_idx = 6
    else:
        p_operand_idx = 4
        sparse_instr_suffix = ""
        i_sp_operand_str = "%3,"
        sp_tmem_addr_str = ""
        mask_start_idx = 5

    mask_signature = ", ".join([f"uint32_t mask{i}" for i in range(len(disable_output_lane))])

    a_operand_str = "[%1]" if use_a_tmem else "%1"
    a_operand_type = "uint32_t" if use_a_tmem else "uint64_t"
    mask_placeholders = ", ".join(
        [f"%{mask_start_idx + i}" for i in range(len(disable_output_lane))]
    )

    scale_placeholder = ""
    if enable_input_d and scale_input_d > 0:
        scale_operand_idx = mask_start_idx + len(disable_output_lane)
        scale_placeholder = f", %{scale_operand_idx}"

    # Build the list of input operands for the asm block
    input_operands_list = [
        '"r"(d_tmem_addr)',  # %0
        f'"{("r" if use_a_tmem else "l")}"(a_operand)',  # %1
        '"l"(b_desc)',  # %2
    ]
    if sparse:
        input_operands_list.append('"r"(sp_tmem_addr)')  # %3 (sparse only)
    input_operands_list.extend(
        [
            '"r"(i_desc)',  # %3 or %4
            f'"r"({"1" if enable_input_d else "0"})',  # %4 or %5
        ]
    )
    for i in range(len(disable_output_lane)):
        input_operands_list.append(f'"r"(mask{i})')
    if enable_input_d and scale_input_d > 0:
        input_operands_list.append(f'"n"({scale_input_d})')

    input_operands_list = ", ".join(input_operands_list)

    func_name = (
        f"ptx_tcgen05_mma_cta_{cta_group}_kind_{kind}"
        + ("_sp" if sparse else "")
        + ("TS" if use_a_tmem else "SS")
        + ("_enable_input_d" if enable_input_d else "")
        + (f"_{scale_input_d}" if scale_input_d > 0 else "")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t d_tmem_addr, {a_operand_type} a_operand, uint64_t b_desc, {sp_tmem_addr_str}uint32_t i_desc, {mask_signature}) {{
    asm volatile(
        "{{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, %{p_operand_idx}, 0;\\n"
        "tcgen05.mma{sparse_instr_suffix}.cta_group::{cta_group}.kind::{kind} [%0], {a_operand_str}, %2, {i_sp_operand_str} "
        "{{{mask_placeholders}}}, p{scale_placeholder};\\n"
        "}}\\n"
        :
        :  {input_operands_list}
    );
}}
"""

    args = [func_name, d_tmem_addr, a_operand, b_desc]
    if sparse:
        args.append(sp_tmem_addr)
    args.append(i_desc)
    args.extend(disable_output_lane)

    return cuda_func_call(*args, source_code=source_code)


@register_codegen("ptx_tcgen05_mma")
def codegen_ptx_tcgen05_mma(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
):
    return _tcgen05_mma_common(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    )


@register_codegen("ptx_tcgen05_mma_sp")
def codegen_ptx_tcgen05_mma_sp(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
):
    return _tcgen05_mma_common(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


def _get_tcgen05_mma_scale_vec_size(kind: str, scale_dtype: str) -> int:
    """
    Determines the scale vector size for a tcgen05 MMA instruction.
    This is a direct translation of the C++ GetTcgen05MMAScaleVecSize function.
    """
    scale_vec_size = 0
    stype = PTXDataType.from_string(scale_dtype)

    if kind == "mxf8f6f4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 1
    elif kind == "mxf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}:
        scale_vec_size = 4

    if scale_vec_size <= 0:
        raise ValueError(
            f"Invalid scale vector size for Tcgen05 MMA, check failed for kind::{kind}, "
            f"scale_dtype: {scale_dtype}"
        )
    return scale_vec_size


def _tcgen05_mma_block_scaled_common(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    sparse=False,
    sp_tmem_addr=None,
):
    d_dtype = str(d_dtype)[1:-1]
    a_dtype = str(a_dtype)[1:-1]
    b_dtype = str(b_dtype)[1:-1]
    sfa_dtype = str(sfa_dtype)[1:-1]
    sfb_dtype = str(sfb_dtype)[1:-1]
    use_a_tmem = bool(use_a_tmem)
    cta_group = int(cta_group)
    enable_input_d = bool(enable_input_d)

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype}, a:{a_dtype}, b:{b_dtype}, sfa:{sfa_dtype}, sfb:{sfb_dtype}"
        )

    scale_vec_size = _get_tcgen05_mma_scale_vec_size(kind, sfa_dtype)

    sparse_instr_suffix = ".sp" if sparse else ""
    sparse_placeholder = "[%7], " if sparse else ""
    a_constraint = '"r"' if use_a_tmem else '"l"'
    a_operand_type = "uint32_t" if use_a_tmem else "uint64_t"
    a_operand_placeholder = "[%1]" if use_a_tmem else "%1"
    enable_input_d_str = "1" if enable_input_d else "0"
    sp_tmem_addr_str = "uint32_t sp_tmem_addr, " if sparse else ""
    sp_tmem_addr_operand = f', "r"({sp_tmem_addr})' if sparse else ""

    func_name = (
        f"ptx_tcgen05_mma_block_scaled_cta_{cta_group}_kind_{kind}_scale_vec_{scale_vec_size}"
        + ("_sp" if sparse else "")
        + ("TS" if use_a_tmem else "SS")
        + ("_enable_input_d" if enable_input_d else "")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t d_tmem_addr, {a_operand_type} a_operand, uint64_t b_desc, {sp_tmem_addr_str}uint32_t i_desc, uint32_t sfa_tmem_addr, uint32_t sfb_tmem_addr) {{
    asm volatile(
        "{{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma{sparse_instr_suffix}.cta_group::{cta_group}.kind::{kind}.block_scale.scale_vec::{scale_vec_size}X "
        "[%0], {a_operand_placeholder}, %2, {sparse_placeholder}%3, [%5], [%6], p;\\n"
        "}}\\n"
        :
        : "r"(d_tmem_addr), {a_constraint}(a_operand), "l"(b_desc), "r"(i_desc), "r"({enable_input_d_str}), "r"(sfa_tmem_addr), "r"(sfb_tmem_addr){sp_tmem_addr_operand}
    );
}}
"""
    args = [func_name, d_tmem_addr, a_operand, b_desc]
    if sparse:
        args.append(sp_tmem_addr)
    args.append(i_desc)
    args.append(sfa_tmem_addr)
    args.append(sfb_tmem_addr)

    return cuda_func_call(*args, source_code=source_code)


@register_codegen("ptx_tcgen05_mma_block_scale")
def codegen_ptx_tcgen05_mma_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=True,
):
    return _tcgen05_mma_block_scaled_common(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


@register_codegen("ptx_tcgen05_mma_sp_block_scale")
def codegen_ptx_tcgen05_mma_sp_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=True,
):
    return _tcgen05_mma_block_scaled_common(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


@register_codegen("ptx_tcgen05_commit")
def codegen_ptx_tcgen05_commit(bar, cta_group, cta_mask):
    cta_group = int(cta_group)
    cta_mask = int(cta_mask)

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")

    is_multicast = cta_mask != 0

    if is_multicast:
        multicast_str = ".multicast::cluster"
        mask_operand_str = ", %1"
        cta_mask_arg_str = ', "h"(cta_mask)'
    else:
        multicast_str = ""
        mask_operand_str = ""
        cta_mask_arg_str = ""

    func_name = "ptx_tcgen05_commit_cta_group_" + str(cta_group)
    if is_multicast:
        func_name += "_multicast"

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* bar, int cta_mask_) {{
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_);
  __asm__ __volatile__(
    "tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster{multicast_str}.b64 [%0]{mask_operand_str};"
    :
    :"r"(bar_addr){cta_mask_arg_str}
  );
}}
"""
    return cuda_func_call(func_name, bar, cta_mask, source_code=source_code)


@register_codegen("ptx_tcgen05_cp")
def codegen_ptx_tcgen05_cp(
    dst_addr,
    row_offset,
    col_offset,
    src_desc,
    shape,
    dst_dtype,
    src_dtype,
    cta_group=1,
    multicast="",
):
    shape = str(shape)[1:-1]
    dst_dtype = str(dst_dtype)[1:-1]
    src_dtype = str(src_dtype)[1:-1]
    cta_group = int(cta_group)
    multicast = str(multicast)[1:-1]

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")

    valid_shapes = {"128x256b", "4x256b", "128x128b", "64x128b", "32x128b"}
    if shape not in valid_shapes:
        raise ValueError(f"Invalid shape for tcgen05 copy, check failed for shape: {shape}")

    err_msg = f"Invalid multicast for tcgen05 copy, check failed for shape: {shape}, multicast: {multicast}"
    if shape == "64x128b":
        if multicast not in {"warpx2::02_13", "warpx2::01_23"}:
            raise ValueError(err_msg)
    elif shape == "32x128b":
        if multicast != "warpx4":
            raise ValueError(err_msg)
    else:
        if multicast != "":
            raise ValueError(err_msg)

    # 2. Decompression Format Logic
    # ===============================
    dst_src_fmt = ""
    dtype_enum = PTXDataType.from_string(dst_dtype)
    stype_enum = PTXDataType.from_string(src_dtype)

    fp8_types = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.FLOAT8_E8M0FNU,
    }
    fp6_types = {PTXDataType.FLOAT6_E2M3FN, PTXDataType.FLOAT6_E3M2FN}

    if dtype_enum in fp8_types:
        if stype_enum == PTXDataType.FLOAT4_E2M1FN:
            dst_src_fmt = ".b8x16.b4x16_p64"
        elif stype_enum in fp6_types:
            dst_src_fmt = ".b8x16.b6x16_p32"

    multicast_str = f".{multicast}" if multicast else ""

    func_name = f"ptx_tcgen05_cp_cta_group_{cta_group}_shape_{shape}_multicast_{multicast}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t dst_addr, int row_offset, int col_offset, uint64_t src_desc) {{
    asm volatile(
        "tcgen05.cp.cta_group::{cta_group}.{shape}{multicast_str}{dst_src_fmt} [%0], %1;"
        :
        : "r"(get_tmem_addr(dst_addr, row_offset, col_offset)), "l"(src_desc)
    );
}}
"""
    return cuda_func_call(
        func_name,
        dst_addr,
        row_offset,
        col_offset,
        src_desc,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_shift")
def codegen_ptx_tcgen05_shift(taddr, cta_group):
    cta_group = int(cta_group)

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")

    func_name = f"ptx_tcgen05_shift_cta_group_{cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t taddr) {{
  __asm__ __volatile__(
    "tcgen05.shift.cta_group::{cta_group}.down %0;"
    :: "r"(taddr)
  );
}}
"""
    return cuda_func_call(func_name, taddr, source_code=source_code)


########################################################
# PTX Parallel Synchronization and Communication Instructions
########################################################


@register_codegen("ptx_bar_arrive")
def codegen_ptx_bar_arrive(name_bar_id, thread_count):
    func_name = "tvm_builtin_ptx_bar_arrive"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int name_bar_id, int thread_count) {{
    asm volatile("bar.arrive %0, %1;" : : "r"(name_bar_id), "r"(thread_count));
}}
"""
    return cuda_func_call(func_name, name_bar_id, thread_count, source_code=source_code)


@register_codegen("ptx_bar_sync")
def codegen_ptx_bar_sync(name_bar_id, thread_count):
    func_name = "tvm_builtin_ptx_bar_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int name_bar_id, int thread_count) {{
    asm volatile("bar.sync %0, %1;" : : "r"(name_bar_id), "r"(thread_count));
}}
"""
    return cuda_func_call(func_name, name_bar_id, thread_count, source_code=source_code)


@register_codegen("ptx_fence_proxy")
def codegen_ptx_fence_proxy(scope):
    scope = str(scope)[1:-1]
    func_name = f"tvm_builtin_ptx_fence_proxy_{scope}"

    if scope == "shared":
        ptx_scope = ".async.shared::cta"
    elif scope == "global":
        ptx_scope = ".async.global"
    else:
        raise ValueError(f"Invalid scope for ptx_fence_proxy: {scope}")

    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  __asm__ __volatile__("fence.proxy{ptx_scope};");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_fence_mbarrier_init_release_cluster")
def codegen_ptx_fence_mbarrier_init_release_cluster():
    func_name = "tvm_builtin_ptx_fence_mbarrier_init_release_cluster"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("fence.mbarrier_init.release.cluster;");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_barrier_cluster_arrive")
def codegen_ptx_barrier_cluster_arrive(sem, aligned):
    sem = str(sem)[1:-1]
    aligned = bool(aligned)

    sem_name = "" if len(sem) == 0 else f"_{sem}"
    aligned_name = "_aligned" if aligned else ""

    sem_inst = "" if len(sem) == 0 else f".{sem}"
    aligned_inst = ".aligned" if aligned else ""

    func_name = f"tvm_builtin_ptx_barrier_cluster_arrive{sem_name}{aligned_name}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("barrier.cluster.arrive{sem_inst}{aligned_inst};\\n" : :);
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_barrier_cluster_wait")
def codegen_ptx_barrier_cluster_wait(acquire, aligned):
    acquire_name = "_acquire" if bool(acquire) else ""
    aligned_name = "_aligned" if bool(aligned) else ""

    acquire_inst = ".acquire" if bool(acquire) else ""
    aligned_inst = ".aligned" if aligned else ""

    func_name = f"tvm_builtin_ptx_barrier_cluster_wait{acquire_name}{aligned_name}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("barrier.cluster.wait{acquire_inst}{aligned_inst};\\n" : :);
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_elect_sync")
def codegen_ptx_elect_sync(mask):
    func_name = "tvm_builtin_ptx_elect_sync"
    source_code = f"""
__forceinline__ __device__ uint32_t {func_name}(uint32_t mask) {{
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{{\\n"
      ".reg .b32 %rx;\\n"
      ".reg .pred %px;\\n"
      "     elect.sync %rx|%px, %2;\\n"
      "@%px mov.s32 %1, 1;\\n"
      "     mov.s32 %0, %rx;\\n"
      "}}\\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(mask));
  return pred;
}}
"""
    return cuda_func_call(func_name, mask, source_code=source_code)


#################### mbarrier


@register_codegen("ptx_mbarrier_init")
def codegen_ptx_mbarrier_init(bar, thread_count):
    func_name = "tvm_builtin_ptx_mbarrier_init"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int thread_count) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.init.shared.b64 [%0], %1;"
    :: "r"(barrier_addr_int), "r"(thread_count)
  );
}}
"""
    return cuda_func_call(func_name, bar, thread_count, source_code=source_code)


@register_codegen("ptx_mbarrier_arrive")
def codegen_ptx_mbarrier_arrive(bar, cta_id=None, pred=None):
    remote = cta_id is not None and pred is not None
    if not remote:
        func_name = "tvm_builtin_ptx_mbarrier_arrive"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.shared.b64 _, [%0];"
    :: "r"(barrier_addr_int)
  );
}}
"""
        return cuda_func_call(func_name, bar, source_code=source_code)
    else:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_remote"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int cta_id, int pred) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{{\\n"
      ".reg .pred p;\\n"
      ".reg .b32 remAddr32;\\n"
      "setp.eq.u32 p, %2, 1;\\n"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\\n"
      "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\\n"
      "}}\\n"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred));
}}
"""
        return cuda_func_call(func_name, bar, cta_id, pred, source_code=source_code)


@register_codegen("ptx_mbarrier_arrive_expect_tx")
def codegen_ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None):
    remote = cta_id is not None and pred is not None
    if not remote:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
    :: "r"(barrier_addr_int), "r"(byte_count)
  );
}}
"""
        return cuda_func_call(func_name, bar, byte_count, source_code=source_code)
    else:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx_remote"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count, int cta_id, int pred) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{{\\n"
      ".reg .pred p;\\n"
      ".reg .b32 remAddr32;\\n"
      "setp.eq.u32 p, %2, 1;\\n"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\\n"
      "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\\n"
      "}}\\n"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred), "r"(byte_count));
}}
"""
        return cuda_func_call(func_name, bar, byte_count, cta_id, pred, source_code=source_code)


@register_codegen("ptx_mbarrier_try_wait")
def codegen_ptx_mbarrier_try_wait(bar, phase):
    func_name = "tvm_builtin_ptx_mbarrier_wait"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int phase) {{
   unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile (
      "{{\\n"
      ".reg .pred                P1;\\n"
      "LAB_WAIT:\\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\\n"
      "@P1                       bra.uni DONE;\\n"
      "bra.uni                   LAB_WAIT;\\n"
      "DONE:\\n"
      "}}\\n"
      ::
      "r"(barrier_addr_int),
      "r"(phase)
  );
}}
"""
    return cuda_func_call(func_name, bar, phase, source_code=source_code)


########################################################
# PTX Data Movement and Conversion Instructions
########################################################


# see https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp#L186
CacheHint = {
    "evict_normal": 0x1000000000000000,
    "evict_first": 0x12F0000000000000,
    "evict_last": 0x14F0000000000000,
}


#################### TMA (cp.async.bulk.tensor)


@register_codegen("ptx_cp_async_bulk_tensor_global_to_cluster")
def codegen_ptx_cp_async_bulk_tensor_global_to_cluster(dim, dst_ptr, bar, tensormap, *args):
    dim = int(dim)
    coords, cta_mask, cta_group, cache_hint = args[:-3], int(args[-3]), int(args[-2]), args[-1]
    func_name = f"tvm_builtin_ptx_cp_async_bulk_tensor_global_to_cluster_{dim}d"
    if len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )
    if cache_hint != "":
        cache_hint = str(cache_hint)[1:-1]

    func_name = (
        f"ptx_cp_async_bulk_tensor_global_to_cluster_{dim}d"
        + (f"_multicast_{cta_mask}" if cta_mask != 0 else "")
        + (f"_cta_group_{cta_group}" if cta_group == 2 else "")
        + (f"_{cache_hint}" if cache_hint != "" else "")
    )
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(dim)])

    # The operand indices are different for unicast vs. multicast
    coord_arg_start = 3
    if cta_mask != 0:
        coord_arg_start += 1
    if cache_hint != "":
        coord_arg_start += 1
    coord_arg_template = "{%" + ", %".join([str(coord_arg_start + i) for i in range(dim)]) + "}"
    coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    def is_sm100_or_higher():
        target = tvm.target.Target.current()
        if target is None:
            return False
        arch = target.arch[3:]
        if not arch[-1].isdigit():
            arch = arch[:-1]
        return int(arch) >= 100

    is_sm100_or_higher = is_sm100_or_higher()

    if cta_group == 2 or (cta_group != -1 and is_sm100_or_higher):
        cta_group_str = f".cta_group::{cta_group}"
    else:
        cta_group_str = ""

    if cache_hint != "":
        cache_hint_str = f".L2::cache_hint"
    else:
        cache_hint_str = ""

    if cta_mask != 0:
        cache_hint_operand = f", %4" if cache_hint != "" else ""
        cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  uint16_t cta_mask = static_cast<uint16_t>({cta_mask});
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster{cta_group_str}{cache_hint_str}"
    " [%0], [%1, {coord_arg_template}], [%2], %3{cache_hint_operand};"
    :
    : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr), "h"(cta_mask){cache_hint_value},
        {coord_list_constraints}
    : "memory"
  );
}}
"""
    else:
        cache_hint_operand = f", %3" if cache_hint != "" else ""
        cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes{cta_group_str}{cache_hint_str}"
    " [%0], [%1, {coord_arg_template}], [%2]{cache_hint_operand};"
    :
    : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr){cache_hint_value},
        {coord_list_constraints}
    : "memory"
  );
}}
"""
    return cuda_func_call(func_name, dst_ptr, bar, tensormap, *coords, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_tensor_shared_to_global")
def codegen_ptx_cp_async_bulk_tensor_shared_to_global(dim, src_ptr, tensormap, *coords):
    dim = int(dim)
    if len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )

    func_name = f"ptx_cp_async_bulk_tensor_shared_to_global_{dim}d"
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(dim)])

    coord_indices = [str(2 + i) for i in range(dim)]
    arg_template = "{%" + ", %".join(coord_indices) + "}"

    coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* src, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int src_addr = __cvta_generic_to_shared(src);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"
    " [%0, {arg_template}], [%1];"
    :
    : "l"(tensormap_addr), "r"(src_addr),
      {coord_list_constraints}
    : "memory"
  );
}}
"""
    return cuda_func_call(func_name, src_ptr, tensormap, *coords, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_commit_group")
def codegen_ptx_cp_async_bulk_tensor_commit_group():
    func_name = "ptx_cp_async_bulk_tensor_commit_group"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("cp.async.bulk.commit_group;");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_wait_group")
def codegen_ptx_cp_async_bulk_wait_group(n, read):
    n = int(n)
    read_str = ".read" if bool(read) else ""
    func_name = "ptx_cp_async_bulk_wait_group" + ("_read" if bool(read) else "") + f"_{n}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("cp.async.bulk.wait_group{read_str} %0;" :: "n"({n}): "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


########################################################
# PTX Miscellaneous
########################################################


@register_codegen("ptx_setmaxnreg")
def codegen_ptx_setmaxnreg(inc, nreg):
    inc = bool(inc)
    nreg = int(nreg)
    action = "inc" if inc else "dec"
    func_name = f"tvm_builtin_ptx_setmaxnreg_{action}_{nreg}    "
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile( "setmaxnreg.{action}.sync.aligned.u32 %0;\\n" : : "n"({nreg}) );
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_ld_global_acquire")
def codegen_ptx_ld_global_acquire(res, addr):
    dtype = str(res.dtype)
    if dtype == "uint32":
        dtype_str, type_str, specifier = "uint32_t", "b32", "r"
    elif dtype == "int32":
        dtype_str, type_str, specifier = "int32_t", "b32", "r"
    elif dtype == "uint64":
        dtype_str, type_str, specifier = "uint64_t", "b64", "l"
    elif dtype == "int64":
        dtype_str, type_str, specifier = "int64_t", "b64", "l"
    else:
        raise ValueError(f"Unsupported data type for ld.global.acquire: {dtype}")

    func_name = f"tvm_builtin_ptx_ld_global_acquire_{type_str}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({dtype_str}& res,{dtype_str}* addr) {{
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile ("ld.global.acquire.gpu.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #else
  asm volatile ("ld.global.cg.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #endif
}}
"""
    return cuda_func_call(func_name, res, addr, source_code=source_code)


@register_codegen("ptx_map_shared_rank")
def codegen_ptx_map_shared_rank(ptr, rank):
    func_name = "tvm_builtin_ptx_map_shared_rank"
    source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(void* addr, uint32_t rank) {{
    uint64_t result;
    asm volatile("mapa.u64  %0, %1, %2;\\n"
                : "=l"(result)
                : "l"(reinterpret_cast<uint64_t>(addr)), "r"(rank));
    return result;
}}
"""
    return cuda_func_call(func_name, ptr, rank, source_code=source_code, return_type="uint64")


@register_codegen("ptx_fetch_register")
def codegen_ptx_fetch_register(bits, reg):
    bits = int(bits)
    reg = str(reg)[1:-1]

    if bits not in [32, 64]:
        raise ValueError(f"Only support 32/64 bits for ptx_fetch_register, but got {bits}.")

    func_name_safe_reg = reg.replace(".", "_")

    func_name = f"tvm_builtin_ptx_fetch_register_{func_name_safe_reg}"
    source_code = f"""
__forceinline__ __device__ int{bits}_t {func_name}() {{
  uint{bits}_t x;
  asm volatile("mov.u{bits} %0, %{reg};\\n" : "=r"(x) : "r"(reg));
  return (int{bits}_t)x;
}}
"""
    return cuda_func_call(func_name, source_code=source_code, return_type=f"int{bits}")


########################################################
# Timer
########################################################


@register_codegen("timer_init_cuda")
def codegen_timer_init_cuda(
    profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id
):

    func_name = "tvm_builtin_timer_init_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int num_groups, int group_id) {{
    // timer init
    const uint32_t NBLOCKS = (uint32_t)(gridDim.x * gridDim.y * gridDim.z);
    const uint32_t BLOCK_IDX = (uint32_t)((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const uint32_t NGROUPS = num_groups;
    const uint32_t GROUP_ID = group_id;
    const uint32_t BLOCK_GROUP_IDX = BLOCK_IDX * NGROUPS + GROUP_ID;
    if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0)) {{
        profiler_buffer[0] = ((uint64_t)NGROUPS << 32) | NBLOCKS;
    }}
    profiler_write_offset[0] = 1 + BLOCK_GROUP_IDX;
    profiler_tag[0] = (uint64_t)BLOCK_GROUP_IDX << 12;
}}
"""
    return cuda_func_call(
        func_name,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        num_groups,
        group_id,
        source_code=source_code,
    )


@register_codegen("timer_start_cuda")
def codegen_timer_start_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    func_name = "tvm_builtin_timer_start_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int event_type, uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond) {{
    // timer start
    if (leader_cond) {{
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_tag[0] | (uint32_t)event_type << 2 | 0x0);
        profiler_write_offset[0] += profiler_write_stride;
    }}
    __threadfence_block();
}}
"""
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
        source_code=source_code,
    ), ["get_time_stamp"]


@register_codegen("timer_end_cuda")
def codegen_timer_end_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    func_name = "tvm_builtin_timer_end_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int event_type, uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond) {{
    // timer end
    __threadfence_block();
    if (leader_cond) {{
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_tag[0] | (uint32_t)event_type << 2 | 0x1);
        profiler_write_offset[0] += profiler_write_stride;
    }}
}}
"""
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
        source_code=source_code,
    ), ["get_time_stamp"]


########################################################
# CUDA C++ miscellaneous
########################################################


@register_codegen("cuda_atomic_add")
def codegen_cuda_atomic_add(res_addr, value):
    func_name = "tvm_builtin_cuda_atomic_add"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* addr, T value) {{
    return atomicAdd(addr, value);
}}
"""
    assert isinstance(value, tvm.tir.PrimExpr)
    return cuda_func_call(
        func_name, res_addr, value, source_code=source_code, return_type=value.dtype
    )


@register_codegen("cuda_thread_fence")
def codegen_cuda_thread_fence():
    func_name = "tvm_builtin_cuda_thread_fence"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __threadfence();
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_syncthreads_and")
def codegen_cuda_syncthreads_and(predicate):
    func_name = "tvm_builtin_cuda_syncthreads_and"
    source_code = f"""
__forceinline__ __device__ int {func_name}(int predicate) {{
    return __syncthreads_and(predicate);
}}
"""
    return cuda_func_call(func_name, predicate, source_code=source_code, return_type="int32")


@register_codegen("cuda_nano_sleep")
def codegen_cuda_nano_sleep(time):
    func_name = "tvm_builtin_cuda_nano_sleep"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t time) {{
    __nanosleep(time);
}}
"""
    return cuda_func_call(func_name, time, source_code=source_code)



@register_codegen("cuda_atomic_cas")
def codegen_cuda_atomic_cas(ptr, old_val, new_val):
    func_name = "tvm_builtin_cuda_atomic_cas"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* address, T compare, T val) {{
    return atomicCAS(address, compare, val);
}}
"""
    return cuda_func_call(
        func_name, ptr, old_val, new_val, source_code=source_code, return_type=old_val.dtype
    )


@register_codegen("cuda_printf")
def codegen_cuda_printf(fmt, *args):
    func_name = "tvm_builtin_cuda_printf"
    if isinstance(fmt, tvm.tir.StringImm):
        fmt = fmt.value
    fmt = repr(fmt)[1:-1]
    source_code = f"""
template<typename... Args>
__forceinline__ __device__ void {func_name}(const char* fmt, Args... args) {{
    printf(fmt, args...);
}}
"""
    return cuda_func_call(func_name, fmt, *args, source_code=source_code)


@register_codegen("cuda_ldg")
def codegen_cuda_ldg(addr, dtype):
    dtype = DataType(str(dtype)[1:-1])
    func_name = "tvm_builtin_cuda_ldg"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* src) {{
    return __ldg(src);
}}
"""
    return cuda_func_call(func_name, addr, source_code=source_code, return_type=dtype)


########################################################
# NVSHMEM related ops
########################################################


@register_codegen("nvshmem_my_pe")
def codegen_nvshmem_my_pe():
    func_name = "tvm_builtin_nvshmem_my_pe"
    source_code = R"""
__forceinline__ __device__ int32_t {func_name}() {{
    return nvshmem_my_pe();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code, return_type="int32"), ["nvshmem"]


@register_codegen("nvshmem_n_pes")
def codegen_nvshmem_n_pes():
    func_name = "tvm_builtin_nvshmem_n_pes"
    source_code = R"""
__forceinline__ __device__ int32_t {func_name}() {{
    return nvshmem_n_pes();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code, return_type="int32"), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi")
def codegen_nvshmem_getmem_nbi(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmem_getmem_nbi(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi")
def codegen_nvshmem_putmem_nbi(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmem_putmem_nbi(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi_warp")
def codegen_nvshmem_getmem_nbi_warp(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_getmem_nbi_warp(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi_warp")
def codegen_nvshmem_putmem_nbi_warp(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_putmem_nbi_warp(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi_block")
def codegen_nvshmem_getmem_nbi_block(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_getmem_nbi_block(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi_block")
def codegen_nvshmem_putmem_nbi_block(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_putmem_nbi_block(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_signal_op")
def codegen_nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_signal_op"
    source_code = R"""
__forceinline__ __device__ void {func_name}(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name, sig_addr, signal, sig_op_val.get(sig_op), pe, source_code=source_code
    ), ["nvshmem"]


@register_codegen("nvshmem_wait_until")
def codegen_nvshmem_wait_until(ivar, cmp, cmp_value, type):
    if not isinstance(cmp, str):
        cmp = cmp.value
    if not isinstance(type, str):
        type = type.value
    type_val = {"uint64_t": ("uint64_t", "uint64"), "uint64": ("uint64_t", "uint64")}
    assert type in type_val, f"Unsupported type for nvshmem_wait_until: {type}"
    TYPE = type_val.get(type)[1]

    func_name = f"tvm_builtin_nvshmem_{TYPE}_wait_until"
    source_code = R"""
__forceinline__ __device__ void {func_name}({type} *ivar, int cmp, {type} cmp_value) {{
    nvshmem_{TYPE}_wait_until(ivar, cmp, cmp_value);
}}
"""
    source_code = source_code.format(func_name=func_name, type=type_val.get(type)[0], TYPE=TYPE)
    cmp_val = {"eq": 0, "ne": 1, "gt": 2, "ge": 3, "lt": 4, "le": 5}
    assert cmp in cmp_val, f"Unsupported cmp operation in nvshmem_wait_until: {cmp}"
    return cuda_func_call(func_name, ivar, cmp_val.get(cmp), cmp_value, source_code=source_code), [
        "nvshmem"
    ]


@register_codegen("nvshmem_quiet")
def codegen_nvshmem_quiet():
    func_name = "tvm_builtin_nvshmem_quiet"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_quiet();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi")
def codegen_nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi_warp")
def codegen_nvshmem_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi_block")
def codegen_nvshmem_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_fence")
def codegen_nvshmem_fence():
    func_name = "tvm_builtin_nvshmem_fence"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_fence();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_barrier_all")
def codegen_nvshmem_barrier_all():
    func_name = "tvm_builtin_nvshmem_barrier_all"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_barrier_all();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]
