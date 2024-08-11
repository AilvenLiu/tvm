import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T
from difflib import unified_diff
import re


def generate_random_data(shape, dtype):
    np.random.seed(0)
    return np.random.randn(*shape).astype(dtype)


def create_tvm_arrays(data_np, device):
    return [tvm.nd.array(data, device=device) for data in data_np]


def build_and_run_tvm_func(sch, target, *args):
    func = tvm.build(sch.mod, target=target)
    func(*args)
    return func, args[-1]


def from_source(code):
    return tvm.script.from_source(code, tirp=False)


def verify_result(C_tvm, C_np):
    tvm.testing.assert_allclose(C_tvm.numpy(), C_np, rtol=1e-5)


def verify_tir_code(code):
    assert from_source(code).script() == code


def verify_cuda_code(func, dim_num, dtype, *dims):
    generated_code = func.imported_modules[0].get_source()

    # Extract the section between "// print_buffer starts" and "// print_buffer ends"
    match = re.search(r"// print_buffer starts(.*?)// print_buffer ends", generated_code, re.DOTALL)
    if not match:
        raise AssertionError("print_buffer section not found in generated code")

    print_buffer_section = match.group(1).strip()

    # Check the number of nested for-loops = dim_num
    loop_pattern = re.compile(r"for \(int i(\d+) = 0; i\1 < (\d+); \+\+i\1\)")
    loops = loop_pattern.findall(print_buffer_section)
    if len(loops) != dim_num:
        raise AssertionError(f"Expected {dim_num} nested loops, but found {len(loops)}")

    # Verify the loop limits = *dims in order
    loop_limits = [int(limit) for _, limit in loops]
    if loop_limits != list(dims):
        raise AssertionError(f"Expected loop limits {dims}, but found {loop_limits}")

    # Verify the printf statement and dtype
    dtype_to_printf = {"float32": "%f", "float16": "%f", "int32": "%d", "uint32": "%u"}
    expected_printf = dtype_to_printf.get(dtype)
    if not expected_printf:
        raise AssertionError(f"Unsupported dtype {dtype}")
    if dtype == "float16":
        printf_pattern = re.compile(
            r'printf\("'
            + re.escape(expected_printf)
            + r'"\s*,\s*static_cast<float>\(\w+\[idx\]\)\s*\)'
        )
    else:
        printf_pattern = re.compile(
            r'printf\("' + re.escape(expected_printf) + r'"\s*,\s*\w+\[idx\]\s*\)'
        )

    if not printf_pattern.search(print_buffer_section):
        raise AssertionError(
            f'Expected printf statement with format "{expected_printf}", but not found'
        )


def test_print():
    DEV = tvm.cuda()
    target = tvm.target.Target.from_device(DEV)

    def vector_add_1D(dtype, dtype_str):
        M = 6
        M_BLK = 6
        dim_num = 1
        A_np, B_np = generate_random_data((M,), dtype), generate_random_data((M,), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M,), dtype_str)
            B = T.match_buffer(B_ptr, (M,), dtype_str)
            C = T.match_buffer(C_ptr, (M,), dtype_str)

            for i in T.grid(M):
                with T.block("C"):
                    vi = T.axis.spatial(M, i)
                    C[vi] = A[vi] + B[vi]
                T.print_buffer(C.data, dtype_str, dim_num, M)

        sch = tvm.tir.Schedule(add_func)
        blk = sch.get_block("C")
        i = sch.get_loops(blk)[0]

        i0, i1 = sch.split(i, factors=[None, M_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(i1, "threadIdx.x")

        C_np_tmp = np.zeros((M,), dtype=dtype)
        C_tvm = tvm.nd.array(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code(func, dim_num, dtype_str, M)

    def vector_add_2D(dtype, dtype_str):
        M, N = 6, 6
        M_BLK, N_BLK = 6, 6
        dim_num = 2
        A_np, B_np = generate_random_data((M, N), dtype), generate_random_data((M, N), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), dtype_str)
            B = T.match_buffer(B_ptr, (M, N), dtype_str)
            C = T.match_buffer(C_ptr, (M, N), dtype_str)

            for i, j in T.grid(M, N):
                with T.block("C"):
                    vi = T.axis.spatial(M, i)
                    vj = T.axis.spatial(N, j)
                    C[vi, vj] = A[vi, vj] + B[vi, vj]
                T.print_buffer(C.data, C.dtype, dim_num, M, N)

        sch = tvm.tir.Schedule(add_func)
        blk = sch.get_block("C")
        i, j = sch.get_loops(blk)

        i0, i1 = sch.split(i, factors=[None, M_BLK])
        j0, j1 = sch.split(j, factors=[None, N_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(j0, "blockIdx.y")
        sch.bind(i1, "threadIdx.x")
        sch.bind(j1, "threadIdx.y")

        C_np_tmp = np.zeros((M, N), dtype=dtype)
        C_tvm = tvm.nd.array(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code(func, dim_num, dtype_str, M, N)

    def vector_add_3D(dtype, dtype_str):
        M, N, K = 6, 6, 6
        M_BLK, N_BLK, K_BLK = 6, 6, 6
        dim_num = 3
        A_np, B_np = generate_random_data((M, N, K), dtype), generate_random_data((M, N, K), dtype)
        C_np = A_np + B_np

        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N, K), dtype_str)
            B = T.match_buffer(B_ptr, (M, N, K), dtype_str)
            C = T.match_buffer(C_ptr, (M, N, K), dtype_str)

            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi = T.axis.spatial(M, i)
                    vj = T.axis.spatial(N, j)
                    vk = T.axis.spatial(K, k)
                    C[vi, vj, vk] = A[vi, vj, vk] + B[vi, vj, vk]
                T.print_buffer(C.data, C.dtype, dim_num, M, N, K)

        sch = tvm.tir.Schedule(add_func)
        blk = sch.get_block("C")
        i, j, k = sch.get_loops(blk)

        i0, i1 = sch.split(i, factors=[None, M_BLK])
        j0, j1 = sch.split(j, factors=[None, N_BLK])
        k0, k1 = sch.split(k, factors=[None, K_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(j0, "blockIdx.y")
        sch.bind(k0, "blockIdx.z")
        sch.bind(i1, "threadIdx.x")
        sch.bind(j1, "threadIdx.y")
        sch.bind(k1, "threadIdx.z")

        C_np_tmp = np.zeros((M, N, K), dtype=dtype)
        C_tvm = tvm.nd.array(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code(func, dim_num, dtype_str, M, N, K)

    vector_add_1D(np.float32, "float32")
    vector_add_2D(np.int32, "int32")
    vector_add_2D(np.float16, "float16")
    vector_add_3D(np.uint32, "uint32")


if __name__ == "__main__":
    test_print()
