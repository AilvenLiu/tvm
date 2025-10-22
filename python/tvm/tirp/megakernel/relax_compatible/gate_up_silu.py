from tvm.script import tir as T, tirp as Tp

from tvm.tirp.megakernel.common import (
    KernelConfig,
    SmemManager
)
from tvm.tirp.megakernel.gate_up_silu import GateUpSiluTile
from tvm.tirp.megakernel.gemm import BarTMA2MMA, BarMMA2TMA, BarMMA2LD, BarLD2MMA


class FuseGateUpSiluTile(GateUpSiluTile):

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager):
        # alloc shared memory
        cls.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent").buffer
        cls.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        cls.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        cls.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        cls.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        cls.tile_idx = T.alloc_cell("int32", scope="local.persistent", name="tile_idx")
        cls.phase = T.alloc_buffer((1,), "int32", scope="local.persistent", name="phase")

    @staticmethod
    def get_func(N, K, ab_dtype, out_dtype, BLK_M, job_type, wait_level, notify_scope, notify_scope_id, prefetch_on=False):
        symbolic_tile = FuseGateUpSiluTile(N, K, ab_dtype, ab_dtype, 1, BLK_M, BLK_M, prefetch_on=prefetch_on, profiler_on=False)
        num_tiles = (1, symbolic_tile.N // symbolic_tile.BLK_N, 1)
        tile_size = (None, symbolic_tile.BLK_N, symbolic_tile.TILE_K)

        @T.prim_func(tirp=True, private=True)
        def gate_up_silu_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "gemm", "megakernel.job_type_id": job_type, "megakernel.wait_level": wait_level,
                        "megakernel.notify_scope": notify_scope, "megakernel.notify_scope_id": notify_scope_id})
            m = T.int32()
            A = T.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
            B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
            output = T.match_buffer(C_ptr, (m, N // 2), out_dtype, layout="default")
            A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            output_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            gemm_tile = T.meta_var(FuseGateUpSiluTile(N, K, ab_dtype, ab_dtype, 1, BLK_M, BLK_M, prefetch_on=prefetch_on, profiler_on=False))
            gemm_tile.set_tensor_map(A_tensor_map, B_tensor_map, output_tensor_map, A, B, output)
            gemm_tile.host_init()
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.persistent.init": True})
                    gemm_tile.class_init(smem_manager)
                with T.cta():
                    T.block_attr({"tirp.tile_class.prefetch": True})
                    if prefetch_on:
                        gemm_tile.prefetch(m_idx, n_idx, k_idx, None)
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    gemm_tile.run(m_idx, n_idx, k_idx, None)
                with T.cta():
                    T.block_attr({"tirp.tile_class.persistent.finalize": True})
                    gemm_tile.class_finalize()    
        return gate_up_silu_func, num_tiles, tile_size
