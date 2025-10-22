from tvm.script import tir as T, tirp as Tp

from tvm.tirp.megakernel.common import (
    KernelConfig,
    SmemManager,
)

from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile

class FuseAddRMSNormTile(AddRMSNormTile):

    @staticmethod
    def get_func(batch_size, rms_norm_eps, hidden_size, job_type, wait_level, notify_scope, notify_scope_id):
        num_tiles = (batch_size, 1, 1)
        tile_size = (1, None, None)

        @T.prim_func(tirp=True, private=True)
        def add_rmsnorm_func(input_ptr: T.handle, residual_ptr: T.handle, weight_ptr: T.handle, output_ptr: T.handle, out_residual_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "add_rmsnorm", "megakernel.job_type_id": job_type, "megakernel.wait_level": wait_level,
                        "megakernel.notify_scope": notify_scope, "megakernel.notify_scope_id": notify_scope_id})
            seq_len = T.int32()
            input = T.match_buffer(input_ptr, (seq_len, hidden_size), "float16", scope="global")
            residual = T.match_buffer(residual_ptr, (seq_len, hidden_size), "float16", scope="global")
            weight = T.match_buffer(weight_ptr, (hidden_size), "float16", scope="global")
            output = T.match_buffer(output_ptr, (seq_len, hidden_size), "float16", scope="global")
            out_residual = T.match_buffer(out_residual_ptr, (seq_len, hidden_size), "float16", scope="global")
            add_rmsnorm_tile = T.meta_var(FuseAddRMSNormTile(rms_norm_eps, hidden_size))
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(add_rmsnorm_tile)
                add_rmsnorm_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    add_rmsnorm_tile.run(m_idx, n_idx, k_idx, input, residual, weight, output, out_residual)   
        return add_rmsnorm_func, num_tiles, tile_size
