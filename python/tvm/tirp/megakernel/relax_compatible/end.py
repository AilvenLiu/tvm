from tvm.script import tir as T, tirp as Tp

from tvm.tirp.megakernel.common import (
    Tile,
    KernelConfig,
)

class FuseEndTile(Tile):

    @staticmethod   
    def get_func():
        num_tiles = (KernelConfig.SM_NUMBER, 1, 1)
        tile_size = (None, None, None)

        @T.prim_func(tirp=True, private=True)
        def end(m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "end"})
            with T.cta():
                T.block_attr({"tirp.tile_class.run": True})
                T.add_to_parent(T.evaluate(0))
        return end, num_tiles, tile_size
