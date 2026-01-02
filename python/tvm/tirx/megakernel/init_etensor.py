from tvm.script import tir as T
from tvm.tirx.megakernel.common import SemaphoreBase, KernelConfig, Tile


import functools


def convert_1d_index_to_nd(idx, shape):
    nd_idx = []
    for i in reversed(range(len(shape))):
        nd_idx.append(idx % shape[i])
        idx = idx // shape[i]
    return list(reversed(nd_idx))

f_init_const = lambda c: lambda *args: c

def f_init_unmatched_dim(dim_len, in_par_size, out_par_size):
    def f_init(i):
        start_out_par = i * out_par_size
        end_out_par = T.min(dim_len, (i + 1) * out_par_size)
        return (end_out_par - 1) // in_par_size - start_out_par // in_par_size + 1
    
    return f_init

class InitETensorTile(Tile):
    
    VEC_SIZE = 1
    
    def __init__(self, etensor_and_f_init_pairs):
        super().__init__()
        self.etensor_and_f_init_pairs = etensor_and_f_init_pairs
        self.total_num_etensors = len(etensor_and_f_init_pairs)
        
    # only set etensor_init_complete in static scheduler
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            if_frames = [T.If(m_idx == i) for i in range(self.total_num_etensors)]
            then_frames = [T.Then() for i in range(self.total_num_etensors)]
            else_frames = [T.Else() for i in range(self.total_num_etensors - 1)]
            idx = T.alloc_local([1], "int32", name="idx")
            T.buffer_store(idx, tid * self.VEC_SIZE, [0])
            for i in range(self.total_num_etensors):
                if_frames[i].__enter__()
                with then_frames[i]:
                    etensor, f_init = self.etensor_and_f_init_pairs[i]
                    if f_init is None:
                        T.evaluate(0)
                    else:
                        nelem = functools.reduce(lambda x, y: x * y, etensor.shape, 1)
                        etensor_1d = etensor.view(-1).buffer
                        with T.While(idx[0] < nelem):
                            with T.vectorized(self.VEC_SIZE) as v:
                                T.buffer_store(etensor_1d, f_init(*convert_1d_index_to_nd(idx[0] + v, etensor.shape)) * (SemaphoreBase.base + 1), idx[0] + v)
                            T.buffer_store(idx, idx[0] + KernelConfig.NUM_THREADS * self.VEC_SIZE, [0])
                if i < self.total_num_etensors - 1:
                    else_frames[i].__enter__()
            for i in range(self.total_num_etensors - 1, -1, -1):
                if i < self.total_num_etensors - 1:
                    else_frames[i].__exit__(None, None, None)
                if_frames[i].__exit__(None, None, None)
