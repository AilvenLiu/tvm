"""Pytest wrapper for kernels/sm100/nvfp4_gemm.py (Tx.gemm_async based)."""
import sys
import pytest

import torch
import torch.nn.functional as F
import tvm
import tvm.testing

sys.path.insert(0, "kernels/sm100")
nvfp4_test = pytest.importorskip("nvfp4_gemm")


@pytest.mark.parametrize("M,N,K", [(1024, 1024, 1024), (4096, 4096, 4096)])
def test_nvfp4_gemm_correctness(M, N, K):
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = nvfp4_test.prepare_data(M, N, K)
    kernel = nvfp4_test.tir_ws_kernel(M, N, K)

    alpha_buf = torch.tensor([alpha.item()], device="cuda", dtype=torch.float32)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")

    ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_buf, out)
    torch.cuda.synchronize()

    cosine_sim = F.cosine_similarity(
        out.float().reshape(-1), C_ref.to("cuda").float().reshape(-1), dim=0
    )
    assert cosine_sim.item() > 0.97, f"cosine similarity {cosine_sim.item():.6f} < 0.97"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
