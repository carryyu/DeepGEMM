import pytest
import torch

from deep_gemm.utils import (
    cast_back_from_nvfp4,
    per_token_cast_to_nvfp4,
    unpack_ue4m3_from_int,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='NVFP4 reference tests require CUDA')


def test_nvfp4_hierarchical_scale_math():
    x = torch.zeros((4, 64), dtype=torch.bfloat16, device='cuda')
    x[1].fill_(1.0)
    x[2].fill_(448.0 * 6.0)
    x[3] = torch.logspace(-8, 8, 64, device='cuda').to(torch.bfloat16)

    packed, packed_sf, global_sf = per_token_cast_to_nvfp4(x)
    restored = cast_back_from_nvfp4(packed, packed_sf, global_sf)
    sf = unpack_ue4m3_from_int(packed_sf)

    assert packed.shape == (4, 32)
    assert packed_sf.shape == (4, 1)
    assert global_sf.shape == (4,) and global_sf.dtype == torch.float32
    assert global_sf[0].item() == 1.0
    assert global_sf[2].item() == 1.0
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert sf[2, 0].item() == 448.0
    assert sf.min().item() >= 2.0 ** -9
    assert sf.max().item() <= 448.0
    assert global_sf[3].item() != 1.0
    assert torch.isfinite(restored).all()


def test_nvfp4_global_scale_is_dequant_multiplier():
    torch.manual_seed(7)
    x = torch.randn((8, 256), dtype=torch.float32, device='cuda')
    x *= torch.pow(
        2.0, torch.linspace(-8, 8, 8, device='cuda').unsqueeze(1))
    packed, packed_sf, global_sf = per_token_cast_to_nvfp4(x)

    with_global = cast_back_from_nvfp4(packed, packed_sf, global_sf)
    without_global = cast_back_from_nvfp4(
        packed, packed_sf, torch.ones_like(global_sf))
    torch.testing.assert_close(
        with_global,
        without_global * global_sf.unsqueeze(1),
        rtol=0,
        atol=0,
    )


def test_nvfp4_bf16_staging_impact_is_bounded():
    torch.manual_seed(11)
    x_fp32 = torch.randn((64, 3072), dtype=torch.float32, device='cuda')
    x_fp32 *= torch.pow(
        2.0, torch.empty((64, 1), device='cuda').uniform_(-6, 6))
    x_bf16 = x_fp32.to(torch.bfloat16)

    direct = per_token_cast_to_nvfp4(x_fp32)
    staged = per_token_cast_to_nvfp4(x_bf16)
    direct_dequant = cast_back_from_nvfp4(*direct)
    staged_dequant = cast_back_from_nvfp4(*staged)

    code_change_rate = (direct[0] != staged[0]).float().mean()
    direct_error = (
        torch.linalg.vector_norm(direct_dequant - x_fp32) /
        torch.linalg.vector_norm(x_fp32)
    )
    staged_error = (
        torch.linalg.vector_norm(staged_dequant - x_fp32) /
        torch.linalg.vector_norm(x_fp32)
    )
    staged_vs_direct = (
        torch.linalg.vector_norm(staged_dequant - direct_dequant) /
        torch.linalg.vector_norm(direct_dequant)
    )

    # Propagate both activation variants through the same quantized L2
    # weights, so the regression also bounds the final GEMM impact.
    l2_weight = torch.randn(
        (128, x_fp32.size(1)), dtype=torch.bfloat16, device='cuda')
    l2_weight_dequant = cast_back_from_nvfp4(
        *per_token_cast_to_nvfp4(l2_weight))
    exact_l2 = x_fp32 @ l2_weight_dequant.t()
    direct_l2 = direct_dequant @ l2_weight_dequant.t()
    staged_l2 = staged_dequant @ l2_weight_dequant.t()
    direct_l2_error = (
        torch.linalg.vector_norm(direct_l2 - exact_l2) /
        torch.linalg.vector_norm(exact_l2)
    )
    staged_l2_error = (
        torch.linalg.vector_norm(staged_l2 - exact_l2) /
        torch.linalg.vector_norm(exact_l2)
    )

    assert code_change_rate.item() < 0.03
    assert (staged_error - direct_error).abs().item() < 0.002
    assert staged_vs_direct.item() < 0.05
    assert (staged_l2_error - direct_l2_error).abs().item() < 0.003
