#!/usr/bin/env python3
import argparse
import gc
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8


GRAN_K = 32


@dataclass(frozen=True)
class Case:
    name: str
    num_experts: int
    m: int
    n: int
    k: int


CASES: Dict[str, Case] = {
    "case3": Case("case3_group_gemm_nvfp4", 24, 16, 4096, 7168),
    "case4": Case("case4_group_gemm_nvfp4", 24, 16, 7168, 2048),
}


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cast_rows(x: torch.Tensor, dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if dtype == "fp4":
        return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=GRAN_K)
    if dtype == "fp8":
        return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=GRAN_K)
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_grouped_operand(
        num_groups: int,
        rows: int,
        k: int,
        dtype: str,
        keep_group0_reference: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype == "fp4":
        data = torch.empty((num_groups, rows, k // 2), dtype=torch.int8, device="cuda")
    else:
        data = torch.empty((num_groups, rows, k), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((num_groups, rows, k // GRAN_K), dtype=torch.float, device="cuda")
    group0_reference = None

    # Quantize group-by-group to keep peak temporary memory small for large expert weights.
    for group_idx in range(num_groups):
        source = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda")
        quantized, scales = cast_rows(source, dtype)
        data[group_idx].copy_(quantized)
        sf[group_idx].copy_(scales)
        if keep_group0_reference and group_idx == 0:
            group0_reference = source

    return data, sf, group0_reference


def transform_grouped_sf(sf: torch.Tensor, mn: int, k: int, num_groups: int) -> torch.Tensor:
    return deep_gemm.transform_sf_into_required_layout(
        sf, mn, k, (1, GRAN_K), num_groups
    )


def run_case(case: Case, num_tests: int, flush_l2: bool, check: bool) -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    print(f"\n[{case.name}]")
    print(
        f"  logical: activation=[{case.num_experts}, {case.m}, {case.k}] "
        f"(FP8, row-major), weight=[{case.num_experts}, {case.n}, {case.k}] "
        f"(FP4, NT/col-major), output=[{case.num_experts}, {case.m}, {case.n}] (BF16)"
    )

    a_data, a_sf, a_ref0 = make_grouped_operand(
        case.num_experts, case.m, case.k, "fp8", check
    )
    b_data, b_sf, b_ref0 = make_grouped_operand(
        case.num_experts, case.n, case.k, "fp4", check
    )

    # Pre-pack UE8M0 scales once. This keeps scale-layout conversion out of timing.
    a_sf = transform_grouped_sf(a_sf, case.m, case.k, case.num_experts)
    b_sf = transform_grouped_sf(b_sf, case.n, case.k, case.num_experts)
    a = (a_data, a_sf)
    b = (b_data, b_sf)

    d = torch.empty(
        (case.num_experts, case.m, case.n),
        dtype=torch.bfloat16,
        device="cuda",
    )
    masked_m = torch.full(
        (case.num_experts,), case.m, dtype=torch.int, device="cuda"
    )

    def kernel() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            case.m,
            recipe_a=(1, GRAN_K),
            recipe_b=(1, GRAN_K),
        )

    # First call triggers JIT compilation and validates all tensor layouts.
    kernel()
    torch.cuda.synchronize()

    if check:
        ref0 = (a_ref0.float() @ b_ref0.float().T).to(torch.bfloat16)
        diff = calc_diff(d[0], ref0)
        tolerance = 0.01
        print(f"  correctness: diff={diff:.6f}, tolerance={tolerance:.3f}")
        if diff >= tolerance:
            raise AssertionError(f"{case.name}: diff {diff} >= {tolerance}")

    elapsed = bench_kineto(
        kernel,
        "gemm_",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )

    flops = 2 * case.num_experts * case.m * case.n * case.k
    bytes_moved = sum(tensor_nbytes(t) for t in (a_data, a_sf, b_data, b_sf, d))
    print(
        f"  performance: {elapsed * 1e6:.1f} us | "
        f"{flops / elapsed / 1e12:.1f} TFLOPS | "
        f"{bytes_moved / elapsed / 1e9:.1f} GB/s"
    )
    print(
        f"  physical: activation.data={list(a_data.shape)} {a_data.dtype}, "
        f"activation.sf={list(a_sf.shape)} {a_sf.dtype}, "
        f"weight.data={list(b_data.shape)} {b_data.dtype}, "
        f"weight.sf={list(b_sf.shape)} {b_sf.dtype}"
    )

    del a, b, a_data, a_sf, b_data, b_sf, d, masked_m, a_ref0, b_ref0
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepGEMM grouped WFP4 x AFP8 on the two FFN cases."
    )
    parser.add_argument(
        "--case", choices=("all", *CASES.keys()), default="all",
        help="Case to run (default: all).",
    )
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--warm-cache", action="store_true",
        help="Do not flush L2 before each profiled launch.",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(
            f"FP4 grouped GEMM requires SM100; got capability {major}.{minor}"
        )

    print("DeepGEMM grouped GEMM benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")
    print(f"  mode: WFP4 x AFP8, E2M1/E4M3 + UE8M0 SF, gran_k={GRAN_K}")
    print("  accumulation/output: FP32/BF16")
    print(
        "  note: this is DeepGEMM MX-style FP4 scaling, not strict NVIDIA NVFP4 "
        "(FP8 scale with 16-value blocks)."
    )

    selected_cases = CASES.values() if args.case == "all" else (CASES[args.case],)
    for case in selected_cases:
        run_case(
            case,
            num_tests=args.num_tests,
            flush_l2=not args.warm_cache,
            check=not args.no_check,
        )


if __name__ == "__main__":
    main()
