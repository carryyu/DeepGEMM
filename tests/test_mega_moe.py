import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple

import deep_gemm
from deep_gemm.utils import (
    align,
    cast_back_from_nvfp4,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
    per_token_cast_to_nvfp4,
)
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


def _to_shared_mega_moe_sf_layout(sf: torch.Tensor, block_m: int, num_max_sf_tokens: int) -> torch.Tensor:
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = align(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype, device=sf.device)
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(sf[block_idx * block_m + m_idx])
    return result


def _cast_fp8_for_mega_moe(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, align(mn, 4)), dtype=x_sf.dtype, device=x_sf.device)
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _to_mn_major_tma_aligned_sf(sf: torch.Tensor) -> torch.Tensor:
    """Copy packed int32 SF to an MN-major, 16-byte TMA-aligned tensor."""
    assert sf.dtype == torch.int32 and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)
    num_groups, mn, packed_sf_k = sf.shape
    aligned_mn = align(mn, 4)
    result = torch.empty_strided(
        (num_groups, mn, packed_sf_k),
        (aligned_mn * packed_sf_k, 1, aligned_mn),
        dtype=sf.dtype, device=sf.device)
    result.copy_(sf)
    return result.squeeze(0) if squeeze_group_dim else result


def _cast_nvfp4_weights(
    bf16_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize 2D/3D K-major weights without using the UE8M0 layout API."""
    assert bf16_weights.dim() in (2, 3)
    squeeze_group_dim = bf16_weights.dim() == 2
    weights = bf16_weights.unsqueeze(0) if squeeze_group_dim else bf16_weights
    num_groups, n, k = weights.shape
    assert k % 64 == 0
    data = torch.empty((num_groups, n, k // 2), device=weights.device, dtype=torch.int8)
    sf = torch.empty((num_groups, n, k // 64), device=weights.device, dtype=torch.int32)
    global_sf = torch.empty((num_groups, n), device=weights.device, dtype=torch.float32)
    for group_idx in range(num_groups):
        data[group_idx], sf[group_idx], global_sf[group_idx] = (
            per_token_cast_to_nvfp4(weights[group_idx]))
    sf = _to_mn_major_tma_aligned_sf(sf)
    if squeeze_group_dim:
        return data.squeeze(0), sf.squeeze(0), global_sf.squeeze(0)
    return data, sf, global_sf


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    is_nvfp4xnvfp4 = args.mma_type == 'nvfp4xnvfp4'
    assert args.mma_type in ('bf16xbf16', 'fp8xfp4', 'nvfp4xnvfp4')
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    num_shared_experts = args.num_shared_experts
    num_experts, num_topk = args.num_experts, args.num_topk
    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    assert num_tokens <= num_max_tokens_per_rank

    # Use a global token offset so every rank constructs a disjoint part of one
    # deterministic, balanced routing sequence, including uneven-token tests.
    local_num_tokens = torch.tensor([num_tokens], dtype=torch.int64, device='cuda')
    num_tokens_per_rank = [torch.zeros_like(local_num_tokens) for _ in range(num_ranks)]
    dist.all_gather(num_tokens_per_rank, local_num_tokens, group=group)
    num_tokens_per_rank = [int(count.item()) for count in num_tokens_per_rank]
    global_token_offset = sum(num_tokens_per_rank[:rank_idx])

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type=args.mma_type
    )

    # Cast routed W4A8 weights into packed FP4 + packed UE8M0.
    def _cast_weights_to_mxfp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, shared_x, shared_l1_x_sf, topk_idx, topk_weights, l1_weights, l2_weights
        global transformed_l1_weights, transformed_l2_weights
        global shared_l1_weights, shared_l2_weights, transformed_shared_l1_weights, transformed_shared_l2_weights
        global cumulative_local_expert_recv_stats_fused, cumulative_local_expert_recv_stats_baseline
        global initial_cumulative_local_expert_recv_stats_fused, initial_cumulative_local_expert_recv_stats_baseline
        shared_x = shared_l1_x_sf = None
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False).values

        # Interleave destination ranks first, then local experts. Every complete
        # cycle visits each global expert once; an incomplete cycle differs by
        # at most one route per expert and distributes the remainder across ranks.
        global_route_ids = (
            torch.arange(num_tokens * num_topk, dtype=torch.int64, device='cuda')
            + global_token_offset * num_topk
        )
        dst_rank = global_route_ids.remainder(num_ranks)
        local_expert = torch.div(
            global_route_ids, num_ranks, rounding_mode='floor'
        ).remainder(num_experts_per_rank)
        topk_idx = (
            dst_rank * num_experts_per_rank + local_expert
        ).view(num_tokens, num_topk)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_fused = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_baseline.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        if num_shared_experts > 0:
            shared_l1_weights = torch.randn(
                (shared_intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
            shared_l2_weights = torch.randn(
                (hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        else:
            shared_l1_weights = shared_l2_weights = None

        if is_nvfp4xnvfp4:
            # NVFP4 uses packed E2M1 activations/weights and packed positive
            # UE4M3 scales, one scale per 16 logical values.
            assert hidden % 256 == 0 and intermediate_hidden % 256 == 0
            assert shared_intermediate_hidden % 256 == 0
            # Exercise scales far from one with token/channel log-uniform
            # magnitudes.
            log_scale_span = float(
                os.getenv('DG_TEST_NVFP4_LOG_SCALE_SPAN', '8'))
            x_scale = torch.pow(
                2.0, torch.empty((num_tokens, 1), device='cuda').uniform_(
                    -log_scale_span, log_scale_span))
            x = (x.float() * x_scale).to(torch.bfloat16)
            l1_scale = torch.pow(
                2.0, torch.empty(
                    (num_experts_per_rank, intermediate_hidden * 2, 1),
                    device='cuda').uniform_(-log_scale_span, log_scale_span))
            l2_scale = torch.pow(
                2.0, torch.empty(
                    (num_experts_per_rank, hidden, 1),
                    device='cuda').uniform_(-log_scale_span, log_scale_span))
            l1_weights = (l1_weights.float() * l1_scale).to(torch.bfloat16)
            l2_weights = (l2_weights.float() * l2_scale).to(torch.bfloat16)
            if num_shared_experts > 0:
                shared_l1_scale = torch.pow(
                    2.0, torch.empty(
                        (shared_intermediate_hidden * 2, 1),
                        device='cuda').uniform_(-log_scale_span, log_scale_span))
                shared_l2_scale = torch.pow(
                    2.0, torch.empty((hidden, 1), device='cuda').uniform_(
                        -log_scale_span, log_scale_span))
                shared_l1_weights = (
                    shared_l1_weights.float() * shared_l1_scale).to(torch.bfloat16)
                shared_l2_weights = (
                    shared_l2_weights.float() * shared_l2_scale).to(torch.bfloat16)

            x_data, x_sf, x_global_sf = per_token_cast_to_nvfp4(x)
            x = (x_data, x_sf, x_global_sf)
            if num_shared_experts > 0:
                block_m = deep_gemm.get_block_m_for_mega_moe(
                    num_ranks, num_experts, buffer.num_max_tokens_per_rank,
                    num_tokens, num_topk, args.mma_type)
                shared_l1_x_sf = _to_shared_mega_moe_sf_layout(
                    x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])
            l1_weights = _cast_nvfp4_weights(l1_weights)
            l2_weights = _cast_nvfp4_weights(l2_weights)
            if num_shared_experts > 0:
                shared_l1_weights = _cast_nvfp4_weights(shared_l1_weights)
                shared_l2_weights = _cast_nvfp4_weights(shared_l2_weights)
        elif not is_bf16xbf16:
            # W4A8 path: FP8 activations and FP4 routed weights with UE8M0 SF.
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0 and shared_intermediate_hidden % 128 == 0
            block_m = deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, buffer.num_max_tokens_per_rank, num_tokens, num_topk, args.mma_type)
            x_fp8, x_sf, x_sf_tma = _cast_fp8_for_mega_moe(x)
            x = (x_fp8, x_sf)
            shared_x = (x_fp8, x_sf_tma)
            if num_shared_experts > 0:
                shared_l1_x_sf = _to_shared_mega_moe_sf_layout(x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])
            l1_weights = _cast_weights_to_mxfp4(l1_weights)
            l2_weights = _cast_weights_to_mxfp4(l2_weights)
            if num_shared_experts > 0:
                shared_l1_weights = _cast_fp8_for_mega_moe(shared_l1_weights)[0::2]
                shared_l2_weights = _cast_fp8_for_mega_moe(shared_l2_weights)[0::2]

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))
        if num_shared_experts > 0:
            transformed_shared_l1_weights, transformed_shared_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe(shared_l1_weights, shared_l2_weights))
        else:
            transformed_shared_l1_weights = transformed_shared_l2_weights = None

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def copy_inputs_to_buffer():
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
            if is_nvfp4xnvfp4:
                buffer.x_global_sf[:num_tokens].copy_(x[2])
            if num_shared_experts > 0:
                _copy_fp8_sf(buffer.shared_l1_acts_sf, shared_l1_x_sf, num_tokens)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused():
        cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
        copy_inputs_to_buffer()

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math))
        if num_shared_experts > 0:
            kernel_kwargs.update(
                shared_l1_weights=transformed_shared_l1_weights,
                shared_l2_weights=transformed_shared_l2_weights
            )
        kernel = deep_gemm.bf16_mega_moe if is_bf16xbf16 else (
            deep_gemm.nvfp4_nvfp4_mega_moe
            if is_nvfp4xnvfp4 else deep_gemm.fp8_fp4_mega_moe)
        kernel(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    def run_nvfp4_reference():
        """Slow decomposed reference using exactly the quantized NVFP4 operands."""
        assert is_nvfp4xnvfp4

        # Every expert rank needs all quantized tokens and routing metadata.
        gathered_x = uneven_all_gather(x[0], group=group)
        gathered_x_sf = uneven_all_gather(x[1], group=group)
        gathered_x_global_sf = uneven_all_gather(x[2], group=group)
        gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
        gathered_topk_weights = uneven_all_gather(topk_weights, group=group)
        # FP4 * E4M3 block-scaled values are exactly representable in BF16.
        # Keep the FP32 token global scale separate because the kernel applies
        # activation_global * weight_global after the MMA accumulator.
        gathered_x_base = cast_back_from_nvfp4(
            gathered_x, gathered_x_sf,
            torch.ones_like(gathered_x_global_sf)).to(torch.bfloat16)

        local_count = torch.tensor([num_tokens], dtype=torch.int64, device='cuda')
        gathered_counts = [torch.zeros_like(local_count) for _ in range(num_ranks)]
        dist.all_gather(gathered_counts, local_count, group=group)
        token_counts = [int(count.item()) for count in gathered_counts]
        local_token_start = sum(token_counts[:rank_idx])
        total_tokens = sum(token_counts)
        assert gathered_x_base.size(0) == total_tokens

        def nvfp4_reference_mm(
            lhs_base: torch.Tensor,
            lhs_global_sf: torch.Tensor,
            rhs_data: torch.Tensor,
            rhs_sf: torch.Tensor,
            rhs_global_sf: torch.Tensor,
        ) -> torch.Tensor:
            """Match the kernel's FP32 accumulation and post-MMA globals.

            Using BF16 inputs with an FP32 output selects the BF16 GEMM path
            while retaining FP32 accumulation. This avoids relying on CUDA
            SGEMM, which is unavailable in some SM100 PyTorch/CUDA builds.
            """
            rhs_base = cast_back_from_nvfp4(
                rhs_data, rhs_sf.contiguous(),
                torch.ones_like(rhs_global_sf)).to(torch.bfloat16)
            output = torch.mm(
                lhs_base, rhs_base.t(), out_dtype=torch.float32)
            output *= lhs_global_sf.float().unsqueeze(1)
            output *= rhs_global_sf.float().unsqueeze(0)
            return output

        def swiglu_staging(l1_output: torch.Tensor,
                          route_weights: Optional[torch.Tensor]) -> torch.Tensor:
            # The kernel rounds accumulators to BF16 before clamp/SwiGLU.
            gate_bf16, up_bf16 = l1_output.to(torch.bfloat16).chunk(2, dim=-1)
            if args.activation_clamp is not None:
                clamp = float(args.activation_clamp)
                gate_bf16 = torch.minimum(
                    gate_bf16, torch.tensor(clamp, dtype=torch.bfloat16, device='cuda'))
                up_bf16 = up_bf16.clamp(min=-clamp, max=clamp)
            gate = gate_bf16.float()
            activation = (gate * torch.sigmoid(gate)) * up_bf16.float()
            if route_weights is not None:
                activation *= route_weights.float().unsqueeze(1)
            return activation.to(torch.bfloat16)

        def swiglu_requantize(l1_output: torch.Tensor,
                              route_weights: Optional[torch.Tensor]):
            activation_bf16 = swiglu_staging(l1_output, route_weights)
            return per_token_cast_to_nvfp4(activation_bf16)

        # Keep one BF16 contribution per Top-K slot.
        routed_slots = torch.zeros(
            (total_tokens, num_topk, hidden),
            dtype=torch.bfloat16, device='cuda')
        local_expert_begin = rank_idx * num_experts_per_rank
        local_expert_end = local_expert_begin + num_experts_per_rank
        local_route_mask = (
            (gathered_topk_idx >= local_expert_begin) &
            (gathered_topk_idx < local_expert_end)
        )
        local_route_ids = gathered_topk_idx[local_route_mask] - local_expert_begin
        stats_delta = torch.bincount(
            local_route_ids, minlength=num_experts_per_rank).to(torch.int32)
        expected_stats = initial_cumulative_local_expert_recv_stats_fused + stats_delta

        for local_expert_idx in range(num_experts_per_rank):
            global_expert_idx = local_expert_begin + local_expert_idx
            token_ids, topk_slots = (gathered_topk_idx == global_expert_idx).nonzero(
                as_tuple=True)
            if token_ids.numel() == 0:
                continue

            l1_output = nvfp4_reference_mm(
                gathered_x_base[token_ids],
                gathered_x_global_sf[token_ids],
                l1_weights[0][local_expert_idx],
                l1_weights[1][local_expert_idx],
                l1_weights[2][local_expert_idx])
            l2_data, l2_sf, l2_global_sf = swiglu_requantize(
                l1_output, gathered_topk_weights[token_ids, topk_slots])
            l2_input_base = cast_back_from_nvfp4(
                l2_data, l2_sf, torch.ones_like(l2_global_sf)).to(torch.bfloat16)
            contribution = nvfp4_reference_mm(
                l2_input_base, l2_global_sf,
                l2_weights[0][local_expert_idx],
                l2_weights[1][local_expert_idx],
                l2_weights[2][local_expert_idx])
            contribution = contribution.to(torch.bfloat16)
            routed_slots[token_ids, topk_slots] = contribution
            del l1_output, l2_data, l2_sf, l2_global_sf, l2_input_base, contribution

        # Exactly one rank owns each routed expert, so the all-reduce only
        # transports the selected BF16 combine slots.
        dist.all_reduce(routed_slots, group=group)
        routed_output = routed_slots.float().sum(dim=1)
        local_output = routed_output[
            local_token_start:local_token_start + num_tokens]

        if num_shared_experts > 0:
            local_x_base = cast_back_from_nvfp4(
                x[0], x[1], torch.ones_like(x[2])).to(torch.bfloat16)
            shared_l1_output = nvfp4_reference_mm(
                local_x_base, x[2],
                shared_l1_weights[0], shared_l1_weights[1],
                shared_l1_weights[2])
            shared_l2_data, shared_l2_sf, shared_l2_global_sf = (
                swiglu_requantize(shared_l1_output, None))
            shared_l2_base = cast_back_from_nvfp4(
                shared_l2_data, shared_l2_sf,
                torch.ones_like(shared_l2_global_sf)).to(torch.bfloat16)
            shared_output = nvfp4_reference_mm(
                shared_l2_base, shared_l2_global_sf,
                shared_l2_weights[0], shared_l2_weights[1],
                shared_l2_weights[2])
            shared_output = shared_output.to(torch.bfloat16)
            local_output = local_output + shared_output.float()

        return local_output.to(torch.bfloat16), expected_stats

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    if is_nvfp4xnvfp4:
        dist_print(' > Combine: BF16', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    can_run_legacy_baseline = is_legacy_loaded and not is_nvfp4xnvfp4
    if is_legacy_loaded and is_nvfp4xnvfp4:
        dist_print('Legacy baseline does not implement NVFP4 global scaling; using the PyTorch reference only.',
                   once_in_node=True)
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=not is_bf16xbf16,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if can_run_legacy_baseline else None

    # Baseline params differ by mma type
    run_baseline = None
    if can_run_legacy_baseline:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            swiglu_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            swiglu_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def get_baseline_shared_bias() -> Optional[torch.Tensor]:
            if num_shared_experts == 0:
                return None

            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            if is_bf16xbf16:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.bf16_gemm_nt(x, shared_l1_weights, l1_out)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=128, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=False, ue8m0_scale=False, output_bf16=True)[-1]
                deep_gemm.bf16_gemm_nt(l2_in, shared_l2_weights, y)
            else:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.fp8_gemm_nt(shared_x, shared_l1_weights, l1_out, recipe=(1, 1, 32), disable_ue8m0_cast=True)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=32, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=True, ue8m0_scale=True, output_bf16=False)
                deep_gemm.fp8_gemm_nt(l2_in, shared_l2_weights, y, recipe=(1, 1, 32), disable_ue8m0_cast=True)
            return y

        def run_baseline():
            cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # SwiGLU
            swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                **swiglu_kwargs)
            l1_y = swiglu_result[-1] if is_bf16xbf16 else swiglu_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            return (
                ep_buffer.combine(l2_y, handle=handle, bias=get_baseline_shared_bias())[0],
                cumulative_local_expert_recv_stats_baseline
            )

    # Check correctness
    # noinspection PyBroadException
    if is_nvfp4xnvfp4 and num_correctness_tests > 0:
        dist_print('Running NVFP4 PyTorch-reference correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            reference_y, reference_stats = run_nvfp4_reference()
            assert torch.equal(fused_stats, reference_stats), (
                f'rank {rank_idx}: stats mismatch: fused={fused_stats}, '
                f'reference={reference_stats}')
            diff = calc_diff(fused_y, reference_y)
            if not isinstance(diff, torch.Tensor):
                diff = torch.tensor(diff, device='cuda')
            dist.all_reduce(diff, op=dist.ReduceOp.MAX, group=group)
            max_diff = float(diff.item())
            assert max_diff < 0.02, f'NVFP4 correctness diff {max_diff} >= 0.02'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(
                    f' > Correctness test #{i + 1}/{num_correctness_tests} passed, '
                    f'max calc_diff={max_diff:.6f}',
                    once_in_node=True)
        dist_print(once_in_node=True)
    elif can_run_legacy_baseline and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            baseline_y, baseline_stats = run_baseline()
            assert torch.equal(fused_stats, baseline_stats)
            if num_shared_experts == 0:
                assert torch.equal(fused_y, baseline_y)
            else:
                assert calc_diff(fused_y, baseline_y) < 1e-8
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    if args.masked_ratio == 0:
        expert_recv_counts = torch.bincount(
            gathered_topk_idx.flatten(), minlength=num_experts)
        assert int(expert_recv_counts.max() - expert_recv_counts.min()) <= 1
        rank_recv_counts = expert_recv_counts.view(
            num_ranks, num_experts_per_rank).sum(dim=1)
        assert int(rank_recv_counts.max() - rank_recv_counts.min()) <= 1
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    barrier_fn = lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json'
    t_fused = bench_kineto(run_fused, 'mega_moe', barrier=barrier_fn, trace_path=trace_path)
    t_baseline = tilelang_bench(
        run_baseline, _n_warmup=5, _n_repeat=1,
        backend='cudagraph', return_mode='median') / 1e3 if can_run_legacy_baseline else 0
    # TFLOPS: routed + shared L1/L2, each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3
    num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3
    num_total_flops = num_routed_flops + num_shared_flops

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    if is_bf16xbf16:
        act_data_bytes, weight_data_bytes, sf_bytes_per_elem = 2.0, 2.0, 0.0
    elif is_nvfp4xnvfp4:
        act_data_bytes, weight_data_bytes, sf_bytes_per_elem = 0.5, 0.5, 1.0 / 16
    else:
        act_data_bytes, weight_data_bytes, sf_bytes_per_elem = 1.0, 0.5, 1.0 / 32
    act_storage_bytes = act_data_bytes + sf_bytes_per_elem
    weight_storage_bytes = weight_data_bytes + sf_bytes_per_elem
    combine_storage_bytes = 2.0
    num_routed_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * weight_storage_bytes  # L1 weights + SF
        + num_touched_experts * hidden * intermediate_hidden * weight_storage_bytes    # L2 weights + SF
        + num_recv_tokens * hidden * act_storage_bytes                                 # L1 acts + SF read
        + num_recv_tokens * intermediate_hidden * act_storage_bytes                    # L1 output + SF write
        + num_recv_tokens * intermediate_hidden * act_storage_bytes                    # L2 acts + SF read
        + num_recv_tokens * hidden * combine_storage_bytes                             # L2 combine write
    )
    num_shared_hbm_bytes = 0 if num_shared_experts == 0 else (
        shared_intermediate_hidden * 2 * hidden * weight_storage_bytes  # Shared L1 weights + SF
        + hidden * shared_intermediate_hidden * weight_storage_bytes    # Shared L2 weights + SF
        + num_tokens * hidden * act_storage_bytes                       # Shared L1 acts + SF read
        + num_tokens * shared_intermediate_hidden * act_storage_bytes   # Shared L1 output + SF write
        + num_tokens * shared_intermediate_hidden * act_storage_bytes   # Shared L2 acts + SF read
        + num_tokens * hidden * combine_storage_bytes                   # Shared L2 combine write
    )
    # if is_nvfp4xnvfp4:
    #     # Additional hierarchical-NVFP4 traffic not represented by
    #     # `act_storage_bytes`: BF16 stage write/read (4 B/element), FP32 K16
    #     # amax write and two reads (0.75 B/element), activation globals, and
    #     # per-output-channel weight globals.
    #     num_routed_hbm_bytes += (
    #         num_recv_tokens * intermediate_hidden * 4.75
    #         + num_recv_tokens * 12
    #         + num_touched_experts * (intermediate_hidden * 2 + hidden) * 4
    #     )
    #     if num_shared_experts > 0:
    #         num_shared_hbm_bytes += (
    #             num_tokens * shared_intermediate_hidden * 4.75
    #             + num_tokens * 12
    #             + (shared_intermediate_hidden * 2 + hidden) * 4
    #         )
    num_hbm_bytes = num_routed_hbm_bytes + num_shared_hbm_bytes

    # NVLink bytes: packed dispatch data + SF (+ NVFP4 token global), then
    # BF16 combine write-back.
    num_nvlink_bytes = (
        num_recv_tokens * hidden * (
            act_storage_bytes + combine_storage_bytes)
        + (num_recv_tokens * 4 if is_nvfp4xnvfp4 else 0)
    )

    # Combine reduction (serial) time approximation
    num_combine_slots = num_topk + (1 if num_shared_experts > 0 else 0)
    t_reduction = num_tokens * hidden * (
        2.0 + num_combine_slots * combine_storage_bytes) / 6.5e12

    # Summary
    def print_perf(elapsed: float, ref_time: float, ref_label: str):
        tflops = safe_div(num_total_flops / 1e12, elapsed)
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, elapsed)
        nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, elapsed)
        approx_factor = safe_div(elapsed, elapsed - t_reduction)
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | '
                   f'{tflops:4.0f} TFLOPS | '
                   f'overlap: '
                   f'{tflops * approx_factor:4.0f} TFLOPS, '
                   f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
                   f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
                   f'{elapsed * 1e6:4.0f} us, '
                   f'reduction: {t_reduction * 1e6:4.1f} us | '
                   f'{safe_div(ref_time, elapsed):.2f}x {ref_label}')

    dist_print(f'Performance (w/{"" if num_shared_experts else "o"} shared):', once_in_node=True)
    print_perf(t_fused, t_baseline, f'legacy{"+shared" if num_shared_experts else ""}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if can_run_legacy_baseline else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--num-shared-experts', type=int, default=1, help='Number of shared experts (use 0 to disable)')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument(
        '--mma-type', type=str, default='fp8xfp4',
        choices=('fp8xfp4', 'nvfp4xnvfp4', 'bf16xbf16'),
        help='MMA type')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
