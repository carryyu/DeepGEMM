#pragma once

#include <functional>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_nvfp4_nvfp4_mega_moe.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    return block_m;
}

using MegaMoEBufferSlices = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static std::tuple<int64_t, std::function<MegaMoEBufferSlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = device_runtime->get_num_sms();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

    // Shared
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Iterate all block candidates to get the maximum ring size
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto with_sf = is_mma_with_sf(mma_kind);
    const auto mma_elem_bits = get_element_bits(mma_kind);
    const auto sf_gran_k = get_sf_gran_k(mma_kind);
    const auto with_nvfp4_global_sf = mma_kind == MmaKind::NVFP4NVFP4;

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }

    // All buffers
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
        num_shared_experts, mma_elem_bits, sf_gran_k, with_nvfp4_global_sf
    );

    // Check SF buffer requirements
    if (with_sf) {
        const auto packed_sf_gran_k = sf_gran_k * 4;
        DG_HOST_ASSERT(hidden % packed_sf_gran_k == 0 and
                       intermediate_hidden % packed_sf_gran_k == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % packed_sf_gran_k == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        const auto is_nvfp4 = mma_kind == MmaKind::NVFP4NVFP4;
        const auto acts_dtype = mma_kind == MmaKind::BF16 ? torch::kBFloat16 :
                                is_nvfp4 ? kPackedFP4 : torch::kFloat8_e4m3fn;
        const auto hidden_storage = is_nvfp4 ? hidden / 2 : hidden;
        const auto intermediate_hidden_storage =
            is_nvfp4 ? intermediate_hidden / 2 : intermediate_hidden;
        const auto shared_intermediate_hidden_storage =
            is_nvfp4 ? shared_intermediate_hidden / 2 : shared_intermediate_hidden;
        const auto hidden_packed_sf = with_sf ? hidden / (sf_gran_k * 4) : 0;
        const auto intermediate_hidden_packed_sf =
            with_sf ? intermediate_hidden / (sf_gran_k * 4) : 0;
        const auto shared_intermediate_hidden_packed_sf =
            with_sf ? shared_intermediate_hidden / (sf_gran_k * 4) : 0;

        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden_storage},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden_packed_sf},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto x_global_sf = is_nvfp4 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_global_sf_buffer.base)),
            {num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), hidden_packed_sf},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden_storage},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), shared_intermediate_hidden_packed_sf},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_staging = (is_nvfp4 and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_staging_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden_storage},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden_packed_sf},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden_storage},
            torch::TensorOptions().dtype(acts_dtype).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden_packed_sf},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_staging = is_nvfp4 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_staging_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(
            x, x_sf, x_global_sf, topk_idx, topk_weights,
            shared_l1_acts, shared_l1_acts_sf,
            shared_l2_acts, shared_l2_acts_sf, shared_l2_staging,
            l1_acts, l1_acts_sf,
            l2_acts, l2_acts_sf, l2_staging);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, _x_global_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf,
                shared_l2_acts, shared_l2_acts_sf, _shared_l2_staging,
                l1_acts, l1_acts_sf,
                l2_acts, l2_acts_sf, _l2_staging] =
        slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               shared_l1_acts, shared_l1_acts_sf,
                               shared_l2_acts, shared_l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               shared_l1_weights, shared_l2_weights,
                               shared_l1_weights_sf, shared_l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void nvfp4_nvfp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf, l1_weights_global_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf, l2_weights_global_sf] = l2_weights_tuple;

    // NVFP4 uses E2M1 data with one packed positive UE4M3 scale per
    // 16 logical K values.
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 16);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() ==
                   shared_l2_weights_tuple_opt.has_value());

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Routed weights are packed FP4, contiguous, and K-major.
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(hidden % 256 == 0 and intermediate_hidden % 256 == 0);

    const auto [num_tokens, output_hidden] = get_shape<2>(y);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(output_hidden == hidden);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());

    constexpr int kGranMN = 1, kGranK = 16;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    DG_HOST_ASSERT(l1_weights_global_sf.scalar_type() == torch::kFloat32 and
                   l1_weights_global_sf.is_contiguous() and
                   l1_weights_global_sf.dim() == 2 and
                   l1_weights_global_sf.size(0) == num_experts_per_rank and
                   l1_weights_global_sf.size(1) == intermediate_hidden * 2);
    DG_HOST_ASSERT(l2_weights_global_sf.scalar_type() == torch::kFloat32 and
                   l2_weights_global_sf.is_contiguous() and
                   l2_weights_global_sf.dim() == 2 and
                   l2_weights_global_sf.size(0) == num_experts_per_rank and
                   l2_weights_global_sf.size(1) == hidden);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l1_weights_global_sf;
    torch::Tensor shared_l2_weights, shared_l2_weights_sf, shared_l2_weights_global_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf, shared_l1_weights_global_sf) =
            shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf, shared_l2_weights_global_sf) =
            shared_l2_weights_tuple_opt.value();

        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        const auto [shared_intermediate_hidden_2, shared_hidden] =
            check_ab_fp8_fp4(shared_l1_weights, cute::UMMA::Major::K, arch_major);
        const auto [shared_hidden_, shared_intermediate_hidden_] =
            check_ab_fp8_fp4(shared_l2_weights, cute::UMMA::Major::K, arch_major);
        shared_intermediate_hidden = shared_intermediate_hidden_;
        DG_HOST_ASSERT(shared_hidden == hidden and shared_hidden_ == hidden);
        DG_HOST_ASSERT(shared_intermediate_hidden_2 == 2 * shared_intermediate_hidden);
        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == kPackedFP4);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == kPackedFP4);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and
                       shared_l2_weights.is_contiguous());

        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden,
                        kGranMN, kGranK, std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden,
                        kGranMN, kGranK, std::nullopt, true, false, torch::kInt);
        DG_HOST_ASSERT(shared_l1_weights_global_sf.scalar_type() == torch::kFloat32 and
                       shared_l1_weights_global_sf.is_contiguous() and
                       shared_l1_weights_global_sf.dim() == 1 and
                       shared_l1_weights_global_sf.numel() == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l2_weights_global_sf.scalar_type() == torch::kFloat32 and
                       shared_l2_weights_global_sf.is_contiguous() and
                       shared_l2_weights_global_sf.dim() == 1 and
                       shared_l2_weights_global_sf.numel() == hidden);
    }

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "nvfp4xnvfp4", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, x_global_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf,
                shared_l2_acts, shared_l2_acts_sf, shared_l2_staging,
                l1_acts, l1_acts_sf,
                l2_acts, l2_acts_sf, l2_staging] =
        slice(sym_buffer);

    if (arch_major == 10) {
        sm100_nvfp4_nvfp4_mega_moe(
            y,
            l1_acts, l1_acts_sf,
            l2_acts, l2_acts_sf, l2_staging,
            shared_l1_acts, shared_l1_acts_sf,
            shared_l2_acts, shared_l2_acts_sf, shared_l2_staging,
            l1_weights, l2_weights,
            l1_weights_sf, l2_weights_sf,
            l1_weights_global_sf, l2_weights_global_sf,
            shared_l1_weights, shared_l2_weights,
            shared_l1_weights_sf, shared_l2_weights_sf,
            shared_l1_weights_global_sf, shared_l2_weights_global_sf,
            cumulative_local_expert_recv_stats,
            sym_buffer_ptrs,
            rank_idx, num_max_tokens_per_rank,
            num_experts_per_rank,
            num_shared_experts,
            num_tokens, num_topk,
            hidden, intermediate_hidden,
            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, _x_global_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf,
                shared_l2_acts, _shared_l2_acts_sf, _shared_l2_staging,
                l1_acts, _l1_acts_sf,
                l2_acts, _l2_acts_sf, _l2_staging] =
        slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_block_m_for_mega_moe", &get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("nvfp4_nvfp4_mega_moe", &nvfp4_nvfp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
#endif
}

} // namespace deep_gemm::mega
