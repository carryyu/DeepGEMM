#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100NVFP4NVFP4MegaMoERuntime final : public LaunchRuntime<SM100NVFP4NVFP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_shared_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        const float* l1_weights_global_sf;
        const float* l2_weights_global_sf;
        const float* shared_l1_weights_global_sf;
        const float* shared_l2_weights_global_sf;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_shared_l1_acts;
        CUtensorMap tensor_map_shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        CUtensorMap tensor_map_shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l1_output;
        CUtensorMap tensor_map_shared_l2_acts;
        CUtensorMap tensor_map_shared_l2_acts_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        CUtensorMap tensor_map_shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_nvfp4_nvfp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_nvfp4_nvfp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_shared_experts,
    args.num_topk,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.l1_weights_global_sf,
            args.l2_weights_global_sf,
            args.shared_l1_weights_global_sf,
            args.shared_l2_weights_global_sf,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.tensor_map_shared_l1_acts,
            args.tensor_map_shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.tensor_map_shared_l1_weights_sf,
            args.tensor_map_shared_l1_output,
            args.tensor_map_shared_l2_acts,
            args.tensor_map_shared_l2_acts_sf,
            args.tensor_map_shared_l2_weights,
            args.tensor_map_shared_l2_weights_sf
        ));
    }
};

static void sm100_nvfp4_nvfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l2_staging,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& shared_l2_staging,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& l1_weights_global_sf, const torch::Tensor& l2_weights_global_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const torch::Tensor& shared_l1_weights_global_sf,
    const torch::Tensor& shared_l2_weights_global_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::NVFP4NVFP4);

    constexpr int kGranK = 16;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    DG_HOST_ASSERT(config.block_k == 128 or config.block_k == 256);
    const int packed_a_swizzle_mode = config.swizzle_acts_mode;
    const int packed_b_swizzle_mode = config.swizzle_weights_mode;
    DG_HOST_ASSERT(packed_a_swizzle_mode == config.block_k / 2);
    DG_HOST_ASSERT(packed_b_swizzle_mode == config.block_k / 2);

    // A/B descriptors keep FP4 packed in SMEM.
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts, hidden, config.num_ring_tokens,
        config.block_k, config.load_block_m,
        static_cast<int>(l1_acts.stride(-2)),
        packed_a_swizzle_mode, 0, false, false);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_acts_sf,
        config.num_sf_ring_tokens, hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false, sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, hidden, num_experts_per_rank * intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(l1_weights.stride(-2)),
        packed_b_swizzle_mode, 0, false, false);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_weights_sf,
        intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        num_experts_per_rank, 0, 0, false, sf_smem_outer_dim);

    // Post-SwiGLU values are staged as BF16 and quantized after the complete
    // row's per-token global scale is known.
    constexpr int kL1OutputSwizzleMode = 128;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_staging, intermediate_hidden, config.num_ring_tokens,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(l2_staging.stride(-2)),
        kL1OutputSwizzleMode, 0, false, false);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_ring_tokens,
        config.block_k, config.load_block_m,
        static_cast<int>(l2_acts.stride(-2)),
        packed_a_swizzle_mode, 0, false, false);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_acts_sf,
        config.num_sf_ring_tokens, intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false, sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights, intermediate_hidden, num_experts_per_rank * hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(l2_weights.stride(-2)),
        packed_b_swizzle_mode, 0, false, false);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_weights_sf,
        hidden, intermediate_hidden,
        config.block_n, kGranK,
        num_experts_per_rank, 0, 0, false, sf_smem_outer_dim);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        packed_a_swizzle_mode, 0, false, false) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false, sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        packed_b_swizzle_mode, 0, false, false) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false, sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l2_staging,
            shared_intermediate_hidden, num_max_tokens_per_rank,
            config.block_n / 2, config.store_block_m,
            static_cast<int>(shared_l2_staging.stride(-2)),
            kL1OutputSwizzleMode, 0, false, false) :
        tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        packed_a_swizzle_mode, 0, false, false) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false, sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        packed_b_swizzle_mode, 0, false, false) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false, sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = device_runtime->get_num_sms();
    const SM100NVFP4NVFP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_shared_experts = num_shared_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .l1_weights_global_sf = l1_weights_global_sf.data_ptr<float>(),
        .l2_weights_global_sf = l2_weights_global_sf.data_ptr<float>(),
        .shared_l1_weights_global_sf = num_shared_experts > 0 ?
            shared_l1_weights_global_sf.data_ptr<float>() : nullptr,
        .shared_l2_weights_global_sf = num_shared_experts > 0 ?
            shared_l2_weights_global_sf.data_ptr<float>() : nullptr,
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .tensor_map_shared_l1_acts_sf = tensor_map_shared_l1_acts_sf,
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .tensor_map_shared_l1_weights_sf = tensor_map_shared_l1_weights_sf,
        .tensor_map_shared_l1_output = tensor_map_shared_l1_output,
        .tensor_map_shared_l2_acts = tensor_map_shared_l2_acts,
        .tensor_map_shared_l2_acts_sf = tensor_map_shared_l2_acts_sf,
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .tensor_map_shared_l2_weights_sf = tensor_map_shared_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100NVFP4NVFP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_nvfp4_nvfp4_mega_moe", code);
    SM100NVFP4NVFP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
