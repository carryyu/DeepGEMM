#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;
    bool with_nvfp4_global_sf;

    // Keep grid/NVLink/schedule counters separated from expert counters.
    // NVIDIA L2 cache lines are 128B, and these counters are hot atomics.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens,
              const bool& with_nvfp4_global_sf = false):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens),
        with_nvfp4_global_sf(with_nvfp4_global_sf) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel task scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Shared L2 full block count
        num_bytes += num_shared_l2_pool_blocks * sizeof(uint32_t);

        if (with_nvfp4_global_sf) {
            // L1 BF16 staging completion count (ring).
            num_bytes += num_ring_blocks * sizeof(uint32_t);

            // Shared-L1 BF16 staging completion count.
            num_bytes += num_shared_l2_pool_blocks * sizeof(uint32_t);
        }

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..31]: `uint32_t` L1 schedule task counter
    // [32..35]: `uint32_t` L2 schedule task counter
    // [36..39]: `uint32_t` shared L1 schedule task counter
    // [40..43]: `uint32_t` shared L2 schedule task counter
    // [44..127]: padding to isolate hot expert counters from barrier/schedule counters
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 40u);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_stage_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        DG_DEVICE_ASSERT(with_nvfp4_global_sf);
        const auto base = get_shared_l2_full_count_ptr(num_shared_l2_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_stage_full_count_ptr(const uint32_t& block_idx = 0) const {
        DG_DEVICE_ASSERT(with_nvfp4_global_sf);
        const auto base = get_l1_stage_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = with_nvfp4_global_sf ?
            get_shared_l1_stage_full_count_ptr(num_shared_l2_pool_blocks) :
            get_shared_l2_full_count_ptr(num_shared_l2_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_global_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l1_global_sf_buffer,
           shared_l2_staging_buffer, shared_l2_amax_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer, shared_l2_global_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_global_sf_buffer,
           l1_topk_weights_buffer,
           l2_staging_buffer,
           l2_amax_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           l2_global_sf_buffer,
           combine_token_buffer;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0,
                  uint32_t mma_elem_bits = 0,
                  const uint32_t& sf_gran_k = 32,
                  const bool& with_nvfp4_global_sf = false) {
        DG_UNIFIED_ASSERT(not with_nvfp4_global_sf or
                          (with_sf and mma_elem_bits == 4 and sf_gran_k == 16));
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens,
                              with_nvfp4_global_sf);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        // Keep the historical inference for existing callers: BF16 uses 16
        // bits and the scaled W4A8 path uses byte-wide activation storage.
        if (mma_elem_bits == 0)
            mma_elem_bits = with_sf ? 8u : 16u;
        DG_UNIFIED_ASSERT(mma_elem_bits == 4 or mma_elem_bits == 8 or mma_elem_bits == 16);
        DG_UNIFIED_ASSERT((hidden * mma_elem_bits) % 8 == 0);
        DG_UNIFIED_ASSERT((intermediate_hidden * mma_elem_bits) % 8 == 0);
        DG_UNIFIED_ASSERT((shared_intermediate_hidden * mma_elem_bits) % 8 == 0);
        if (with_sf) {
            DG_UNIFIED_ASSERT(sf_gran_k > 0);
            // Four one-byte SF values are packed into each exposed int32.
            DG_UNIFIED_ASSERT(hidden % (sf_gran_k * 4) == 0);
            DG_UNIFIED_ASSERT(intermediate_hidden % (sf_gran_k * 4) == 0);
            DG_UNIFIED_ASSERT(shared_intermediate_hidden % (sf_gran_k * 4) == 0);
        }

        const auto input_token_layout = layout::Data(hidden * mma_elem_bits / 8);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * mma_elem_bits / 8);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * mma_elem_bits / 8);
        const auto input_sf_layout = layout::Data(with_sf ? hidden / sf_gran_k : 0);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / sf_gran_k : 0);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / sf_gran_k : 0);
        const auto global_sf_layout = layout::Data(
            with_nvfp4_global_sf ? sizeof(float) : 0, false);
        const auto intermediate_bf16_staging_layout = layout::Data(
            with_nvfp4_global_sf ? intermediate_hidden * sizeof(nv_bfloat16) : 0);
        const auto shared_intermediate_bf16_staging_layout = layout::Data(
            with_nvfp4_global_sf ? shared_intermediate_hidden * sizeof(nv_bfloat16) : 0);
        // L1 persists one row maximum per K64 output tile. Exact K16
        // maxima are recomputed from BF16 during full-row quantization.
        constexpr uint32_t kNVFP4AmaxGranK = 64;
        const auto intermediate_amax_layout = layout::Data(
            with_nvfp4_global_sf ?
                intermediate_hidden / kNVFP4AmaxGranK * sizeof(float) : 0);
        const auto shared_intermediate_amax_layout = layout::Data(
            with_nvfp4_global_sf ?
                shared_intermediate_hidden / kNVFP4AmaxGranK * sizeof(float) : 0);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);
        // Packed FP4 tensor maps require a 32-byte global base.
        const auto workspace_bytes = workspace.get_num_bytes();
        const auto input_buffer_offset = mma_elem_bits == 4 ?
            math::align<uint64_t>(workspace_bytes, 32) : workspace_bytes;
        const auto input_buffer_base = math::advance_ptr(base, input_buffer_offset);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            input_buffer_base);
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_global_sf_buffer = Buffer(
            global_sf_layout, 1, num_max_tokens_per_rank,
            input_sf_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            input_global_sf_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_global_sf_buffer = input_global_sf_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_staging_buffer = Buffer(
            shared_intermediate_bf16_staging_layout, 1,
            num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_l1_sf_buffer.get_end_ptr());
        shared_l2_amax_buffer = Buffer(
            shared_intermediate_amax_layout, 1,
            num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_l2_staging_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_l2_amax_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());
        shared_l2_global_sf_buffer = Buffer(
            global_sf_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_l2_sf_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                shared_l2_global_sf_buffer.get_end_ptr() :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_global_sf_buffer = Buffer(
            global_sf_layout, 1, num_ring_tokens,
            l1_sf_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            l1_global_sf_buffer.get_end_ptr());

        l2_staging_buffer = Buffer(
            intermediate_bf16_staging_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_amax_buffer = Buffer(
            intermediate_amax_layout, 1, num_ring_tokens,
            l2_staging_buffer.get_end_ptr());
        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l2_amax_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());
        l2_global_sf_buffer = Buffer(
            global_sf_layout, 1, num_ring_tokens,
            l2_sf_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            l2_global_sf_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - static_cast<uint8_t*>(workspace.base);
    }
};

} // namespace deep_gemm::layout
