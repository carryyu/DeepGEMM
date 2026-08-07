#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_nvfp4_nvfp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const float* l1_weights_global_sf,
                            const float* l2_weights_global_sf,
                            const float* shared_l1_weights_global_sf,
                            const float* shared_l2_weights_global_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts,
        // Host integration extends MegaMoEBuffer with these optional layout arguments.
        /*mma_elem_bits=*/ 4,
        /*sf_gran_k=*/ 16,
        /*with_nvfp4_global_sf=*/ true
    );
    const auto workspace = buffer.workspace;

    // SF and its buffer configs
    constexpr uint32_t kGranK = 16;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // All routed/shared activations and weights stay packed E2M1 in SMEM.
    using a_dtype_t = cutlass::float_e2m1_t;
    using b_dtype_t = cutlass::float_e2m1_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 64;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128 or BLOCK_K == 256, "NVFP4 supports BLOCK_K 128 or 256");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Invalid block K");

    // Swizzle configs
    // One K-major swizzle atom spans the full packed tile: 64B for BK128,
    // 128B for the expected BK256 configuration.
    constexpr uint32_t kSwizzleAMode = BLOCK_K / 2;
    constexpr uint32_t kSwizzleBMode = BLOCK_K / 2;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment)
            nv_bfloat16 l1[kNumEpilogueWarpgroups][kNumTMAStoreStages]
                           [STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        // L1 produces four K16 amax values per post-SwiGLU K64 tile (one per
        // warp). Reduce them in shared memory and persist only the K64 tile
        // max. The exact K16 max is recomputed from staged BF16 values when
        // the complete row is quantized.
        alignas(sizeof(float2))
        float l1_tile_amax[kNumEpilogueWarpgroups][4][STORE_BLOCK_M];
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K / 2];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K / 2];
        // One uint32 packs four UE4M3 scales, covering one logical K64 slab.
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 64)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 64)];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        cutlass::arch::ClusterBarrier quantize_pair_ready_barrier;
        Barrier quantize_owner_barrier;
        cutlass::arch::ClusterBarrier quantize_pair_done_barrier;
        uint32_t quantize_owner;
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFSlabs = BLOCK_K / UMMA_K;
    constexpr uint32_t kNumSFATmemColsPerSlab = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemColsPerSlab = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFATmemCols = kNumSFSlabs * kNumSFATmemColsPerSlab;
    constexpr uint32_t kNumSFBTmemCols = kNumSFSlabs * kNumSFBTmemColsPerSlab;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<
        kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
            // Each CTA already knows its own local work is complete, so its
            // local barrier only waits for the partner CTA's remote arrival.
            shared_storage.quantize_pair_ready_barrier.init(1);
            shared_storage.quantize_owner_barrier.init(1);
            shared_storage.quantize_pair_done_barrier.init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumPackedTokenBytes = kHidden / 2;
            constexpr uint32_t kNumChunks = kNumPackedTokenBytes / kNumBytesPerPull;
            DG_STATIC_ASSERT(kHidden % 2 == 0, "Packed FP4 hidden must be even");
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kNumPackedTokenBytes,
                             "kNumBytesPerPull must divide packed hidden bytes");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / (kGranK * 4);
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % (kGranK * 4) == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                // Activation global scale follows the token through dispatch.
                const auto activation_global_sf = *sym_buffer.map(
                    buffer.input_global_sf_buffer.get_data_buffer(src_token_idx)
                        .template get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                *buffer.l1_global_sf_buffer
                    .get_data_buffer(pool_token_idx % kNumRingTokens)
                    .template get_base_ptr<float>() = activation_global_sf;

                // Load weights
                const auto weight = *sym_buffer.map(
                    buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }
        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx;
                 i < workspace.num_shared_l2_pool_blocks;
                 i += kNumDispatchThreads) {
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
                *workspace.get_shared_l1_stage_full_count_ptr(i) = 0;
            }
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_stage_full_count_ptr(
                        (expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::Linear2) {
                const auto ptr = workspace.get_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks =
                    pool_block_idx / kNumRingBlocks + 1;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                constexpr uint32_t num_expected_blocks = 1;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / (kGranK * 4));

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier.
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr,
                        &shared_storage.full_barriers[stage_idx],
                        shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr,
                        &shared_storage.full_barriers[stage_idx],
                        shared_storage.smem_sfa[stage_idx],
                        sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx]
                            .arrive_and_expect_tx(
                                sizeof(SharedStorage::smem_a[0]) * 2 +
                                sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ?
                    k_block_idx * (BLOCK_K / (kGranK * 4)) :
                    task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / (kGranK * 4));

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                        shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx],
                        shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(SharedStorage::smem_b[0]) * 2 +
                            sizeof(SharedStorage::smem_sfb[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    b_dtype_t, a_dtype_t, float, cutlass::float_ue4m3_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
                >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(
                    shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(
                    shared_storage.smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(SharedStorage::smem_a[0]) / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t sf_slab_idx = 0; sf_slab_idx < kNumSFSlabs; ++ sf_slab_idx) {
                            // Each asynchronous K64 MMA gets a distinct SF TMEM region.
                            const uint32_t tmem_sfa =
                                kTmemStartColOfSFA + sf_slab_idx * kNumSFATmemColsPerSlab;
                            const uint32_t tmem_sfb =
                                kTmemStartColOfSFB + sf_slab_idx * kNumSFBTmemColsPerSlab;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] +
                                    sf_slab_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, tmem_sfa + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] +
                                    sf_slab_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, tmem_sfb + i * 4);
                            }

                            // block16 consumes all four UE4M3 bytes in this K64
                            // slab, so each independent SF region uses sf_id 0.
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                    a_desc_base_lo, 0, sf_slab_idx * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                    b_desc_base_lo, 0, sf_slab_idx * UMMA_K);
                            ptx::SM100_MMA_MXF4NVF4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or sf_slab_idx > 0, runtime_instr_desc,
                                tmem_sfb, tmem_sfa);
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        uint32_t quantize_pair_phase = 0;
        uint32_t quantize_pair_done_phase = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx =
                current_iter_idx % kNumEpilogueStages;
            const auto accum_phase =
                (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[
                accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or
                task_info.block_phase ==
                    sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;
                const auto activation_global_base =
                    task_info.is_shared() ?
                    buffer.shared_l1_global_sf_buffer
                        .get_base_ptr<float>() :
                    buffer.l1_global_sf_buffer
                        .get_base_ptr<float>();
                const auto l1_weight_global_base =
                    task_info.is_shared() ?
                    shared_l1_weights_global_sf :
                    l1_weights_global_sf +
                        task_info.local_expert_idx * L1_SHAPE_N;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    nv_bfloat162 bf16x2_output[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint2 raw_values[4];
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // The MMA accumulator only contains block-scaled
                        // products. Apply activation-token and weight-channel
                        // global scales before the existing BF16 gate/up
                        // rounding.
                        auto fp32_values = reinterpret_cast<float*>(raw_values);
                        const uint32_t row_base =
                            epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M +
                            (lane_idx % 4) * 2;
                        const uint32_t activation_global_m =
                            (task_info.is_shared() ? m_idx : ring_m_idx) + row_base;
                        const float2 activation_global = {
                            activation_global_base[activation_global_m],
                            activation_global_base[activation_global_m + 1]
                        };

                        // Apply SwiGLU: silu(gate) * up
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            const uint32_t post_channel_in_warp =
                                k * 8 + lane_idx / 4;
                            const uint32_t interleave_group =
                                post_channel_in_warp / 8;
                            const uint32_t gate_channel =
                                n_block_idx * BLOCK_N + warp_idx_in_wg * 32 +
                                interleave_group * 16 + post_channel_in_warp % 8;
                            const float gate_global =
                                l1_weight_global_base[gate_channel];
                            const float up_global =
                                l1_weight_global_base[gate_channel + 8];
                            const float2 gate_scaled = __fmul2_rn(
                                make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]),
                                __fmul2_rn(activation_global, {gate_global, gate_global}));
                            const float2 up_scaled = __fmul2_rn(
                                make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]),
                                __fmul2_rn(activation_global, {up_global, up_global}));
                            auto bf16_gate = __float22bfloat162_rn(gate_scaled);
                            auto bf16_up = __float22bfloat162_rn(up_scaled);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            // The staged value is explicitly BF16. K16 amax
                            // below is computed from this rounded value.
                            bf16x2_output[i][k] = __float22bfloat162_rn(
                                __fmul2_rn(__fmul2_rn(gate, up), weights));
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            const auto staged = __bfloat1622float2(bf16x2_output[i][k]);
                            thread_local_amax.x =
                                cute::max(thread_local_amax.x, cute::abs(staged.x));
                            thread_local_amax.y =
                                cute::max(thread_local_amax.y, cute::abs(staged.y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());
                    }

                    // Exchange the four per-warp K16 maxima through shared
                    // memory. The following warpgroup barrier also protects
                    // these stores before warp 0 reduces the K64 tile max.
                    if (lane_idx < 4) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                            const uint32_t row = i * ATOM_M + lane_idx * 2;
                            auto dst = reinterpret_cast<float2*>(
                                &shared_storage.l1_tile_amax
                                    [epilogue_wg_idx][warp_idx_in_wg][row]);
                            *dst = amax_values[i];
                        }
                    }

                    // Wait shared memory release from previous TMA store
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(
                        128,
                        kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t row = lane_idx % 8;
                        const uint32_t col =
                            warp_idx_in_wg * 2 + lane_idx / 8;
                        const auto smem_ptr =
                            shared_storage.smem_d.l1[
                                epilogue_wg_idx][tma_stage_idx]
                            + (i * ATOM_M + row) * L1_OUT_BLOCK_N
                            + (col ^ row) *
                                (kNumBankGroupBytes /
                                 sizeof(nv_bfloat16));
                        ptx::SM90_U32x2_STSM_T<__nv_bfloat162>::copy(
                            bf16x2_output[i][0],
                            bf16x2_output[i][1], smem_ptr);

                        // Warp 0 reduces the four K16 maxima into one K64
                        // tile max per token. This is sufficient for the
                        // complete-row global scale; exact K16 maxima are
                        // recomputed from BF16 in the quantization pass.
                        if (warp_idx_in_wg == 0 and lane_idx < ATOM_M) {
                            const uint32_t token_base_idx =
                                epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M + i * ATOM_M;
                            const uint32_t token_idx =
                                m_idx + token_base_idx + lane_idx;
                            auto amax_base = task_info.is_shared() ?
                                buffer.shared_l2_amax_buffer
                                    .get_base_ptr<float>() :
                                buffer.l2_amax_buffer
                                    .get_base_ptr<float>();
                            const uint32_t shape_k =
                                task_info.is_shared() ?
                                SHARED_L2_SHAPE_K : L2_SHAPE_K;
                            const uint32_t num_amax_tiles =
                                shape_k / L1_OUT_BLOCK_N;
                            const uint32_t row = i * ATOM_M + lane_idx;
                            float tile_amax = 0.0f;
                            #pragma unroll
                            for (uint32_t source_warp = 0;
                                 source_warp < 4; ++ source_warp) {
                                tile_amax = cute::max(
                                    tile_amax,
                                    shared_storage.l1_tile_amax
                                        [epilogue_wg_idx][source_warp][row]);
                            }
                            amax_base[
                                token_idx * num_amax_tiles + n_block_idx] =
                                tile_amax;
                        }
                    }
                    ptx::sync_aligned(
                        128,
                        kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        const uint32_t out_m_idx =
                            m_idx + epilogue_wg_idx * WG_BLOCK_M +
                            s * STORE_BLOCK_M;
                        if constexpr (BLOCK_M >= 64) {
                            ptx::tma_store_2d(
                                tensor_map_l1_output_ptr,
                                shared_storage.smem_d.l1[
                                    epilogue_wg_idx][tma_stage_idx],
                                out_n_idx, out_m_idx,
                                cute::TMA::CacheHintSm90::EVICT_LAST);
                        } else if (task_info.is_shared()) {
                            // Shared-expert staging is consumed as one
                            // concatenated row, so retaining all contributing
                            // L1 stores helps even for decoding BLOCK_M.
                            ptx::tma_store_2d(
                                tensor_map_l1_output_ptr,
                                shared_storage.smem_d.l1[
                                    epilogue_wg_idx][tma_stage_idx],
                                out_n_idx, out_m_idx,
                                cute::TMA::CacheHintSm90::EVICT_LAST);
                        } else {
                            ptx::tma_store_2d(
                                tensor_map_l1_output_ptr,
                                shared_storage.smem_d.l1[
                                    epilogue_wg_idx][tma_stage_idx],
                                out_n_idx, out_m_idx,
                                cute::TMA::CacheHintSm90::EVICT_LAST);
                        }
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Publish BF16 staging completion as one CTA-pair. Both CTAs
                // must finish their adjacent N blocks before the leader adds
                // two to the row's completion count. The pair that completes
                // the full intermediate row cooperatively quantizes it.
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                // All L1 consumers have now finished reading the routed input
                // slot. Release it before the full-row quantization so that
                // dispatch can refill the ring while the epilogue warps keep
                // working on the L2 input.
                if (not task_info.is_shared() and
                    epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    ptx::red_add(
                        workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                }
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    shared_storage.quantize_pair_ready_barrier.arrive(
                        is_leader_cta ? 1u : 0u, 1u);
                }
                shared_storage.quantize_pair_ready_barrier.wait(
                    quantize_pair_phase);

                if (is_leader_cta and epilogue_warp_idx == 0 and
                    cute::elect_one_sync()) {
                    const uint32_t num_l1_n_blocks = task_info.is_shared() ?
                        (kNumSharedExperts * L1_SHAPE_N) / BLOCK_N :
                        L1_SHAPE_N / BLOCK_N;
                    const uint32_t generation = task_info.is_shared() ?
                        0u : pool_block_idx / kNumRingBlocks;
                    auto stage_count_ptr = task_info.is_shared() ?
                        workspace.get_shared_l1_stage_full_count_ptr(pool_block_idx) :
                        workspace.get_l1_stage_full_count_ptr(ring_block_idx);
                    const uint32_t old_count =
                        ptx::atomic_add_acq_rel(stage_count_ptr, 2u);
                    const uint32_t quantize_owner =
                        old_count + 2 == num_l1_n_blocks * (generation + 1);

                    shared_storage.quantize_owner = quantize_owner;
                    const uint32_t owner_addr =
                        cute::cast_smem_ptr_to_uint(
                            &shared_storage.quantize_owner);
                    const uint32_t owner_barrier_addr =
                        cute::cast_smem_ptr_to_uint(
                            &shared_storage.quantize_owner_barrier);
                    shared_storage.quantize_owner_barrier
                        .arrive_and_expect_tx(sizeof(uint32_t), 1u);
                    cute::store_shared_remote(
                        quantize_owner, owner_addr,
                        owner_barrier_addr, 1u);
                }
                if (is_leader_cta) {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    shared_storage.quantize_owner_barrier.wait(
                        quantize_pair_phase);
                }
                quantize_pair_phase ^= 1;

                if (shared_storage.quantize_owner) {

                    const uint32_t shape_k = task_info.is_shared() ?
                        SHARED_L2_SHAPE_K : L2_SHAPE_K;
                    const uint32_t num_k16 = shape_k / kGranK;
                    const uint32_t num_amax_tiles =
                        shape_k / L1_OUT_BLOCK_N;
                    auto staging_base = task_info.is_shared() ?
                        buffer.shared_l2_staging_buffer.get_base_ptr<nv_bfloat16>() :
                        buffer.l2_staging_buffer.get_base_ptr<nv_bfloat16>();
                    auto amax_base = task_info.is_shared() ?
                        buffer.shared_l2_amax_buffer.get_base_ptr<float>() :
                        buffer.l2_amax_buffer.get_base_ptr<float>();
                    auto global_sf_base = task_info.is_shared() ?
                        buffer.shared_l2_global_sf_buffer.get_base_ptr<float>() :
                        buffer.l2_global_sf_buffer.get_base_ptr<float>();
                    auto packed_base = task_info.is_shared() ?
                        buffer.shared_l2_token_buffer.get_base_ptr<uint8_t>() :
                        buffer.l2_token_buffer.get_base_ptr<uint8_t>();
                    auto sf_base = task_info.is_shared() ?
                        buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() :
                        buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                    const uint32_t sf_num_tokens = task_info.is_shared() ?
                        kNumSharedSFTokens : kNumSFRingTokens;

                    // Use a power-of-two subgroup for every valid row. The
                    // former row-per-thread mapping left almost the entire CTA
                    // idle for decoding shapes (e.g. 8 valid rows with 256
                    // epilogue threads), and each active lane performed all
                    // K16 loads and conversions serially.
                    uint32_t quant_threads_per_row = 1;
                    if (valid_m * 32 <= kNumEpilogueThreads)
                        quant_threads_per_row = 32;
                    else if (valid_m * 16 <= kNumEpilogueThreads)
                        quant_threads_per_row = 16;
                    else if (valid_m * 8 <= kNumEpilogueThreads)
                        quant_threads_per_row = 8;
                    else if (valid_m * 4 <= kNumEpilogueThreads)
                        quant_threads_per_row = 4;
                    else if (valid_m * 2 <= kNumEpilogueThreads)
                        quant_threads_per_row = 2;
                    const uint32_t quant_group_shift =
                        31u - static_cast<uint32_t>(
                            __clz(quant_threads_per_row));
                    const uint32_t quant_lane_idx =
                        epilogue_thread_idx & (quant_threads_per_row - 1);
                    const uint32_t row =
                        epilogue_thread_idx >> quant_group_shift;
                    const bool is_valid_quant_row = row < valid_m;
                    const uint32_t physical_token_idx = m_idx + row;

                    // All warp lanes participate in the subgroup reduction,
                    // including lanes assigned to padding rows. XOR shuffles
                    // stay within the aligned power-of-two row subgroup.
                    float thread_row_amax = 0.0f;
                    if (is_valid_quant_row) {
                        #pragma unroll 4
                        for (uint32_t tile_idx = quant_lane_idx;
                             tile_idx < num_amax_tiles;
                             tile_idx += quant_threads_per_row) {
                            thread_row_amax = cute::max(
                                thread_row_amax,
                                amax_base[
                                    physical_token_idx * num_amax_tiles +
                                    tile_idx]);
                        }
                    }
                    float row_amax = thread_row_amax;
                    if (quant_threads_per_row == 32)
                        row_amax = math::warp_reduce<32, false>(
                            row_amax, math::ReduceMax<float>());
                    else if (quant_threads_per_row == 16)
                        row_amax = math::warp_reduce<16, false>(
                            row_amax, math::ReduceMax<float>());
                    else if (quant_threads_per_row == 8)
                        row_amax = math::warp_reduce<8, false>(
                            row_amax, math::ReduceMax<float>());
                    else if (quant_threads_per_row == 4)
                        row_amax = math::warp_reduce<4, false>(
                            row_amax, math::ReduceMax<float>());
                    else if (quant_threads_per_row == 2)
                        row_amax = math::warp_reduce<2, false>(
                            row_amax, math::ReduceMax<float>());

                    float global_sf_inv = 1.0f;
                    if (is_valid_quant_row and quant_lane_idx == 0) {
                        const float global_sf =
                            math::get_nvfp4_global_sf(row_amax);
                        global_sf_inv = kFastMath ?
                            math::fast_rcp(global_sf) : 1.0f / global_sf;
                        if (is_leader_cta)
                            global_sf_base[physical_token_idx] = global_sf;
                    }
                    global_sf_inv = ptx::exchange(
                        global_sf_inv,
                        lane_idx & ~(quant_threads_per_row - 1));

                    if (is_valid_quant_row) {
                        auto staged_row =
                            staging_base + physical_token_idx * shape_k;
                        auto packed_row =
                            packed_base + physical_token_idx * (shape_k / 2);
                        const uint32_t sf_token_idx =
                            block_idx * SF_BLOCK_M + transform_sf_token_idx(row);
                        // Both CTAs compute the inexpensive row amax so they
                        // have the same global scale in registers. Split the
                        // conversion-heavy K16 pass between the pair.
                        const uint32_t quant_k16_offset =
                            is_leader_cta ? 0u : quant_threads_per_row;
                        const uint32_t quant_k16_stride =
                            quant_threads_per_row * 2;
                        uint32_t k16_idx =
                            quant_lane_idx + quant_k16_offset;
                        if (k16_idx < num_k16) {
                            auto staged_vectors =
                                reinterpret_cast<const uint4*>(
                                    staged_row + k16_idx * kGranK);
                            uint4 current_staged[2] = {
                                staged_vectors[0], staged_vectors[1]};

                            // Software-pipeline two K16 blocks per lane.
                            // Issue the next pair of aligned 128-bit staging
                            // loads before converting the current block.
                            #pragma unroll 2
                            while (k16_idx < num_k16) {
                                const uint32_t next_k16_idx =
                                    k16_idx + quant_k16_stride;
                                uint4 next_staged[2] = {};
                                if (next_k16_idx < num_k16) {
                                    const auto next_vectors =
                                        reinterpret_cast<const uint4*>(
                                            staged_row +
                                            next_k16_idx * kGranK);
                                    next_staged[0] = next_vectors[0];
                                    next_staged[1] = next_vectors[1];
                                }
                                const auto staged_pairs =
                                    reinterpret_cast<const nv_bfloat162*>(
                                        current_staged);
                                nv_bfloat162 bf16_amax = {0.0f, 0.0f};
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < kGranK / 2; ++ k) {
                                    bf16_amax = __hmax2(
                                        bf16_amax, __habs2(staged_pairs[k]));
                                }
                                const auto fp32_amax =
                                    __bfloat1622float2(bf16_amax);
                                const float current_block_amax = cute::max(
                                    fp32_amax.x, fp32_amax.y);

                                float dequant_inv;
                                const uint8_t raw_sf =
                                    math::get_nvfp4_sf_and_sf_inv<kFastMath>(
                                        current_block_amax,
                                        global_sf_inv, dequant_inv);
                                using fp4_converter_t =
                                    cutlass::NumericArrayConverter<
                                        cutlass::float_e2m1_t, float, 4,
                                        cutlass::FloatRoundStyle::
                                            round_to_nearest_satfinite>;
                                uint64_t packed_fp4 = 0;
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < kGranK / 2; k += 2) {
                                    cutlass::Array<float, 4> scaled;
                                    const auto values_0 =
                                        __bfloat1622float2(staged_pairs[k]);
                                    const auto values_1 =
                                        __bfloat1622float2(
                                            staged_pairs[k + 1]);
                                    const auto scaled_0 = __fmul2_rn(
                                        values_0,
                                        {dequant_inv, dequant_inv});
                                    const auto scaled_1 = __fmul2_rn(
                                        values_1,
                                        {dequant_inv, dequant_inv});
                                    scaled[0] = scaled_0.x;
                                    scaled[1] = scaled_0.y;
                                    scaled[2] = scaled_1.x;
                                    scaled[3] = scaled_1.y;
                                    const auto quantized =
                                        fp4_converter_t::convert(scaled);
                                    DG_STATIC_ASSERT(
                                        sizeof(quantized) == sizeof(uint16_t),
                                        "Four E2M1 values must occupy two bytes");
                                    const uint16_t packed_chunk =
                                        *reinterpret_cast<const uint16_t*>(
                                            &quantized);
                                    packed_fp4 |=
                                        static_cast<uint64_t>(packed_chunk)
                                            << (k * 8);
                                }
                                *reinterpret_cast<uint64_t*>(
                                    packed_row +
                                    k16_idx * (kGranK / 2)) = packed_fp4;

                                const uint32_t k_uint_idx = k16_idx / 4;
                                const uint32_t byte_idx = k16_idx % 4;
                                const uint32_t sf_addr =
                                    k_uint_idx * sf_num_tokens *
                                        sizeof(uint32_t) +
                                    sf_token_idx * sizeof(uint32_t);
                                if (quant_threads_per_row >= 4) {
                                    // Four adjacent lanes own the four UE4M3
                                    // bytes in one K64 group. A two-stage XOR
                                    // butterfly packs them with two shuffles
                                    // instead of gathering all four lanes.
                                    const uint32_t active_mask =
                                        __activemask();
                                    uint32_t packed_sf =
                                        static_cast<uint32_t>(raw_sf);
                                    const uint32_t peer_byte =
                                        __shfl_xor_sync(
                                            active_mask, packed_sf, 1);
                                    packed_sf = (lane_idx & 1u) ?
                                        peer_byte | (packed_sf << 8) :
                                        packed_sf | (peer_byte << 8);
                                    const uint32_t peer_pair =
                                        __shfl_xor_sync(
                                            active_mask, packed_sf, 2);
                                    packed_sf = (lane_idx & 2u) ?
                                        peer_pair | (packed_sf << 16) :
                                        packed_sf | (peer_pair << 16);
                                    if ((quant_lane_idx & 3u) == 0)
                                        *reinterpret_cast<uint32_t*>(
                                            sf_base + sf_addr) = packed_sf;
                                } else {
                                    sf_base[sf_addr + byte_idx] = raw_sf;
                                }

                                k16_idx = next_k16_idx;
                                current_staged[0] = next_staged[0];
                                current_staged[1] = next_staged[1];
                            }
                        }
                    }

                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        shared_storage.quantize_pair_done_barrier.arrive(
                            is_leader_cta ? 1u : 0u, 1u);
                    }
                    shared_storage.quantize_pair_done_barrier.wait(
                        quantize_pair_done_phase);
                    quantize_pair_done_phase ^= 1;
                    // One release publishes both CTAs' disjoint packed-data
                    // and scale stores to the dependent L2 task.
                    if (is_leader_cta and epilogue_warp_idx == 0 and
                        cute::elect_one_sync()) {
                        if (task_info.is_shared()) {
                            ptx::red_add_rel(
                                workspace.get_shared_l2_full_count_ptr(
                                    pool_block_idx), 1u);
                        } else {
                            ptx::red_add_rel(
                                workspace.get_l2_full_count_ptr(
                                    ring_block_idx), 1u);
                        }
                    }
                }

                __syncwarp();
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;
                const auto activation_global_base =
                    task_info.is_shared() ?
                    buffer.shared_l2_global_sf_buffer
                        .get_base_ptr<float>() :
                    buffer.l2_global_sf_buffer
                        .get_base_ptr<float>();
                const auto l2_weight_global_base =
                    task_info.is_shared() ?
                    shared_l2_weights_global_sf :
                    l2_weights_global_sf +
                        task_info.local_expert_idx * L2_SHAPE_N;
                const uint32_t output_channel_base =
                    n_idx + warp_idx_in_wg * 32;

                    // Default L2 BF16 epilogue.
                    #pragma unroll
                    for (uint32_t s = 0;
                         s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                        if (epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M >= valid_m) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage
                                .tmem_empty_barriers[accum_stage_idx]
                                .arrive(0u);
                            break;
                        }

                        #pragma unroll
                        for (uint32_t i = 0;
                             i < STORE_BLOCK_M / ATOM_M; ++ i) {
                            const uint32_t tmem_addr =
                                accum_stage_idx * UMMA_N +
                                epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M + i * ATOM_M;
                            uint32_t values[ATOM_M];
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                                tmem_addr,
                                values[0], values[1],
                                values[2], values[3]);
                            cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                                tmem_addr | 0x00100000,
                                values[4], values[5],
                                values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();

                            const uint32_t activation_global_m_base =
                                m_idx +
                                epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M + i * ATOM_M +
                                (lane_idx % 4) * 2;
                            const float2 activation_global = {
                                activation_global_base[
                                    activation_global_m_base],
                                activation_global_base[
                                    activation_global_m_base + 1]
                            };
                            auto fp32_values =
                                reinterpret_cast<float*>(values);
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                const float weight_global =
                                    l2_weight_global_base[
                                        output_channel_base +
                                        k * 8 + lane_idx / 4];
                                fp32_values[k * 2] *=
                                    activation_global.x * weight_global;
                                fp32_values[k * 2 + 1] *=
                                    activation_global.y * weight_global;
                            }

                            if (i == 0 and s > 0) {
                                ptx::sync_aligned(
                                    128,
                                    kEpilogueWGBarrierStartIdx +
                                        epilogue_wg_idx);
                            }

                            if (s ==
                                    WG_BLOCK_M / STORE_BLOCK_M - 1 and
                                i ==
                                    STORE_BLOCK_M / ATOM_M - 1) {
                                ptx::tcgen05_before_thread_sync();
                                shared_storage
                                    .tmem_empty_barriers[
                                        accum_stage_idx]
                                    .arrive(0u);
                            }

                            const uint32_t row = lane_idx % 8;
                            const uint32_t col =
                                (epilogue_warp_idx % 2) * 4 +
                                lane_idx / 8;
                            const auto smem_ptr =
                                reinterpret_cast<uint8_t*>(
                                    shared_storage.smem_d.l2[
                                        epilogue_wg_idx]) +
                                (warp_idx_in_wg / 2) *
                                    STORE_BLOCK_M * kSwizzleCDMode +
                                i * ATOM_M * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) +
                                (col ^ row) * kNumBankGroupBytes;
                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                math::cast_into_bf16_and_pack(
                                    values[0], values[1]),
                                math::cast_into_bf16_and_pack(
                                    values[2], values[3]),
                                math::cast_into_bf16_and_pack(
                                    values[4], values[5]),
                                math::cast_into_bf16_and_pack(
                                    values[6], values[7]),
                                smem_ptr);
                        }

                        ptx::sync_aligned(
                            128,
                            kEpilogueWGBarrierStartIdx +
                                epilogue_wg_idx);

                        const uint32_t row_in_atom =
                            (warp_idx_in_wg * 2 +
                             lane_idx / 16) % ATOM_M;
                        const uint32_t bank_group_idx =
                            lane_idx % 8;
                        #pragma unroll
                        for (uint32_t j = 0;
                             j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_store =
                                j * 8 + warp_idx_in_wg * 2 +
                                lane_idx / 16;
                            const uint32_t m_idx_in_block =
                                epilogue_wg_idx * WG_BLOCK_M +
                                s * STORE_BLOCK_M + row_in_store;
                            if (m_idx_in_block >= valid_m)
                                break;

                            uint32_t dst_rank_idx;
                            uint32_t dst_token_idx;
                            uint32_t dst_topk_idx;
                            if (task_info.is_shared()) {
                                dst_rank_idx = sym_buffer.rank_idx;
                                dst_token_idx =
                                    pool_m_idx + m_idx_in_block;
                                dst_topk_idx = kNumTopk;
                            } else {
                                const auto src_metadata =
                                    *workspace
                                        .get_token_src_metadata_ptr(
                                            pool_m_idx +
                                            m_idx_in_block);
                                dst_rank_idx =
                                    src_metadata.rank_idx;
                                dst_token_idx =
                                    src_metadata.token_idx;
                                dst_topk_idx =
                                    src_metadata.topk_idx;
                            }

                            const auto smem_ptr =
                                reinterpret_cast<uint8_t*>(
                                    shared_storage.smem_d.l2[
                                        epilogue_wg_idx]) +
                                (lane_idx % 16 / 8) *
                                    STORE_BLOCK_M *
                                    kSwizzleCDMode +
                                row_in_store * kSwizzleCDMode +
                                (bank_group_idx ^ row_in_atom) *
                                    kNumBankGroupBytes;
                            const auto packed = ptx::ld_shared(
                                reinterpret_cast<float4*>(smem_ptr));

                            const auto dst_token =
                                buffer.combine_token_buffer
                                    .get_rank_buffer(dst_topk_idx)
                                    .get_data_buffer(dst_token_idx);
                            const auto dst_ptr =
                                math::advance_ptr<float4>(
                                    dst_token.get_base_ptr(),
                                    n_idx *
                                        static_cast<uint32_t>(
                                            sizeof(nv_bfloat16)) +
                                    (lane_idx % 16) *
                                        static_cast<uint32_t>(
                                            sizeof(float4)));
                            *sym_buffer.map(
                                dst_ptr, dst_rank_idx) = packed;
                        }
                    }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
        // Grid sync + cross-rank signal + grid sync: ~4 us.
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumEpilogueThreads,
            kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() {
                ptx::sync_aligned(
                    kNumEpilogueThreads,
                    kEpilogueFullBarrierIdx);
            });
        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Default BF16 combine.
        constexpr uint32_t kNumHiddenBytes =
            kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 =
            sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        constexpr bool kCanUseFourChunks =
            kHidden % 4 == 0 and
            (kNumHiddenBytes / 4) % (sizeof(uint4) * 32) == 0;
        constexpr bool kCanUseTwoChunks =
            kHidden % 2 == 0 and
            (kNumHiddenBytes / 2) % (sizeof(uint4) * 32) == 0;
        constexpr uint32_t kNumChunks =
            kCanUseFourChunks ? 4 : (kCanUseTwoChunks ? 2 : 1);
        constexpr uint32_t kNumChunkBytes =
            kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 =
            kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane =
            kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(
            kHidden % kNumChunks == 0,
            "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(
            kNumChunkSlots * kNumEpilogueWarps *
                kNumHiddenBytes / kNumChunks <=
                kNumReusableSmemBytes,
            "Hidden is too large");
        DG_STATIC_ASSERT(
            kHidden / kNumChunks <=
                32 * kNumMaxRegistersForBuffer,
            "Combine chunk register footprint is too large");
        DG_STATIC_ASSERT(
            kNumChunkBytes % sizeof(uint4) == 0,
            "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(
            kNumChunkUint4 % 32 == 0,
            "Combine chunk must have one vector per lane");

        const auto combine_load_buffer =
            utils::PatternVisitor([&](const uint32_t& i) {
                return math::advance_ptr<uint4>(
                    smem_buffer,
                    (epilogue_warp_idx +
                     i * kNumEpilogueWarps) *
                        kNumChunkBytes);
            });
        const auto combine_store_buffer =
            math::advance_ptr<uint4>(
                smem_buffer,
                (epilogue_warp_idx +
                 kNumEpilogueWarps * 2) *
                    kNumChunkBytes);
        auto combine_load_barriers =
            utils::PatternVisitor([&](const uint32_t& i) {
                return &shared_storage.combine_barriers[
                    i + epilogue_warp_idx * 2];
            });

        uint32_t warps_per_token = 1;
        if (kNumChunks >= 2 and num_tokens < kNumSMs)
            warps_per_token = 2;
        if (kNumChunks >= 4 and
            num_tokens * warps_per_token < kNumSMs)
            warps_per_token = 4;

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        const uint32_t num_combine_items =
            num_tokens * warps_per_token;
        const uint32_t global_combine_warp_idx =
            epilogue_warp_idx * kNumSMs + sm_idx;
        for (uint32_t combine_item_idx =
                 global_combine_warp_idx;
             combine_item_idx < num_combine_items;
             combine_item_idx +=
                 kNumSMs * kNumEpilogueWarps) {
            const uint32_t token_idx =
                combine_item_idx / warps_per_token;
            const uint32_t first_chunk =
                combine_item_idx % warps_per_token;
            const int stored_topk_slot_idx =
                lane_idx < kNumTopk ?
                static_cast<int>(__ldg(
                    buffer.input_topk_idx_buffer
                        .get_base_ptr<int64_t>() +
                    token_idx * kNumTopk + lane_idx)) :
                (kNumSharedExperts > 0 and
                         lane_idx == kNumTopk ?
                     static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(
                0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = first_chunk;
                 chunk < kNumChunks;
                 chunk += warps_per_token) {
                const uint32_t chunk_byte_offset =
                    chunk * kNumChunkBytes;
                uint32_t mask = total_mask;
                const auto move_mask_and_load =
                    [&](const uint32_t& i) {
                        if (mask == 0)
                            return false;
                        const uint32_t slot_idx =
                            __ffs(mask) - 1;
                        mask ^= 1u << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr =
                                math::advance_ptr<uint8_t>(
                                    buffer.combine_token_buffer
                                        .get_rank_buffer(slot_idx)
                                        .get_data_buffer(token_idx)
                                        .get_base_ptr(),
                                    chunk_byte_offset);
                            ptx::tma_load_1d(
                                combine_load_buffer[i], src_ptr,
                                combine_load_barriers[i],
                                kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(
                                combine_load_barriers[i],
                                kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    };

                bool do_reduce =
                    move_mask_and_load(load_stage_idx);
                float2 reduced[
                    kNumUint4PerLane *
                    kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(
                        load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]
                        ->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0;
                         j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values =
                            combine_load_buffer[
                                load_stage_idx]
                                [j * 32 + lane_idx];
                        const auto bf16_values =
                            reinterpret_cast<
                                const nv_bfloat162*>(
                                    &uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0;
                             l < kNumElemsPerUint4; ++ l) {
                            ptx::accumulate(
                                reduced[
                                    j * kNumElemsPerUint4 +
                                    l],
                                bf16_values[l]);
                        }
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0;
                     j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 =
                        reinterpret_cast<nv_bfloat162*>(
                            &casted);
                    #pragma unroll
                    for (uint32_t l = 0;
                         l < kNumElemsPerUint4; ++ l) {
                        casted_bf16[l] =
                            __float22bfloat162_rn(
                                reduced[
                                    j * kNumElemsPerUint4 +
                                    l]);
                    }
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(
                        combine_store_buffer +
                            j * 32 + lane_idx,
                        casted.x, casted.y,
                        casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(
                            y,
                            static_cast<uint64_t>(
                                token_idx) *
                                kNumHiddenBytes +
                            chunk_byte_offset),
                        combine_store_buffer,
                        kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
