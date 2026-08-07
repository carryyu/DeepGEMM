# 乱谈 Kernel 之 DeepGEMM MegaMoE 的设计与实现：当前实现版

> 参考文章：[乱谈 kernel 之 DeepGEMM MegaMoE 的设计与实现](https://zhuanlan.zhihu.com/p/2032133031234888468)
>
> 本文沿用参考文章“先建立全局图景，再沿代码执行顺序下钻”的讲述方式，只把已经变化的实现替换成当前仓库代码。分析基于 2026-07-30 的工作区，仓库 `HEAD` 为 `559d79f`；涉及源码事实时，以工作区内容为准。

## 阅读路线

MegaMoE 同时涉及跨 GPU 通信、GEMM、量化、动态调度和多级同步。如果一开始就追某个 barrier 或某条 PTX，很容易只看见局部。

本文分成四层：

1. 先看一条 token 从 dispatch 到 combine 的完整路径，建立全局心智模型。
2. 再看 kernel 启动前如何规划 symmetric buffer、ring buffer 和 TMA descriptor。
3. 然后解释 scheduler 如何产生任务，以及 full/empty 计数如何保证 ring slot 安全复用。
4. 最后沿 dispatch、loader、MMA、epilogue、combine 的实际执行顺序逐段阅读代码。

章节与这四层的对应关系是：

```text
第 0 节           全局数据流、warp 分工、差异地图
第 1～2 节        Buffer 规划与 kernel 启动准备
第 3 节           Scheduler、TaskInfo、ring 生命周期
第 4～8 节        Dispatch -> Loader -> MMA -> Epilogue -> Combine
第 9 节           从 warmup 开始重新串起完整时间线
第 10 节          用四条原则收束全文
```

全文有三个下标空间，必须从一开始就分清。

| 下标空间 | 代表什么 | 生命周期 | 典型字段 |
| --- | --- | --- | --- |
| 原始 token 空间 | 当前 rank 输入中的第几个 token | 整个 kernel | `src_token_idx`、`y[token]` |
| 逻辑 expert pool 空间 | token 路由到 expert 后形成的第几行 | 覆盖本轮全部 routed rows | `pool_token_idx`、`pool_block_idx`、`token_src_metadata` |
| 物理 ring 空间 | 当前逻辑 block 落到哪一个可复用槽位 | 只覆盖同时存活的窗口 | `ring_block_idx`、`l1/l2_*_count` |

后文所有“为什么要取模”“为什么 metadata 不取模”“为什么要有 generation”，本质上都来自这三个空间的区别。

---

## 0. 先看全局：MegaMoE 到底融合了什么

### 0.1 一次 kernel 完成整个 MoE MLP

MegaMoE 把下面五个阶段放进一个 persistent multi-role kernel：

```text
dispatch
  -> Linear1 GEMM
  -> SwiGLU + FP8 requantization
  -> Linear2 GEMM
  -> combine
```

对 routed experts，一条路由记录的完整路径是：

```text
源 rank 的 x[token] 和 top-k
  -> 源 rank 把 metadata 写到 expert 所在 rank
  -> expert 所在 rank 主动 pull token / SF / top-k weight
  -> 本地 routed L1 ring
  -> L1 GEMM
  -> topk_weight * SiLU(gate) * up
  -> FP8 + UE8M0 SF
  -> 本地 routed L2 ring
  -> L2 GEMM
  -> 远端写回源 rank 的 combine_token_buffer[topk_slot, token]
  -> 源 rank 本地累加所有有效 slot
  -> y[token]
```

如果配置了 shared experts，还会增加一条本 rank 内路径：

```text
x
  -> SharedLinear1
  -> shared L2 activation
  -> SharedLinear2
  -> combine_token_buffer[shared_slot]
  -> 与 routed top-k partial results 一起求和
```

### 0.2 Dispatch 与 combine 的通信方向相反

Dispatch 使用：

```text
metadata push + payload pull
```

源 rank 不直接把完整 token 推到目标 rank，只告诉目标 rank：

```text
local expert 的这个 slot
来自哪个 src_token_idx 和 src_topk_idx
```

目标 rank 等 metadata 可见后，再主动 pull token、SF 和 weight。这样目标 rank 可以按 local expert 顺序组织自己的逻辑 pool。

Combine 则反过来：

```text
expert rank 计算完成
  -> 直接把 partial result push 回源 rank
  -> 源 rank 本地 reduce
```

这种不对称是合理的：

- dispatch 需要目标 rank 控制 expert 输入布局；
- combine 已经知道每条 expert row 的来源，直接写回最短。

### 0.3 当前代码相对参考文章，核心变化在哪里

先说不变的部分：MoE 数学、`metadata push + payload pull` 的 dispatch 方向、L2 结果远端 push 后本地 combine 的方向，以及 loader/MMA/epilogue 的 warp specialization 都仍然成立。

真正改变的是“中间状态由谁管理、保存多久、如何发布”。下面这张表先给出完整差异地图；后文再在对应执行环节解释每项变化为什么成立。

| # | 参考文章中的实现 | 当前仓库实现 | 详细解释 |
| --- | --- | --- | --- |
| 1 | warp 7 基本预留 | leader CTA 的 warp 7 是专用 scheduler | 0.4、3.3 |
| 2 | 各计算角色分别枚举 block | scheduler 统一生产双缓冲 `TaskInfo`，所有角色消费同一序列 | 3.3 |
| 3 | 按固定 expert wave，先一批 L1、再一批 L2 | 全局原子计数动态领取；L1 warmup 后，L2/L1 交叉推进 | 3.4～3.8 |
| 4 | routed L1/L2 activation 按完整 pool 分配 | activation 只按最大 live window 分配 ring；完整 pool 只保留逻辑身份 | 1.5～1.8 |
| 5 | SF pool 由完整 pool block 数推导 | SF ring 按每个候选 `BLOCK_M` 的 `num_ring_tokens / BLOCK_M * align(BLOCK_M, 128)` 取最大值 | 1.9、4.10 |
| 6 | workspace barrier 热区为 32 B | 前 128 B 同时隔离 grid/NVLink barrier 与 routed/shared task counters | 1.10 |
| 7 | dispatch 通过 `l1_arrival_count` 发布逻辑 pool block | `l1_empty_count` 保护覆盖，`l1_full_count` 按 generation 发布物理 ring slot | 3.11、4.8～4.12 |
| 8 | L1 epilogue 置 `l2_arrival_mask` 的 N-block bit，L2 按 K block 等 mask | `l2_empty_count` 保护覆盖，`l2_full_count` 累计全部 L1 N blocks；L2 在任务入口一次等待完整输入 | 3.12、5.2、8.9 |
| 9 | 完整 pool 的 block 不会复用 | `pool_block_idx` 被拆成 `ring_block_idx + generation`，full/empty 都使用累计代际目标 | 3.10～3.13 |
| 10 | 容易理解成 MMA 算完后再把 accumulator 写入 TMEM | UMMA 直接在 TMEM 中累加，最后一个 K stage 用 `tmem_full` 发布 | 7.3 |
| 11 | 主要覆盖 routed experts | 新增 `SharedLinear1/SharedLinear2`，复用同一 GEMM 流水线并占一个额外 combine slot | 0.1、3.9、8.12 |
| 12 | routed L1/L2 共九个 TMA descriptors | routed 与 shared 两组共十八个 descriptors | 2.4 |
| 13 | combine buffer 只有 top-k routed slots | shape 增加可选 shared slot：`topk + (has_shared ? 1 : 0)` | 8.12、8.15 |

把 13 项再压缩，只剩三条结构性主线：

```text
存储：
    完整 activation pool
      -> 逻辑 full pool + 物理 live-window ring

调度：
    每个角色静态枚举
      -> scheduler warp 统一发布 TaskInfo
      -> warmup 后 L2/L1 动态交叉

同步：
    单向 arrival count / bitmask
      -> 带 generation 的 L1/L2 full/empty 生命周期协议
```

后文不再把“旧实现”和“当前实现”割裂成两篇文章，而是沿参考文章的执行顺序，在每个机制出现时直接换成当前代码。

### 0.4 Warp 怎么分工

kernel 以 2-CTA cluster 运行。默认 FP8/FP4 配置中，每个 CTA 的 warp 角色如下：

| Warp | 角色 | 主要工作 |
| --- | --- | --- |
| 0–3 | dispatch warps | 统计路由、写 metadata、远端 pull、填 L1 ring、清 workspace |
| 4 | token/SFA loader | 将 L1/L2 activation 和 SFA 搬到 SMEM |
| 5 | weight/SFB loader | 将 routed/shared weight 和 SFB 搬到 SMEM |
| 6 | MMA issue warp | UTCCP 搬 SF、发出 2-CTA UMMA、管理 SMEM/TMEM 流水线 |
| 7 | scheduler warp | 动态领取任务，向两个 CTA 发布双缓冲 `TaskInfo` |
| 8 以后 | epilogue warps | L1 SwiGLU/量化、L2 远端写回、最终 combine |

参考文章中的 warp 7 还是预留角色；当前代码中它已经是控制整个 GEMM 任务流的 scheduler warp。

### 0.5 同步关系先压成一张表

| 生产者 | 消费者 | 当前同步对象 | 保证什么 |
| --- | --- | --- | --- |
| dispatch 统计 | scheduler / pull | `expert_recv_count_sum` | expert token 数已经由所有 SM、所有 rank 聚合 |
| scheduler warp | loaders / MMA / epilogue | `task_info_full/empty_barriers` | 所有角色看到同一条任务流 |
| dispatch pull | L1 loader | `l1_full_count` | 当前 generation 的 L1 token block 已写完 |
| L1 epilogue | dispatch pull | `l1_empty_count` | L1 ring slot 可以被下一代覆盖 |
| L1 epilogue | L2 loader | `l2_full_count` | 当前 generation 的全部 L1 输出已写入 L2 ring |
| L2 epilogue | L1 epilogue | `l2_empty_count` | L2 ring slot 可以被下一代覆盖 |
| A/B loader | MMA | `full_barriers` | 当前 SMEM stage 的 A/B/SF 都 ready |
| MMA | A/B loader | `empty_barriers` | 当前 SMEM stage 已消费，可以复用 |
| MMA | epilogue | `tmem_full_barriers` | 当前 TMEM accumulator tile 完成 |
| epilogue | MMA | `tmem_empty_barriers` | epilogue 已不再读取该 TMEM stage |
| L2 epilogue | combine | `nvlink_barrier` | 所有远端 partial-result 写已完成 |

这张表后面会逐项展开。现在只需要记住：kernel 中同时存在 task 流水线、SMEM 流水线、TMEM 流水线和两个 global-memory ring 生命周期。

---

## 1. 通信 Buffer 的分配和划分

本节只回答一个问题：

> kernel 启动前，需要准备哪些长期存在的 global-memory 状态，它们为什么这样分配？

### 1.1 为什么需要 symmetric memory

MegaMoE 不是：

```text
通信 kernel
  -> GEMM kernel
  -> activation kernel
  -> GEMM kernel
  -> 通信 kernel
```

而是在同一个 kernel 内直接访问其他 rank 的 buffer。因此每个 rank 必须：

1. 分配大小和布局完全相同的 raw buffer；
2. 建立各 rank 基地址之间的映射；
3. 保证相同字段在所有 rank 上拥有相同 offset。

完成 `symm_mem.rendezvous` 后：

```cpp
sym_buffer.map(local_ptr, dst_rank_idx)
```

等价于：

```text
目标 rank 基地址
  + local_ptr 在本地 raw buffer 中的 offset
```

所以 symmetric memory 解决的是“同 offset 的远端寻址”，不是自动帮 kernel 完成 all-to-all。

### 1.2 Host 侧调用链

buffer 建立流程是：

```text
get_symm_buffer_for_mega_moe(...)
  -> 对齐 num_max_tokens_per_rank
  -> SymmBuffer(...)
      -> _C.get_symm_buffer_size_for_mega_moe(...)
          -> 计算 ring 上界
          -> 构造 MegaMoEBuffer 布局
          -> 返回总字节数和 slice lambda
      -> symmetric allocator 分配 raw buffer
      -> rendezvous 建立远端映射
      -> slice lambda 创建各 tensor view
```

这里最重要的是：`get_symm_buffer_size_for_mega_moe()` 是布局规划器，不是真正的分配器。

### 1.3 四个布局对象

#### `TokenSrcMetadata`

```cpp
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};
```

它回答 combine 的反向寻址问题：

```text
逻辑 expert pool 的这一行
原来属于哪个 rank 的哪个 token、哪个 top-k slot？
```

#### `Data`

`Data` 描述一个逻辑元素占多少字节，以及是否需要 TMA 对齐。例如：

```text
一个 FP8 token row       -> hidden bytes
一个 BF16 output row     -> hidden * 2 bytes
一个 routed top-k weight -> sizeof(float)
```

#### `Buffer`

`Buffer` 在 `Data` 上增加：

```text
外层有多少组
每组有多少个 data elements
base 在哪里
```

`get_rank_buffer(i)` 的名字有历史色彩，实际语义是“取外层第 i 组”。对 `combine_token_buffer`，这个 i 是 top-k slot，不一定是分布式 rank。

#### `Workspace` 与 `MegaMoEBuffer`

`Workspace` 负责控制面：

- barrier 和 task counters；
- expert send/recv counts；
- ring full/empty counters；
- dispatch metadata；
- combine 反向 metadata。

`MegaMoEBuffer` 在 workspace 后继续串接输入、shared expert、中间 ring 和 combine 数据区。

### 1.4 先算 routed rows 的最坏上界

定义：

```text
R  = num_ranks
M  = num_max_tokens_per_rank
K  = num_topk
Er = num_experts_per_rank
```

一个原始 token 最多给当前 rank 贡献：

```text
min(K, Er)
```

条 routed expert rows，所以当前 rank 的 routed row 总上界为：

```text
num_max_routed_tokens =
    M * R * min(K, Er)
```

这里的一行对应一个有效 `(token_idx, topk_slot)`，不是原始 token 本身。

### 1.5 为什么逻辑 pool 仍然必须覆盖整轮

每个 local expert 在逻辑 pool 中占一段：

```text
expert 0: ceil(tokens_0 / BLOCK_M) blocks
expert 1: ceil(tokens_1 / BLOCK_M) blocks
...
```

逻辑 pool 的最大 token 数仍由：

```cpp
get_num_max_pool_tokens(...)
```

估计：

```text
num_max_pool_tokens =
    align(
        num_max_routed_tokens
          + Er * (kMaxCandidateBlockM - 1),
        384)
```

当前它主要服务：

```text
token_src_metadata[num_max_pool_tokens]
```

这份 metadata 不能做成 ring。原因是 L2 epilogue 需要按逻辑 `pool_token_idx` 反查来源；某个 activation slot 即使已被下一代复用，尚未完成 combine 的逻辑行身份仍不能丢失。

### 1.6 为什么 activation 可以只保留 live window

L1/L2 activation 与 metadata 的生命周期不同。

一个 block 一旦：

```text
完成 L2 GEMM 对输入的消费
```

它的 L1/L2 activation 就不再需要保留。调度继续向后推进时，可以让更晚的逻辑 block 复用这个物理槽位。

因此：

```text
token_src_metadata -> 完整逻辑 pool
l1/l2 activation   -> 物理 ring
```

### 1.7 `num_ring_tokens` 如何计算

buffer 分配时，host 还不知道最终 heuristic 会选择哪个 `BLOCK_M`，所以必须遍历：

```text
B in {8, 16, 32, 64, 96, 128, 192}
```

对每个候选 B：

```text
P(B) =
    ceil(num_max_routed_tokens / B) + Er
```

`+ Er` 保守覆盖每个 expert 各自按 B 补齐造成的额外 block。

然后计算当前动态调度下最多同时存活多少个逻辑 block：

```text
L(B) =
    get_num_max_live_pool_blocks(
        P(B), num_sms, hidden, intermediate_hidden)
```

当前候选需要：

```text
candidate_ring_tokens(B) = L(B) * B
```

最终：

```text
num_ring_tokens =
    align(max_B(candidate_ring_tokens(B)), 384)
```

`384` 是所有候选 `BLOCK_M` 的最小公倍数，保证同一份分配能被不同候选整除。

### 1.8 `get_num_max_live_pool_blocks()` 的第一性原理

先定义：

```text
P  = num_total_m_blocks
S  = num_sms / 2
C1 = intermediate_hidden / 128
C2 = hidden / 256
```

解释如下：

- 一个 2-CTA cluster 占两个 SM，因此全局并行 cluster 数是 `S`；
- L1 输出宽度是 `2 * intermediate_hidden`，每个 cluster 覆盖两个 `BLOCK_N=128`，所以每个 M block 的 L1 cluster tasks 是 `C1`；
- L2 输出宽度是 `hidden`，同理每个 M block 的 L2 cluster tasks 是 `C2`。

#### 第一部分：warmup 会制造多少 live blocks

L1 总 wave 数：

```text
num_l1_waves = ceil(P * C1 / S)
```

第一波 L2 最多触及：

```text
first_l2_wave_m_blocks = ceil(S / C2)
```

为了让这些 block 的 L1 task 都已经发布，至少需要：

```text
warmup_for_first_l2 =
    ceil(first_l2_wave_m_blocks * C1 / S)
```

当 `C1 > C2` 时，稳态每轮 L2/L1 交叉仍可能积累 L1 前沿，因此再计算：

```text
diff = max(C1 - C2, 0)

warmup_for_interleave =
    ceil((C1 + (P - 1) * diff) / S) + 1
```

实际 warmup：

```text
warmup_waves =
    min(
        max(warmup_for_first_l2,
            warmup_for_interleave),
        num_l1_waves)
```

warmup 后已经进入 live window 的 M blocks：

```text
warmup_clusters =
    min(warmup_waves * S, P * C1)

live_after_warmup =
    ceil(warmup_clusters / C1)
```

#### 第二部分：稳态前沿还可能增长多少

代码为 L1/L2 前沿速率差和不完整 wave 留两个保守余量：

```text
frontier_growth =
    C2 > C1
      ? ceil(P * (C2 - C1) / C2)
      : 0

wave_margin =
    ceil(S / min(C1, C2))
```

最终：

```text
max_live_blocks =
    min(
        P,
        live_after_warmup
          + frontier_growth
          + wave_margin)
```

这个函数不是运行时动态规划，也不是精确模拟每个 CTA 的时间线；它是针对当前调度策略的保守闭式上界。

### 1.9 SF ring 为什么比 token ring 大

Token ring 每个 block 占：

```text
BLOCK_M
```

SFA 为了满足 128-row UTCCP 布局，每个 block 占：

```text
SF_BLOCK_M = align(BLOCK_M, 128)
```

所以对候选 B：

```text
sf_ring_tokens(B) =
    (num_ring_tokens / B) * align(B, 128)
```

host 对所有候选取最大：

```text
num_sf_ring_tokens =
    max_B(sf_ring_tokens(B))
```

因此：

```text
l1_token_buffer -> num_ring_tokens rows
l1_sf_buffer    -> num_sf_ring_tokens SF rows
```

它们不是同一个 token 维容量。

### 1.10 当前 Workspace 布局

FP8/FP4 路径的核心 workspace 字段如下：

| 区域 | 容量 | 作用 |
| --- | --- | --- |
| barrier/schedule hot area | 128 B | grid/NVLink barrier 和 routed/shared task counters |
| `expert_send_count` | `uint64[num_experts]` | 每个全局 expert 的 send count + SM 到达数 |
| `expert_recv_count` | `uint64[R, Er]` | local expert 从每个源 rank 收到多少行 |
| `expert_recv_count_sum` | `uint64[Er]` | local expert 总行数 + 全局统计到达数 |
| `l1_full_count` | ring-block count array | dispatch 发布 L1 rows |
| `l1_empty_count` | ring-block count array | L1 epilogue释放 L1 slot |
| `l2_full_count` | ring-block count array | L1 epilogue发布 L2 block |
| `l2_empty_count` | ring-block count array | L2 epilogue释放 L2 slot |
| `shared_l2_full_count` | shared M-block count array | shared L1 发布 shared L2 |
| `src_token_topk_idx` | `[Er, R, R*M]` | dispatch pull 的源索引 |
| `token_src_metadata` | full logical pool | routed combine 的反查索引 |

workspace 要支持最小候选 `BLOCK_M=8`，因此 host 布局中的 count 数组长度按：

```text
num_ring_tokens / 8
```

分配。具体 kernel 实例只使用：

```text
kNumRingBlocks =
    kNumRingTokens / BLOCK_M
```

个物理槽位。

### 1.11 当前数据区布局

忽略 FP8/FP4 路径中大小为零的 NVFP4 专用字段，主要区域如下：

| Buffer | 逻辑 shape | 生命周期 |
| --- | --- | --- |
| `x` | FP8 `[M, hidden]` | 原始输入 |
| `x_sf` | packed SF `[M, hidden/128]` | 原始输入 SF |
| `topk_idx` | int64 `[M, topk]` | 路由 |
| `topk_weights` | FP32 `[M, topk]` | 路由权重 |
| `shared_l1_acts` | alias `x` | shared L1 输入 |
| `shared_l2_acts` | FP8 `[M, I*num_shared]` | shared L2 输入 |
| `l1_acts` | FP8 `[num_ring_tokens, hidden]` | routed L1 ring |
| `l1_acts_sf` | `[num_sf_ring_tokens, hidden/128]` | routed L1 SF ring |
| `l1_topk_weights` | FP32 `[num_ring_tokens]` | routed row 权重 |
| `l2_acts` | FP8 `[num_ring_tokens, I]` | routed L2 ring |
| `l2_acts_sf` | `[num_sf_ring_tokens, I/128]` | routed L2 SF ring |
| `combine_token_buffer` | BF16 `[topk+has_shared, M, hidden]` | partial results |

到这里，global-memory 的长期对象已经齐了。下一步看 kernel 启动前如何根据这些 tensor 创建具体执行配置。

---

## 2. Kernel 启动前的准备

### 2.1 `MegaMoEConfig` 的职责

当前配置保存四类信息：

1. GEMM tile：`block_m/n/k`、`load_block_m/n`、`store_block_m`；
2. SF tile：`sf_block_m/n`；
3. ring 容量：`num_ring_tokens`、`num_sf_ring_tokens`；
4. 资源配置：swizzle、mainloop stages、shared-memory bytes 和各角色线程数。

参考文章中的 `num_experts_per_wave` 已不在当前配置里；当前调度由 runtime task counters 和 warmup 公式决定。

### 2.2 Block 配置

FP8/FP4 路径固定：

```text
BLOCK_N = 128
```

host heuristic 根据 workload 选择：

```text
BLOCK_M
STORE_BLOCK_M
BLOCK_K
num_epilogue_threads
```

kernel 内：

```text
LOAD_BLOCK_M = BLOCK_M / 2
LOAD_BLOCK_N = BLOCK_N
SF_BLOCK_M   = align(BLOCK_M, 128)
SF_BLOCK_N   = BLOCK_N
```

`LOAD_BLOCK_M` 只有一半，是因为 2-CTA UMMA 将 activation operand 分布在两个 CTA 的 SMEM 中。`SF_BLOCK_M` 反而可能大于 `BLOCK_M`，是因为 SFA 要按 128-row 页面组织。

### 2.3 Shared-memory stage 数

host 先计算固定开销：

- dispatch expert count 和 pull staging；
- L1/L2 epilogue staging union；
- amax reduction；
- 双缓冲 `TaskInfo`；
- dispatch、TMEM、combine、schedule barriers；
- TMEM pointer。

再计算每个 GEMM stage 的开销：

```text
smem_a + smem_b + smem_sfa + smem_sfb
+ full barrier + empty barrier
```

最终：

```text
num_stages =
    floor((smem_capacity - smem_fixed)
          / smem_per_stage)
```

并要求至少两个 stage。

### 2.4 TMA descriptors

Routed 路径有九个 descriptor：

```text
l1_acts
l1_acts_sf
l1_weights
l1_weights_sf
l1_output
l2_acts
l2_acts_sf
l2_weights
l2_weights_sf
```

Shared 路径也有对应九个。kernel 接口总共携带十八个 descriptor；没有 shared experts 时，host 用 routed descriptor 作为不会被有效任务访问的占位。

`l1_output` 与 `l2_acts` 指向同一块数据。SwiGLU 将 gate/up 两个通道合并，所以：

```text
L1 output tile width = BLOCK_N / 2
L1 output swizzle    = activation swizzle / 2
```

### 2.5 SFA tensor view 为什么是特殊 stride

`l1_acts_sf` 的 Python view 是：

```text
shape  = [num_sf_ring_tokens, hidden_packed_sf]
stride = [1, num_sf_ring_tokens]
```

因此它的物理地址是：

```text
SF_addr(j, sf_token_idx) =
    j * num_sf_ring_tokens + sf_token_idx
```

即每个 K 方向 packed-SF group 拥有一整个连续 SF ring plane。这一点会直接出现在 dispatch 写 SFA 的地址公式中。

---

## 3. Kernel 的整体执行模型

本节先解释控制面：2-CTA cluster、初始化、`TaskInfo`、scheduler 和 ring 生命周期。理解这些之后，再进入各 warp 的具体代码。

### 3.1 2-CTA cluster 如何定义一个 GEMM task

`TaskInfo` 中记录的是：

```text
n_cluster_idx
```

两个 CTA 分别得到：

```text
leader CTA:
    n_block_idx = 2 * n_cluster_idx

non-leader CTA:
    n_block_idx = 2 * n_cluster_idx + 1
```

所以一个 scheduler task 实际覆盖同一 M block 上相邻的两个 N blocks。

逻辑上：

```text
两个 CTA 使用同一批 token
但使用不同的 weight/output N block
```

### 3.2 初始化阶段

kernel 开头依次完成：

1. 预取 routed/shared TMA descriptors；
2. 根据 symmetric-buffer base 构造 `MegaMoEBuffer`；
3. 建立 shared-memory 视图；
4. 2-CTA cluster sync；
5. 清 shared expert counters；
6. 初始化所有 mbarriers；
7. 为 2-CTA TMEM 分配 columns；
8. 再次 cluster sync；
9. 构造 scheduler。

关键 barrier 初值：

```text
full_barriers[stage]       = 2 CTAs * 2 operand producers
empty_barriers[stage]      = 1
tmem_full_barriers[stage]  = 1
tmem_empty_barriers[stage] = 2 * kNumEpilogueThreads
task_info_full[stage]      = 1
task_info_empty[stage]     = 2 * kNumEpilogueThreads
```

### 3.3 `TaskInfo` 是所有 GEMM 角色的共同控制消息

当前 `TaskInfo` 包含：

```cpp
block_phase
local_expert_idx
m_block_idx
n_cluster_idx
pool_block_idx
valid_m
shape_n
shape_k
```

`block_phase` 可能是：

```text
Linear1
Linear2
SharedLinear1
SharedLinear2
None
```

warp 7 的 scheduler 生产 `TaskInfo`；token loader、weight loader、MMA 和 epilogue 各自调用 `get_next_task()`，但消费的是同一序列。

双缓冲过程：

```text
scheduler 等 task_info_empty[stage]
  -> 构造 TaskInfo
  -> st_async_cluster 写两个 CTA 的 task_infos[stage]
  -> task_info_full[stage]
  -> loaders / MMA / epilogue 依次读取
  -> epilogue release_task_info()
  -> task_info_empty[stage]
```

最后发布 `BlockPhase::None`，所有消费者退出 persistent loop。

### 3.4 Scheduler 第一步：读取每个 expert 的 token 数

dispatch 向 `expert_recv_count_sum[local_expert]` 累加一个 64-bit 状态：

```text
低 32 位：token count
高 32 位：有多少个 (rank, SM) 已贡献统计
```

scheduler 等：

```text
high32(value) == kNumRanks * kNumSMs
```

再缓存低 32 位。

每个 expert 的 M-block 数：

```text
num_m_blocks[e] =
    ceil(num_tokens[e] / BLOCK_M)
```

其前缀和定义逻辑 pool：

```text
pool_block_offset[e] =
    sum_{i<e} num_m_blocks[i]
```

### 3.5 Scheduler 第二步：把全局 task index 反解成 expert tile

L1 与 L2 各有一个全局原子 counter：

```text
l1_task_count
l2_task_count
```

领取的 task index 先映射为：

```text
global_m_block_idx =
    task_idx / num_n_clusters

n_cluster_idx =
    task_idx % num_n_clusters
```

再用 expert block 前缀和反解：

```text
local_expert_idx
expert 内 m_block_idx
pool_block_idx
valid_m
```

这叫动态任务领取，不是动态规划算法。

### 3.6 Warmup：先建立不会死锁的 L1 前沿

每个 M block 的 cluster task 数：

```text
C1 = kNumL1Clusters
C2 = kNumL2Clusters
```

warmup 期间，每个 cluster scheduler 每轮领取一个 L1 task；所有 cluster 合起来形成一个全局 L1 wave。

warmup 的目的不是要求所有 L1 都完成，而是保证进入稳态后：

```text
任何准备发布的 L2 task
都不会依赖一个尚未有机会被领取的 L1 task
```

具体 warmup wave 数与 1.8 节的 live-block 公式使用同一套计算。

### 3.7 稳态：L2 与 L1 交叉领取

warmup 之后，一个 scheduler 的局部顺序是：

```text
领取一个 L2
  -> 下一次领取一个 L1
  -> 再领取一个 L2
  -> 再领取一个 L1
```

所有 cluster 使用同一组全局 counters，所以宏观上形成动态交叉，而不是先把所有 L1 调完再调 L2。

### 3.8 为什么 L2 不会造成任务层死锁

scheduler 领取 L2 后，先计算：

```text
num_required_l1_tasks =
    (pool_block_idx + 1) * kNumL1Clusters
```

只有观察到：

```text
l1_task_count >= num_required_l1_tasks
```

才发布该 L2 `TaskInfo`。

这只是任务依赖。真正的数据依赖还由 loader 检查：

```text
任务门：
    所需 L1 tasks 已经被领取

数据门：
    所需 L1 epilogue 已经写完 L2 ring
    即 l2_full_count 达到 generation 目标
```

把这两层分开，是理解当前 scheduler 的关键。

### 3.9 Shared experts 的调度位置

存在 shared experts 时，scheduler 顺序是：

```text
SharedLinear1
  -> 等 routed dispatch count
  -> routed L1/L2 warmup + steady state
  -> SharedLinear2
  -> sentinel
```

SharedLinear1 直接读本 rank 原始 `x`，不依赖 routed dispatch，因此可以先运行。

### 3.10 Ring 的三个坐标

对一个 routed `pool_block_idx=p`：

```text
RING = kNumRingBlocks

ring_block_idx = p % RING
generation     = p / RING
```

例如 `RING=4`：

```text
logical block:  0 1 2 3 4 5 6 7 8 ...
ring slot:      0 1 2 3 0 1 2 3 0 ...
generation:     0 0 0 0 1 1 1 1 2 ...
```

取模解决“写到哪个物理槽位”，generation 解决“当前等的是这个槽位的第几次使用”。

### 3.11 L1 ring 的 full/empty 协议

定义：

```text
L1_N = L1_SHAPE_N / BLOCK_N
g    = generation
r    = ring_block_idx
```

Dispatch 覆盖 L1 slot 前等待：

```text
l1_empty_count[r] >= g * L1_N
```

每个 token row 完成 pull 后：

```text
l1_full_count[r] += 1
```

尾 block 最后一个有效 token 一次补齐 padding，使每代恰好增加 `BLOCK_M`。L1 loader 因而统一等待：

```text
l1_full_count[r] ==
    BLOCK_M * (g + 1)
```

每个 L1 N-block 的 epilogue 完成后：

```text
l1_empty_count[r] += 1
```

一代累计 `L1_N` 次，下一代 dispatch 才能覆盖该 slot。

### 3.12 L2 ring 的 full/empty 协议

L1 epilogue 写 L2 ring 前等待：

```text
l2_empty_count[r] ==
    g * (L2_SHAPE_N / BLOCK_N)
```

每个 L1 N-block 完成 SwiGLU 和 TMA store 后：

```text
l2_full_count[r] += 1
```

L2 loader 等：

```text
l2_full_count[r] ==
    (L2_SHAPE_K / BLOCK_N)
    * 2
    * (g + 1)
```

因为：

```text
2 * L2_SHAPE_K = L1_SHAPE_N
```

右侧正好等价于“这一代全部 L1 N blocks 都已经写完”。

每个 L2 N-block 进入 epilogue 后：

```text
l2_empty_count[r] += 1
```

一代全部 L2 N blocks 消费后，下一代 L1 epilogue 才能覆盖 L2 slot。

### 3.13 为什么用累计 generation，而不是清零后重用

如果只使用 0/1 标志：

```text
旧代迟到写
新代提前清零
```

可能形成 ABA 混淆。

累计目标将同一 slot 的状态写成：

```text
generation 0 -> target T
generation 1 -> target 2T
generation 2 -> target 3T
```

生产者和消费者都明确知道自己等待哪一代。整轮 kernel 结束后，cleanup 再统一清零。

### 3.14 三条流水线互相独立

#### TaskInfo 流水线

```text
scheduler -> loaders -> MMA -> epilogue
```

由 `task_info_full/empty_barriers` 管理。

#### SMEM operand 流水线

```text
A/SFA + B/SFB loaders
  -> full_barrier
  -> MMA
  -> empty_barrier
```

按 K stage 轮转。

#### TMEM accumulator 流水线

```text
MMA
  -> tmem_full
  -> epilogue
  -> tmem_empty
```

按完整 GEMM task 轮转。

这三条流水线的 stage index 和释放时机不同，不能把它们看成一个统一 barrier。

---

## 4. Dispatch Warps

本节沿代码执行顺序展开 dispatch 的三个阶段。

### 4.1 输入与输出

输入：

- 本 rank 的 `x/x_sf`；
- `topk_idx/topk_weights`；
- symmetric buffer 的远端映射。

输出：

- `expert_recv_count` 和 `expert_recv_count_sum`；
- `src_token_topk_idx`；
- routed `l1_token/l1_sf/l1_topk_weights` ring；
- 非 ring 的 `token_src_metadata`；
- `l1_full_count`。

### 4.2 第一阶段：统计每个 expert 的 token 数

dispatch warp 以 grid-stride 遍历原始 token/top-k 条目。

第一遍在 CTA shared memory 中执行：

```text
expert_token_count[global_expert]++
```

随后每个 SM 向 global：

```text
expert_send_count[global_expert]
```

原子加：

```text
(1ULL << 32) | local_count
```

低 32 位累计 token 数，高 32 位累计贡献统计的 SM 数。

### 4.3 第二遍：为每条路由分配 metadata slot

同一个 top-k 条目编码为：

```text
token_topk_idx =
    token_idx * kNumTopk + topk_slot
```

源 rank 根据目标 expert 分配 slot，然后远端写：

```text
src_token_topk_idx[
    dst_local_expert,
    src_rank,
    dst_slot
] = token_topk_idx
```

这张表回答的是：

> 目标 rank 的这个 local expert，要去哪个源 rank 拉哪个 token 的哪个 top-k 槽位？

### 4.4 Grid sync 后发布 expert counts

所有 SM 完成 metadata slot 分配后，SM0 把：

```text
expert_recv_count[src_rank, dst_local_expert]
```

写到目标 rank，再以 system-scope atomic add 聚合：

```text
expert_recv_count_sum[dst_local_expert]
```

scheduler 和 dispatch pull 都会读取这份统计。

### 4.5 为什么 pull 前需要 NVLink barrier

目标 rank pull 前必须保证所有源 rank 的：

- `src_token_topk_idx`；
- `expert_recv_count`；
- `expert_recv_count_sum`

都已经可见。

因此执行 `kBeforeDispatchPullBarrierTag` 的 NVLink barrier。这个 barrier 不是仅等待计数；它还通过 system-scope release/acquire 建立跨 GPU 数据可见性。

### 4.6 第二阶段：遍历目标 rank 的逻辑 expert pool

目标 rank 先从 cached expert counts 得到：

```text
expert_start_idx
expert_end_idx
expert_pool_block_offset
```

全局 `token_idx` 落在某个 expert 的累计区间时：

```text
token_idx_in_expert =
    token_idx - expert_start_idx
```

逻辑 pool row：

```text
pool_token_idx =
    expert_pool_block_offset * BLOCK_M
    + token_idx_in_expert

pool_block_idx =
    pool_token_idx / BLOCK_M
```

### 4.7 如何找回源 rank

`expert_recv_count[src_rank, expert]` 给出各源 rank 对当前 expert 的贡献数量。

代码用 round-robin min-peeling 将 expert 内线性 slot 映射为：

```text
current_rank_in_expert_idx
token_idx_in_rank
```

然后读取：

```text
src_token_topk_idx[
    current_expert,
    current_rank,
    token_idx_in_rank
]
```

并解码：

```text
src_token_idx =
    src_token_topk_idx / kNumTopk

src_topk_idx =
    src_token_topk_idx % kNumTopk
```

### 4.8 覆盖 L1 ring 前先等上一代释放

```text
ring_block_idx =
    pool_block_idx % kNumRingBlocks

generation =
    pool_block_idx / kNumRingBlocks
```

Dispatch 等：

```text
l1_empty_count[ring_block_idx]
    >= generation * kNumL1BlockNs
```

这一步限制生产者不能覆盖仍被 L1 pipeline 使用的旧数据。

### 4.9 Token 如何 pull

一个 token row 按 `kNumBytesPerPull` 分 chunk：

```text
远端 GMEM
  -> TMA load
  -> 每 warp 的 pull_buffer
  -> TMA store
  -> 本地 L1 token ring
```

目标地址是：

```cpp
pool_token_idx % kNumRingTokens
```

最后一个 chunk 的远端 load 与 SF 的标量搬运重叠。

### 4.10 SFA 如何写入 SF ring

输入 SF 对一个 token 是连续的：

```text
kNumSFUint32 = hidden / 128
```

每个 `uint32` 打包四个 UE8M0 scale，覆盖四个 32-element K groups。

本地 SF ring 不是普通 token-major，而是：

```text
[packed-K group][SF ring token]
```

所以地址是：

```cpp
local_sf_ptr[
    j * kNumSFRingTokens
    + sf_ring_token_idx
]
```

其中：

```cpp
ring_block_idx =
    pool_block_idx % kNumRingBlocks;

token_idx_in_block =
    token_idx_in_expert % BLOCK_M;

sf_ring_token_idx =
    ring_block_idx * SF_BLOCK_M
    + transform_sf_token_idx(token_idx_in_block);
```

这里的取模已经发生在 block 层。不能写成：

```text
pool_token_idx % kNumSFRingTokens
```

因为每个逻辑 `BLOCK_M` 在 SF ring 中占 `SF_BLOCK_M`，并且 block 内还进行了 128-row 转置。

地址范围天然满足：

```text
sf_ring_token_idx
  < kNumRingBlocks * SF_BLOCK_M
  <= kNumSFRingTokens
```

### 4.11 Weight 与反向 metadata

dispatch 同时远端读取：

```text
topk_weights[src_token_topk_idx]
```

写入物理：

```text
l1_topk_weights[
    pool_token_idx % kNumRingTokens
]
```

但来源身份写到逻辑：

```text
token_src_metadata[pool_token_idx] = {
    src_rank,
    src_token_idx,
    src_topk_idx
}
```

一个取模、一个不取模，正好对应数据 ring 与完整 metadata 的不同生命周期。

### 4.12 发布 `l1_full_count`

最后一个 token chunk store 完成后，dispatch release-add：

```text
l1_full_count[ring_block_idx]
```

普通行加 1；expert 最后一个不足 `BLOCK_M` 的 block 在最后一行一次补齐 padding：

```text
increment =
    BLOCK_M - token_idx_in_expert % BLOCK_M
```

因此每个逻辑 block、每一代都恰好贡献 `BLOCK_M`，L1 loader 无需对尾块使用不同等待公式。

### 4.13 第三阶段：清理 workspace

Dispatch pull 结束后，dispatch warps 等 epilogue 到达 combine 边界，然后与 combine reduction 重叠清理。

SM0 清：

- `expert_send_count`；
- routed/shared task counters；
- `shared_l2_full_count`。

其他 SM 分摊：

- `expert_recv_count_sum`；
- `expert_recv_count`；
- 本轮用到的四组 routed ring counters。

最后通过 `kAfterWorkspaceCleanBarrierTag` 保证所有 rank 都完成清理。

---

## 5. Token/SFA Loader Warp

### 5.1 它先消费 `TaskInfo`

根据 `block_phase` 选择：

```text
Linear1       -> routed L1 activation/SFA
Linear2       -> routed L2 activation/SFA
SharedLinear1 -> shared L1 activation/SFA
SharedLinear2 -> shared L2 activation/SFA
```

对 routed task：

```text
pool_block_idx -> ring_block_idx
```

对 shared task：

```text
block_idx = pool_block_idx
```

因为 shared 中间 buffer 不使用 routed ring。

### 5.2 进入 K loop 前先等 global-memory 输入 ready

当前等待粒度是完整 M block：

```text
Routed L1:
    wait l1_full_count generation target

Routed L2:
    wait l2_full_count generation target

Shared L1:
    原始 x 已存在，不需要 arrival counter

Shared L2:
    wait shared_l2_full_count
```

参考文章中的 L2 per-K `l2_arrival_mask` 已不存在。当前 L2 task 开始前一次性等全部 L1/SwiGLU 输出。

### 5.3 每个 K stage 的 SMEM 生产过程

对每个 K block：

```text
wait empty_barrier[stage]
  -> TMA A
  -> TMA SFA
  -> arrive/expect full_barrier[stage]
```

`empty_barrier` 保护当前 SMEM stage 不被 loader 提前覆盖。

### 5.4 为什么 non-leader CTA 只偏移 activation 的 `m_idx`

代码中：

```cpp
uint32_t m_idx = block_idx * BLOCK_M;
const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;

if (not is_leader_cta)
    m_idx += task_info.get_umma_aligned_valid_m() / 2;
```

每个 CTA 的 activation SMEM 只容纳：

```text
LOAD_BLOCK_M = BLOCK_M / 2
```

因此两个 CTA 分别装 activation tile 的两半。

SFA 不同。每个 CTA 的 `smem_sfa` 容纳完整：

```text
SF_BLOCK_M * (BLOCK_K / 128)
```

两个 CTA 都从同一个 `sfa_m_idx` 读取完整 SFA page。原因是两个 CTA 逻辑上处理同一批 token、不同输出 N blocks；activation scale 只依赖 token row 和 K group，不依赖输出 N block。

barrier transaction bytes 也直接说明了这一点：

```text
2 * sizeof(smem_a)
+ 2 * sizeof(smem_sfa)
```

即 A 是两份半 tile，SFA 是两个 CTA 各一份完整 page。

后面的：

```text
SM100_UTCCP_4x32dp128bit_2cta
```

要求两个 CTA 的相同 SMEM offset 都具备所需 SFA。若给 non-leader 的 `sfa_m_idx` 加 `valid_m/2`，不仅会丢失完整 page，还可能跨入下一个 `SF_BLOCK_M` 页面。

### 5.5 `full_barrier` 何时 ready

A/SFA loader 和 B/SFB loader 都使用：

```text
full_barriers[stage]
```

leader CTA 登记两 CTA 的 transaction bytes，non-leader CTA arrive。只有两个 CTA 的 A、B、SFA、SFB 全部完成，MMA 才能通过：

```cpp
full_barriers[stage].wait(phase)
```

---

## 6. Weight/SFB Loader Warp

### 6.1 Weight 不需要 global arrival counter

Weight 是本 rank 常驻参数，不依赖 dispatch 或前一级 epilogue，所以 weight loader 取得 `TaskInfo` 后直接进入 K-stage loop。

### 6.2 两个 CTA 使用不同 N block

```text
n_block_idx =
    n_cluster_idx * 2
    + cta_rank
```

Routed weight 地址还要加：

```text
local_expert_idx * shape_n
```

Shared weight 没有 routed expert 维，因此直接按 `n_block_idx` 寻址。

SFB 的 `sfb_n_idx` 已包含 CTA rank 对应的 `n_block_idx`，所以两个 CTA 自然读取不同的 weight SF。

### 6.3 每个 K stage

```text
wait empty_barrier
  -> TMA weight B
  -> TMA SFB
  -> full_barrier producer arrival
```

A/SFA 与 B/SFB 是同一 SMEM stage 的两个生产者；MMA 消费后统一通过 `empty_barrier` 释放。

---

## 7. MMA Issue Warp

只有 leader CTA 的 warp 6 发出 2-CTA UMMA。

### 7.1 为什么代码里 A/B 看起来是反的

逻辑 GEMM 是：

```text
activation [M, K] * weight^T [K, N]
```

当前 UMMA 路径交换 A/B，使硬件视角变成：

```text
weight [N, K] * activation^T [K, M]
```

所以：

```text
UMMA_M = 256
UMMA_N = BLOCK_M
```

TMEM accumulator 的逻辑视角也变为 `[N, M]`，这就是 epilogue 后续需要特殊 fragment 重排的根源。

### 7.2 SMEM stage 流水线

MMA 对每个 K stage：

1. 等 `full_barriers[stage]`；
2. 用 UTCCP 将 SFA/SFB 从 SMEM 搬到 TMEM SF 区；
3. 发出 UMMA；
4. commit 时 multicast arrive `empty_barriers[stage]`。

从这一刻开始，两个 loader 可以覆盖该 SMEM stage。

### 7.3 TMEM accumulator 流水线

一个完整 `TaskInfo` 的所有 K blocks 共用同一个：

```text
accum_stage_idx
```

开始前等：

```text
tmem_empty_barriers[accum_stage_idx]
```

UMMA 直接在 TMEM 中累加，不存在“先在寄存器算完整，再复制进 TMEM”的额外步骤。

最后一个 K block commit 时，同时 arrive：

```text
tmem_full_barriers[accum_stage_idx]
```

把完整 accumulator tile 交给 epilogue。

### 7.4 为什么 SMEM stage 与 TMEM stage 的轮转粒度不同

```text
SMEM stage:
    每个 K block 轮转一次

TMEM accumulator stage:
    每个完整 GEMM tile 轮转一次
```

因为一个 GEMM tile 的所有 K blocks 要累加到同一个 accumulator，而 A/B operand 每个 K block 都不同。

---

## 8. Epilogue Warps

### 8.1 统一入口

epilogue 对每个任务先等：

```text
tmem_full_barriers[accum_stage_idx]
```

然后释放对应 schedule slot：

```cpp
scheduler.release_task_info();
```

这里释放的是 `TaskInfo` 双缓冲，不是 TMEM。TMEM 要等最后一次 TMEM load 后才能释放。

### 8.2 Linear1 epilogue 的目标

输入：

```text
TMEM 中的 FP32 gate/up accumulator
```

输出：

```text
FP8 l2_acts
UE8M0 l2_acts_sf
l2_full_count
l1_empty_count
```

数学上：

```text
output =
    topk_weight * SiLU(gate) * up
```

SharedLinear1 的 weight 因子为 1。

### 8.3 覆盖 L2 ring 前先等上一代释放

Routed L1 epilogue 在写 L2 ring 前等：

```text
l2_empty_count[ring_block_idx]
    == generation * kNumL2BlockNs
```

这是 L2 ring 的生产者侧 back-pressure。

### 8.4 从 TMEM 取 gate/up

由于 A/B 交换，TMEM 的 M/N 视角与逻辑 GEMM 相反。epilogue 使用：

```text
SM100_TMEM_LOAD_16dp256b1x
```

按硬件 fragment 读取，再恢复 gate/up 配对。

当前权重预处理已经把 gate/up 以固定粒度交错，因此 epilogue 可按对应 fragment 位置取出两组值。

### 8.5 SwiGLU、clamp 和 top-k weight

对 routed 行，epilogue 从物理 ring 读取：

```text
l1_topk_weights[ring_m_idx + row]
```

随后：

```text
gate = SiLU(clamp(gate))
up   = clamp(up)
act  = gate * up * topk_weight
```

`kFastMath` 决定 sigmoid 中倒数和指数使用快速实现还是更精确路径。

### 8.6 Amax reduction 与 FP8 scale

每个线程先计算自己的局部 amax，再进行：

```text
线程局部
  -> warp 内小组 reduce
  -> warp-pair 通过 shared memory 交换
```

得到对应 token 行的 amax 后计算：

```text
UE8M0 SF
SF inverse
```

再将激活转换为 E4M3。

### 8.7 为什么需要 STSM + TMA store

TMEM fragment 的 lane 分布不能直接作为 row-major L2 activation。epilogue 先用 STSM 将量化后的 FP8 fragment 重排到带 swizzle 的 shared-memory staging，再由 TMA store 写入 L2 activation。

每个 L1 `BLOCK_N=128` 输入 tile 在 SwiGLU 后只产生：

```text
BLOCK_N / 2 = 64
```

个输出通道，因此 L1 output descriptor 使用半宽 tile 和半宽 swizzle。

### 8.8 L2 SF 的地址

L2 SF 同样以：

```text
[packed-K group][SF ring token]
```

组织。Routed 路径使用：

```text
ring block -> block_idx
```

Shared 路径使用完整 shared block index。

每个 `SF_BLOCK_M` 页面内部仍使用 `transform_sf_token_idx()` 对 128-row group 做转置，以匹配后面 SFA loader + UTCCP 的读取方式。

### 8.9 L1 epilogue 如何发布 L2、释放 L1

所有 L1 output TMA stores 完成后：

```text
Routed:
    l2_full_count[ring] += 1
    l1_empty_count[ring] += 1

Shared:
    shared_l2_full_count[block] += 1
```

每个实际 L1 N-block 增加一次。L2 loader 要等当前 generation 的全部 L1 N-block，所以不会读到只完成一部分通道的 L2 activation。

### 8.10 Linear2 epilogue 的目标

L2 不再做 activation 或量化。它需要：

1. 从 TMEM 读取 FP32 accumulator；
2. 转换为 BF16；
3. 在 shared memory 中重排；
4. 找到原始 token 身份；
5. 远端写入 combine slot。

### 8.11 为什么一进入 L2 epilogue 就能增加 `l2_empty_count`

进入 L2 epilogue 时，L2 GEMM 已经完成，activation input 已被 loader/MMA 消费，不再需要保留。

因此 routed 路径可以先：

```text
l2_empty_count[ring] += 1
```

这只释放 L2 input ring，与后面的 TMEM load、BF16 转换和远端 combine 写是否完成无关。

### 8.12 Routed 与 shared 的写回目标

Routed：

```text
metadata =
    token_src_metadata[pool_token_idx]

dst_rank  = metadata.rank_idx
dst_token = metadata.token_idx
dst_slot  = metadata.topk_idx
```

Shared：

```text
dst_rank  = current rank
dst_token = original token
dst_slot  = kNumTopk
```

最终写：

```text
combine_token_buffer[dst_slot][dst_token]
```

### 8.13 TMEM 什么时候释放

epilogue 在最后一次需要的 TMEM load 后 arrive：

```text
tmem_empty_barriers[accum_stage_idx]
```

它可能早于：

- L1 全部 TMA stores 完成；
- L2 全部远端 stores 完成。

因为后续数据已经位于寄存器或 SMEM，不再依赖 TMEM。

### 8.14 Combine 前的 NVLink barrier

所有 L2 tasks 结束并释放 TMEM 后，epilogue 执行：

```text
nvlink_barrier<
    kBeforeCombineReduceBarrierTag>
```

它包含 grid 内同步、跨 rank signal 和必要的 system-scope 可见性。只有此后，源 rank 才能确信所有 partial results 都已写入本地 combine buffer。

### 8.15 Combine 如何本地求和

epilogue warps 改为按原始 token 做 grid-stride 遍历。

每个 token 的有效 mask 来自：

- `topk_idx >= 0` 的 routed slots；
- 可选 shared slot。

对每个 hidden chunk：

```text
TMA load slot 0
  -> 预取 slot 1
  -> FP32 寄存器累加 slot 0
  -> 预取 slot 2
  -> 累加 slot 1
  -> ...
  -> BF16
  -> TMA store y
```

代码只选择一个或两个 hidden chunks，以同时满足 shared-memory 和寄存器预算。

---

## 9. 从 Warmup 开始看完整时间线

把前面的机制重新串成时间线：

```text
Kernel 初始化
  |
  +--> SharedLinear1 tasks（可选）
  |
  +--> Dispatch 统计路由、写 metadata
           |
           v
      NVLink barrier
           |
           v
      Dispatch pull
      wait l1_empty -> 写 L1 ring -> l1_full
           |
           v
Scheduler 读 expert_recv_count_sum
           |
           v
L1 warmup waves
           |
           v
稳态 L2 / L1 交叉领取
  |
  +--> L1 loader 等 l1_full
  |      -> A/SFA + B/SFB
  |      -> full_barrier
  |      -> UMMA
  |      -> tmem_full
  |      -> L1 epilogue
  |      -> wait l2_empty
  |      -> SwiGLU + FP8 TMA store
  |      -> l2_full + l1_empty
  |
  +--> L2 task 先过“所需 L1 tasks 已领取”控制门
         -> L2 loader 再过 l2_full 数据门
         -> A/SFA + B/SFB
         -> UMMA
         -> L2 epilogue
         -> l2_empty
         -> remote combine write
           |
           v
SharedLinear2 tasks（可选）
           |
           v
TaskInfo sentinel
           |
           v
所有 rank 的 L2 remote writes 完成
           |
           v
NVLink barrier
           |
           +--> Dispatch cleanup
           |
           +--> Combine 本地 reduce
           |
           v
          y
```

这张图中有两种重叠：

1. 不同 GEMM task 通过 SMEM/TMEM 多 stage 流水线重叠；
2. dispatch cleanup 与 combine reduction 重叠。

---

## 10. 最后收束：当前实现的四个核心设计

### 10.1 对称内存解决跨 rank 直接寻址

所有 rank 保持相同 buffer layout，kernel 用相同 offset 映射远端地址，从而在单 kernel 内完成 metadata push、payload pull 和 result push。

### 10.2 逻辑 pool 与物理 ring 解耦

逻辑 pool 保留全轮身份与调度顺序；物理 ring 只保留同时存活的 activation window。这样显存需求由最坏 routed rows 降为最大 live set。

### 10.3 Scheduler 把任务依赖与数据依赖分开

```text
TaskInfo 发布门
    防止 L2 等尚未领取的 L1

full-count 数据门
    防止 loader 读取尚未写完的 activation
```

warmup 和稳态交叉调度负责前者，`l1/l2_full_count` 负责后者。

### 10.4 多级 full/empty 把不同生命周期接起来

```text
TaskInfo full/empty
    管任务消息

SMEM full/empty
    管 A/B/SF stage

TMEM full/empty
    管 accumulator stage

L1/L2 ring full/empty
    管 global-memory 中间激活

NVLink barrier
    管跨 rank 阶段边界
```

MegaMoE 的性能并不来自某一条孤立指令，而是这些不同粒度的生产者—消费者协议共同把：

```text
通信
GEMM
SwiGLU
量化
远端写回
本地 reduce
```

压进同一个长期驻留 kernel，并尽量让每一层都只等待自己真正依赖的数据。

---

## 当前代码导航

| 主题 | 代码位置 |
| --- | --- |
| ring 容量规划 | `csrc/apis/mega.hpp:56` |
| workspace / buffer 布局 | `deep_gemm/include/deep_gemm/layout/mega_moe.cuh:46`、`:358` |
| warmup 与 live-block 公式 | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:15`、`:47` |
| `TaskInfo` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:91` |
| 动态任务领取 | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:316` |
| barrier 初始化 | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:230` |
| dispatch metadata | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:328` |
| dispatch pull / SF ring | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:414` |
| token/SFA loader | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:669` |
| weight/SFB loader | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:735` |
| MMA issue | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:794` |
| scheduler warp | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:920` |
| L1/L2 epilogue | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:927` |
| combine barrier / reduce | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1313` |
