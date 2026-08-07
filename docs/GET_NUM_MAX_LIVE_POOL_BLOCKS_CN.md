# `get_num_max_live_pool_blocks` 计算逻辑

## 1. 函数解决什么问题

`get_num_max_live_pool_blocks()` 用来估算 Mega MoE 调度过程中：

> 最多会有多少个 routed M blocks 已经进入 L1/L2 流水，但尚未被 L2 完全消费。

这个值随后用于计算 routed activation ring buffer 的容量：

```text
candidate_ring_tokens
    = max_live_pool_blocks × BLOCK_M
```

host 会对所有候选 `BLOCK_M` 分别计算，取最大值后按 384 对齐：

```text
num_ring_tokens
    = align(
        max_over_BLOCK_M(
            get_num_max_live_pool_blocks(...) × BLOCK_M
        ),
        384
      )
```

函数位于：

```text
deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
```

---

## 2. 四个核心变量

为了简化公式，定义：

```text
P  = num_total_m_blocks
S  = num_sms / 2
C1 = intermediate_hidden / 128
C2 = hidden / 256
```

它们的含义如下。

### 2.1 `P`：总逻辑 M block 数

```text
P = num_total_m_blocks
```

它表示当前 rank 上所有 routed experts 的 M blocks 总数：

```text
P = Σ ceil(tokens_of_expert_e / BLOCK_M)
```

host 做最坏容量估算时使用一个保守上界：

```text
P ≈ ceil(max_routed_tokens / BLOCK_M)
    + experts_per_rank
```

### 2.2 `S`：2-CTA cluster 数

SM100 Mega MoE 使用两个 CTA 组成一个计算 cluster，因此：

```text
S = num_sms / 2
```

一个全局调度 wave 中，最多有 `S` 个 cluster tasks 同时被领取。

### 2.3 `C1`：一个 M block 的 L1 cluster tasks 数

L1 输出形状为：

```text
[BLOCK_M, 2 × intermediate_hidden]
```

每个 CTA 处理一个 `BLOCK_N=128` 的 N block，两个 CTA 组成一个 cluster，因此：

```text
C1
  = (2 × intermediate_hidden) / (2 × 128)
  = intermediate_hidden / 128
```

也就是说，一个 routed M block 的完整 L1 需要 `C1` 个 cluster tasks。

### 2.4 `C2`：一个 M block 的 L2 cluster tasks 数

L2 输出形状为：

```text
[BLOCK_M, hidden]
```

同样每个 cluster 处理两个 `BLOCK_N=128` 的 N blocks，因此：

```text
C2
  = hidden / (2 × 128)
  = hidden / 256
```

一个 routed M block 的完整 L2 需要 `C2` 个 cluster tasks。

---

## 3. 最终公式

函数最终返回：

```text
max_live_pool_blocks
    = min(
        P,
        warmup_live_blocks
        + frontier_growth
        + wave_margin
      )
```

三项分别表示：

```text
warmup_live_blocks
    warmup 阶段提前产生的存活 M blocks

frontier_growth
    稳态阶段 L1/L2 前沿速度不同造成的额外积压

wave_margin
    不完整 wave、任务领取偏差和流水延迟的保守余量
```

下面逐项推导。

---

## 4. 第一项：`warmup_live_blocks`

### 4.1 第一波 L2 会覆盖多少个 M blocks

第一波最多同时领取 `S` 个 L2 tasks。

每个 M block 包含 `C2` 个 L2 tasks，因此第一波 L2 最多触及：

```text
first_l2_m_blocks
    = ceil(S / C2)
```

个 M blocks。

在这些 L2 tasks 被调度前，对应 M blocks 的全部 L1 tasks 必须已经被领取，否则所有 scheduler 可能同时等待 L1 task counter，形成死锁。

这些 M blocks 共需要：

```text
first_l2_m_blocks × C1
```

个 L1 tasks。

每个 warmup wave 能领取 `S` 个 L1 tasks，因此第一项 warmup 下界为：

```text
W_first
    = ceil(
        first_l2_m_blocks × C1 / S
      )
```

代入 `first_l2_m_blocks`：

```text
W_first
    = ceil(
        ceil(S / C2) × C1 / S
      )
```

### 4.2 稳态 L1/L2 task 数不匹配

稳定态近似按照：

```text
L2 task → L1 task → L2 task → L1 task
```

交错调度。

如果：

```text
C1 > C2
```

那么完整调度一个 M block 的 L2 期间，只能插入大约 `C2` 个 L1 tasks，但下一个 M block 需要 `C1` 个 L1 tasks。

每推进一个 M block 的缺口为：

```text
diff = C1 - C2
```

统一写成：

```text
diff = max(C1 - C2, 0)
```

为了保证最后一个 M block 的 L1 也能在其 L2 前全部领取，warmup 需要预付：

```text
C1 + (P - 1) × diff
```

个 L1 tasks。

因此第二项 warmup 下界是：

```text
W_interleave
    = ceil(
        [C1 + (P - 1) × diff] / S
      ) + 1
```

末尾额外的 `+1` 是一个 CTA-pair wave，用来覆盖不完整 wave 的取整误差。

### 4.3 得到实际 warmup waves

满足正确性所需的最小 warmup waves：

```text
W_min = max(W_first, W_interleave)
```

但 warmup 不可能超过全部 L1 tasks 本身需要的 wave 数：

```text
total_l1_waves
    = ceil(P × C1 / S)
```

所以实际使用：

```text
W = min(W_min, total_l1_waves)
```

### 4.4 Warmup waves 转换成存活 M blocks

warmup 最多领取：

```text
warmup_l1_tasks
    = min(W × S, P × C1)
```

个 L1 cluster tasks。

一个 M block 需要 `C1` 个 L1 tasks，因此 warmup 对应的存活 M blocks 上界为：

```text
warmup_live_blocks
    = ceil(warmup_l1_tasks / C1)
```

完整写法：

```text
warmup_live_blocks
    = ceil(
        min(W × S, P × C1) / C1
      )
```

---

## 5. 第二项：`frontier_growth`

稳定态按 task 粒度近似 1:1 交错 L1 和 L2，但两者按 M-block 计算的前进速度不同：

```text
L1 每 C1 个 tasks 前进一个 M block
L2 每 C2 个 tasks 前进一个 M block
```

### 5.1 当 `C1 >= C2`

L1 前进速度不快于 L2：

```text
1 / C1 <= 1 / C2
```

因此稳定态不会由于 L1 生产过快继续扩大 activation 积压：

```text
frontier_growth = 0
```

如果 `C1 > C2` 带来调度 task 缺口，该缺口已经在 warmup 中通过：

```text
(P - 1) × (C1 - C2)
```

提前预付。

### 5.2 当 `C2 > C1`

此时 L1 在 M-block 空间前进得更快：

```text
1 / C1 > 1 / C2
```

也就是说，L1 产生 L2 activation blocks 的速度高于 L2 消费这些 blocks 的速度。

代码使用下面的保守闭式上界估算额外积压：

```text
frontier_growth
    = ceil(
        P × (C2 - C1) / C2
      )
```

统一写成：

```text
frontier_growth =
    C2 > C1
        ? ceil(P × (C2 - C1) / C2)
        : 0
```

这不是逐 task 精确模拟结果，而是一个计算成本更低的保守上界。

---

## 6. 第三项：`wave_margin`

代码额外增加：

```text
wave_margin
    = ceil(
        S / min(C1, C2)
      )
```

它表示一个全局 wave 最多可能跨越多少个 M blocks。

这部分用于覆盖：

- 最后一个 wave 不完整；
- 不同 scheduler 原子领取任务的先后偏差；
- TaskInfo、TMA、MMA 和 epilogue 之间的流水延迟；
- 闭式 `frontier_growth` 与真实离散调度之间的取整误差。

---

## 7. 合并结果

最终：

```text
max_live_pool_blocks
    = min(
        P,
        warmup_live_blocks
        + frontier_growth
        + wave_margin
      )
```

之所以再与 `P` 取最小值，是因为同时存活的 M blocks 不可能超过总逻辑 M blocks。

如果计算出的 live window 接近或超过 `P`，ring buffer 就退化成接近完整 pool。

---

## 8. 与源码一一对应的伪代码

```cpp
int get_num_max_live_pool_blocks(
    int P,
    int num_sms,
    int hidden,
    int intermediate_hidden) {

    // 2-CTA cluster 和 BLOCK_N=128
    int S  = num_sms / 2;
    int C1 = intermediate_hidden / 128;
    int C2 = hidden / 256;

    // 第一波 L2 所需的 L1 warmup
    int first_l2_m_blocks = ceil_div(S, C2);
    int W_first =
        ceil_div(first_l2_m_blocks * C1, S);

    // 稳态 L1/L2 task 数差异所需的 warmup
    int diff = max(C1 - C2, 0);
    int W_interleave =
        ceil_div(C1 + (P - 1) * diff, S) + 1;

    int W_min = max(W_first, W_interleave);

    // Warmup 不超过全部 L1 waves
    int total_l1_waves = ceil_div(P * C1, S);
    int W = min(W_min, total_l1_waves);

    // Warmup 对应的存活 M blocks
    int warmup_l1_tasks = min(W * S, P * C1);
    int warmup_live_blocks =
        ceil_div(warmup_l1_tasks, C1);

    // 稳态 L1 前沿快于 L2 时的额外积压
    int frontier_growth =
        C2 > C1
            ? ceil_div(P * (C2 - C1), C2)
            : 0;

    // 一个全局 wave 的安全余量
    int wave_margin =
        ceil_div(S, min(C1, C2));

    return min(
        P,
        warmup_live_blocks
        + frontier_growth
        + wave_margin
    );
}
```

---

## 9. 一个数值例子

假设：

```text
P  = 100
S  = 60
C1 = 16
C2 = 28
```

这对应：

```text
num_sms = 120
intermediate_hidden = 2048
hidden = 7168
```

### 9.1 Warmup

第一波 L2 覆盖：

```text
first_l2_m_blocks
    = ceil(60 / 28)
    = 3
```

需要的第一项 warmup：

```text
W_first
    = ceil(3 × 16 / 60)
    = 1
```

因为：

```text
C1 < C2
```

所以：

```text
diff = 0
```

第二项：

```text
W_interleave
    = ceil(16 / 60) + 1
    = 2
```

实际 warmup：

```text
W = max(1, 2) = 2
```

warmup 发出：

```text
2 × 60 = 120
```

个 L1 tasks，对应：

```text
warmup_live_blocks
    = ceil(120 / 16)
    = 8
```

### 9.2 稳态前沿增长

因为：

```text
C2 > C1
```

所以：

```text
frontier_growth
    = ceil(100 × (28 - 16) / 28)
    = 43
```

### 9.3 Wave margin

```text
wave_margin
    = ceil(60 / min(16, 28))
    = ceil(60 / 16)
    = 4
```

### 9.4 最终结果

```text
max_live_pool_blocks
    = min(100, 8 + 43 + 4)
    = 55
```

如果当前候选：

```text
BLOCK_M = 64
```

则该候选要求：

```text
candidate_ring_tokens
    = 55 × 64
    = 3520
```

个 token slots。

最终 `num_ring_tokens` 还要与其他候选 `BLOCK_M` 的结果取最大值，再按 384 对齐。

---

## 10. 一句话总结

`get_num_max_live_pool_blocks()` 计算的是：

```text
最大同时存活 M blocks
    =
    warmup 提前产生的 blocks
    + 稳态 L1 比 L2 前进更快造成的积压
    + 一个全局 wave 的安全余量
```

再用总逻辑 M blocks `P` 封顶：

```text
return min(
    P,
    warmup_live_blocks
    + frontier_growth
    + wave_margin
);
```
