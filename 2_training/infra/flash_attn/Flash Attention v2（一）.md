# FlashAttention v2: 极致压榨GPU性能的工程艺术

- FlashAttention 1/2/3
    - [五张图片看懂Flash Attention v1（一）](https://zhuanlan.zhihu.com/p/1936750158621676144)
    - [Flash Attention v2（一）](https://zhuanlan.zhihu.com/p/1936809531221972067)
    - [Flash Attention v3（一）](https://zhuanlan.zhihu.com/p/1936809729683861528)

FlashAttention-1 已经很厉害了，但还不够完美。它的瓶颈从“内存访问”转移到了“计算”，但由于工作调度不够优化，GPU 的计算单元没有被充分利用。

**FlashAttention-2 的核心思想是：在 FlashAttention-1 的分块思想之上，进行更精细的“工作调度 (Work Scheduling)”，以最大化 GPU 的并行计算效率。**

#### 核心公式思想（概念层面）

FlashAttention-2 并没有改变注意力的数学本质，所以基础公式还是那三个。它的改进体现在**算法实现层面**，而不是数学公式层面。我们可以用伪代码来理解它的核心逻辑：

```
// O 是最终输出, l 和 m 是在线 Softmax 的统计量
O = 0, l = 0, m = -infinity

// 把 K 和 V 按列切块
for j = 1 to num_blocks(K):
  // 从 HBM 加载一小块 K 和 V 到 SRAM
  K_j, V_j = load_from_HBM(K_block_j, V_block_j)

  // 把 Q 按行切块 (这是 FA-2 的改进点之一，并行处理)
  for i = 1 to num_blocks(Q):
    // 从 HBM 加载一小块 Q 和之前的 O, l, m 到 SRAM
    Q_i, O_i, l_i, m_i = load_from_HBM(Q_block_i, O_i, l_i, m_i)

    // 在 SRAM 中进行核心计算
    S_ij = Q_i * K_j^T
    
    // -- 在线 Softmax 的核心 --
    m_i_new = max(m_i, row_max(S_ij))
    P_ij_scaled = exp(S_ij - m_i_new)
    l_i_new = exp(m_i - m_i_new) * l_i + row_sum(P_ij_scaled)
    
    // 更新输出 O_i
    O_i = ( (exp(m_i - m_i_new) * l_i) * O_i + P_ij_scaled * V_j ) / l_i_new

    // -- 更新统计量 --
    m_i = m_i_new
    l_i = l_i_new

    // 将更新后的 O_i, l_i, m_i 写回 HBM
    // (注意：这里只写回很小的 O_i 和统计量，而不是巨大的 S_ij)
    write_to_HBM(O_i, l_i, m_i)
```

**FlashAttention-2 的主要改进点：**

- **在Head维度上增加并行度**：
    *   FlashAttention-1 的并行主要是在序列长度 `N` 这个维度上。一个线程块（Thread Block）里的不同线程束（Warps）处理 `Q` 的不同行。
    *   **FlashAttention-2 做了调整**：它让一个线程块内的不同线程束不仅可以处理不同行，还可以**共同处理同一行但不同的特征维度 `d`**。
    *   **通俗理解**：假设有32个工人（线程束）要粉刷一面很长的墙（序列长度N）。FA1 的方法是给每个工人分配一小段墙去刷。如果墙很窄（特征维度 `d` 很小），工人大部分时间都在走来走去换位置，而不是在刷墙。FA2 的方法是，对于比较宽的墙（`d` 较大），可以让几个工人并排站，同时粉刷同一段墙的不同高度，这样效率更高。这显著提高了当 `d` 比较大时的计算单元利用率。
- **优化的工作分区**：
    *   FlashAttention-2 重新设计了线程块（Thread Block）和线程束（Warp）之间的工作分配。它在不同硬件（如 A100 和 H100）上做了微调，使得数据在 SRAM 中的传输和计算更加均衡，减少了线程之间的等待和同步开销。
- **减少非矩阵乘法（non-MATMUL）的计算量**：
    *   在上面的伪代码中，有很多缩放、指数、求和等操作。这些操作虽然不慢，但也会占用计算资源。FlashAttention-2 通过调整算法流程，减少了这类操作的比例，让更多的计算时间花在最高效的矩阵乘法（MATMUL）上。

### 核心实验结果

FlashAttention-2 的实验结果非常亮眼，证明了这些优化的巨大价值。

- **速度大幅提升**：
    *   论文的图表显示，与 PyTorch 的标准实现相比，FlashAttention-2 最高可达 **9倍** 的速度提升。
    *   与它的前身 FlashAttention-1 相比，它也有显著的速度优势，通常能快 **2倍** 左右。
    *   这个加速效果在最新的 H100 GPU 上比在 A100 上更明显，因为它更好地利用了 H100 更强的计算能力。
- **实现端到端的模型训练加速**：
    *   这种底层的算子优化，直接带来了整个大模型（如 GPT-3 风格模型）训练速度的提升。
    *   实验表明，使用 FlashAttention-2 训练一个 GPT-3 模型，总训练速度可以提升高达 **15%**，这在动辄花费数百万美元的训练中，可以节省大量的时间和金钱。
- **支持更长的序列**：
    *   这继承自 FlashAttention-1 的优点。由于内存占用是 `O(N)` 而不是 `O(N^2)`，模型可以处理更长的上下文序列，这对于需要长依赖关系的任务（如长文档问答、代码生成）至关重要。

### 总结

-   **核心问题**：标准注意力机制因产生巨大的 `N x N` 中间矩阵，导致大量缓慢的 HBM 内存读写，成为性能瓶颈。
-   **FlashAttention-1 的方案**：通过**分块 (Tiling)** 和 **在线 Softmax** 技巧，避免生成完整的 `N x N` 矩阵，将计算瓶颈从内存转移到计算。
-   **FlashAttention-2 的核心贡献**：在 FlashAttention-1 的基础上，通过**更优化的工作调度**，包括：
    - 在特征维度 `d` 上增加并行处理。
    - 调整线程块和线程束的工作分配。
    - 减少非核心计算的开销。
    
    从而**极大地提高了 GPU 计算单元的利用率**，实现了比 FlashAttention-1 快约2倍的速度，并最终显著加速了整个大模型的训练。

简单来说，**FlashAttention-1 是算法层面的革命，FlashAttention-2 是工程实现和硬件优化层面的极致压榨**，两者结合，成为了当前大模型训练中注意力计算的事实标准。