# Native Sparse Attention (NSA) 原理图解

本文将通过图解的方式，详细介绍 Native Sparse Attention (NSA) 的工作原理。NSA 是一种稀疏注意力机制，它通过组合三种不同的注意力模式来近似模拟标准的全量注意力（Full Attention），从而在保持模型性能的同时，显著降低计算复杂度和内存消耗。

## 1. 回顾：标准自注意力（Standard Self-Attention）

在深入了解 NSA 之前，我们先回顾一下标准自注意力机制的计算流程。

![normal_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/normal_attention.png)

**图解说明:**

上图展示了在自回归（Autoregressive）任务中预测下一个 token 的标准注意力计算过程。

*   **输入 (Input Tokens):** 序列为 `1, +, 1, =, 2`，长度 `seq_len=5`。我们的目标是预测接下来的 `?` token。
*   **Q, K, V 向量:**
    *   **Query (q):** 查询向量 `q` (蓝色) 来自当前需要预测的位置（即 `?` 的位置）。它代表了“我需要寻找什么样的信息来做出预测”。
    *   **Key (K) & Value (V):** 键向量 `K` (绿色) 和值向量 `V` (黄色) 来自于上下文中的所有 `input_tokens`（包括 `1, +, 1, =, 2`）。`K` 向量用于和 `q` 向量匹配，衡量每个上下文 token 的重要性；`V` 向量则包含了每个 token 的实际信息。
*   **计算过程:**
    1.  查询向量 `q` 与上下文中 **每一个** 键向量 `K` 进行点积运算，得到注意力分数（Attention Score）。
    2.  将这些分数通过 `softmax` 函数进行归一化，得到注意力权重（Attention Weights）。
    3.  最后，将这些权重与对应的值向量 `V` 进行加权求和，得到最终的输出向量。

**核心痛点:** 这种标准的自注意力机制需要计算 `q` 与所有 `K` 之间的关系，其计算复杂度为 O(N²)，其中 N 是序列长度。当序列非常长时，计算量和内存占用会急剧增加，成为性能瓶颈。

## 2. Native Sparse Attention (NSA) 的整体计算流程

为了解决标准注意力的性能问题，NSA 提出了一种巧妙的分解策略。

![nsa_total](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/nsa_total.png)

**图解说明:**

NSA 的核心思想是，最终的注意力输出并不是通过一次性计算所有 token 得到的，而是由三个独立的、计算更高效的注意力“分支”的输出相加而成。

**最终输出 (?) = O1 + O2 + O3**

这三个分支分别是：
*   **O1: Compressed Attention (压缩注意力):** 关注全局的、低分辨率的概要信息。
*   **O2: Top-n Attention (Top-n 注意力):** 关注内容上最相关的几个信息块。
*   **O3: Sliding Window Attention (滑动窗口注意力):** 关注局部上下文信息。

通过将这三种不同粒度的信息进行融合，NSA 旨在用更低的计算成本来近似模拟标准全量注意力的效果。下面我们来逐一拆解这三个分支。

### O1: 压缩注意力分支 (Compressed Attention)

这个分支的目标是高效地捕获长距离依赖和全局信息。

![compress_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/compress_attention.png)

**图解说明:**

*   **分块与压缩:**
    1.  首先，将序列中的 `K` 和 `V` 向量进行分块。图中示例是将每 3 个 `kv` 对划分为一个块（由红色虚线框标出）。
    2.  这些块是重叠的，`stride=2` 意味着每次分块时，窗口向右移动 2 个 token。
    3.  对每个块内的 `K` 和 `V` 向量执行一个“压缩”操作（例如平均池化），将 3 个 `kv` 对压缩成 1 个新的、信息更浓缩的 `kv` 对（图中用深绿色和深棕色表示）。
*   **注意力计算:**
    *   查询向量 `q` 不再与原始的、长序列的 `K` 向量进行计算，而是与这些被 **压缩后** 的、数量更少的 `K` 向量进行注意力计算。
    *   这样，计算量从 `seq_len` 级别降低到了 `seq_len / stride` 级别，极大地提升了效率。

**作用:** 这个分支让模型能够以一种“鸟瞰”的方式快速掌握整个序列的概要信息，捕捉全局上下文。

### O2: Top-n 注意力分支 (Top-n Attention)

这个分支的目标是让注意力聚焦于与当前查询最相关的信息块，无论它们在序列中的哪个位置。

![top_n_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/top_n_attention.png)

**图解说明:**

*   **分块与评分:**
    1.  与压缩分支类似，首先对 `K` 和 `V` 向量进行分块（图中同样是每 3 个 `kv` 一块，`stride=2`）。
    2.  计算每个块的重要性分数。图中示例是计算 `q` 与块内所有 `K` 的 **平均注意力分数**。例如，第一个块的平均分是 0.4，第二个是 0.2，第三个是 0.4。
*   **选择与计算:**
    1.  根据评分，选择分数最高的 `Top-n` 个块。在图中，`n=2`，因此选择了得分均为 0.4 的第一个和第三个块。
    2.  然后，`q` 只与这些被选中的“高相关性”块内的 `K` 和 `V` 向量进行标准的注意力计算。
    3.  未被选中的块（如中间得分 0.2 的块）则被忽略，从而节省了计算资源。

**作用:** 这个分支实现了基于内容的稀疏性，确保模型能够将计算能力集中在最关键的信息上，而不是均匀地分散在整个序列上。

### O3: 滑动窗口注意力分支 (Sliding Window Attention)

这个分支专注于捕捉局部的、相邻 token 之间的关系，这对于理解语法结构和短期依赖至关重要。

![sliding_window_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/sliding_window_attention.png)

**图解说明:**

*   **局部关注:**
    *   查询向量 `q` 只关注其附近的一个固定大小的“窗口”内的 `K` 和 `V` 向量。
    *   在图中，`q` 只与它前面最近的 3 个 token 的 `K` 和 `V` 进行注意力计算。
*   **高效计算:**
    *   这种方法的计算复杂度是 O(N * W)，其中 W 是固定的窗口大小。由于 W 通常远小于序列长度 N，所以这个分支的计算非常高效。

**作用:** 这个分支类似于卷积神经网络（CNN）中的卷积核，专门处理局部模式，弥补了其他稀疏模式可能忽略的精细局部信息。

## 总结

Native Sparse Attention (NSA) 通过将复杂的全量注意力分解为三个并行且互补的分支，实现了一种高效的注意力近似：

1.  **压缩注意力 (O1):** 提供全局的、概要性的上下文。
2.  **Top-n 注意力 (O2):** 聚焦于内容上最相关的关键信息。
3.  **滑动窗口注意力 (O3):** 捕捉局部的、相邻的依赖关系。

最终，将这三个分支的输出简单相加，模型便能以远低于 O(N²) 的成本，融合来自不同维度和尺度的信息，从而在长序列任务上实现性能和效率的双重提升。