# 大模型型长文训练（三）YaRN代码详解

## 一、问题的根源：模型的“视野”局限

大语言模型（LLM）在训练时，都有一个固定的“上下文窗口”，比如4096个词。这就像它天生只有一个固定大小的视野范围。一旦输入的文本超过这个范围，模型的效果就会急剧下降，就像一个人看得太远就会模糊不清一样。

为了让模型能“看得更远”（处理更长的文本），研究者们提出了多种上下文扩展技术，YaRN就是其中非常出色的一种。

## 二. 核心比喻：一把神奇的橡皮筋尺子

为了彻底理解YaRN，我们先建立一个简单但强大的心智模型：

*   **你 (模型)**：你的任务是精确测量不同长度的**绳子 (文本序列)**。
*   **你的工具**：一把非常精密的**橡皮筋尺子**。
    *   **尺子自然长度**: **1米**。这对应模型的**原始上下文长度** (`original_max_position_embeddings = 4096`)。
    *   **尺子上的刻度**: 在这1米上，有非常密集的**毫米刻度**。用它测量1米内的东西极其精准。这对应原始的**旋转位置编码 (RoPE)**。

---

## 三、YaRN (静态YaRN)：一把被精心改造的尺子

静态YaRN的目标是让模型稳定地处理一个**固定**的更长上下文。

### 理论：

你接到通知，说以后要测量的绳子最长可能有 **4米**。你不想简单粗暴地把它拉长，因为那样量短绳子就不准了。于是你进行了一次**精心的改造**：

*   **做法**：你不是均匀地拉伸整把尺子。
    1.  你保护了尺子**最开始的那一小段（比如前10厘米）**，让它的毫米刻度**保持原样**，不做任何拉伸。因为这部分负责测量最短、最精细的部分。
    2.  对于尺子剩下的部分，你才进行拉伸，让整把尺子最终达到4米长。
    3.  在“未拉伸区”和“拉伸区”之间，你做了一个平滑的过渡。
*   **结果**：你得到了一把全新的、固定为4米长的尺子。这把尺子很神奇：
    *   它的**前端刻度依然非常精密**，量短绳子（比如5厘米）时和原始尺子一样准。
    *   它的**后端刻度被拉伸了**，可以用来测量长绳子。
    *   它是一把经过**一次性、智能化改造**后定型的尺子。

**结论**：YaRN通过**区别对待**尺子的不同部分（高频/低频维度），在扩展长度的同时，最大限度地保留了对短距离（高频信息）的感知精度。

### 代码实现 (`LlamaYaRNScaledRotaryEmbedding`)

在静态YaRN中，所有的改造工作都在模型初始化时一次性完成。

```python
class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        # 1. 记录尺子属性：原始长度、材料特性，以及固定的拉伸倍率(scale)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale # <--- 拉伸倍率在这里被固定！
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 2. “动手改造！”——直接调用核心的yarn()魔法，完成尺子的改造定型
        self.yarn(device)

        # 3. 给改造好的新尺子拍照存档
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # 4. 每次测量，都直接用这把改造好的尺子的快照
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    # 附：共用的yarn()魔法实现
    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs           # 方案B: 保持原始毫米刻度 (外推)
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs) # 方案A: 均匀拉伸所有刻度 (插值)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor
        
        # 按计划混合，生成最终的新刻度
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 计算并记录尺子被拉细的修正值
        self.mscale = float(_yarn_get_mscale(self.scale) * self.attn_factor)
```

---

## 四、动态YaRN：一把能“按需变形”的智能尺子

动态YaRN更进一步，它认为“一次性改造定型”还不够灵活。如果我今天量短绳子，明天量长绳子，能不能让尺子自己适应呢？

### 理论：

你拥有了那把原始的1米橡皮筋尺子，但它被赋予了“按需变形”的魔法。

#### **场景1：测量一根20厘米的短绳子** (`seq_len` 短)
*   **做法**：绳子比尺子短。你**根本不触发变形魔法**，直接用原始的、最精密的毫米刻度去量。
*   **结果**：读数超级精准。

#### **场景2：测量一根2米长的长绳子** (`seq_len` 长)
*   **做法**：绳子比尺子长。你大喊一声“变！”，尺子**立刻当场执行了上面提到的“精心改造”过程**，把自己变成了一把为“2米”这个长度量身定制的、前端精密后端拉伸的尺子。
*   **结果**：你用这把“临时2米尺”完成了测量，既准又长。

#### **场景3：测量另一根1.8米长的绳子** (`seq_len` 也长)
*   **做法**：你记得尺子刚变成过2米形态。既然2米都量得了，1.8米肯定也行。
*   **结果**：你**懒得让它再变一次**了，直接用刚才的2米形态（**复用缓存**）去量这根1.8米的绳子。高效又省力。

**结论**: 动态YaRN不是一次性改造，而是将**YaRN的“精心改造”过程变成了一个可以随时按需触发的技能**。它只在需要的时候，才把尺子变成最适合当前任务的形态。

### 代码实现 (`LlamaDynamicYaRNScaledRotaryEmbedding`)

动态YaRN的代码完美体现了“按需触发”的逻辑。

```python
class LlamaDynamicYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        # 1. 记录尺子属性：原始长度、材料特性等。注意，这里没有固定的 self.scale
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 2. 制造尺子最原始、最精密的“毫米刻度” (inv_freq)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = 1.0 # 初始没有拉伸，粗细修正为1

        # 3. 准备记事本，记下测量过的最长记录
        self.max_seq_len_cached = max_position_embeddings
        
        # 4. 给尺子原始的1米状态，拍一张高清快照 (cos_cached, sin_cached)
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)


    def forward(self, x, seq_len=None):
        # 5. 拿到一根长度为 seq_len 的绳子 (x)

        # 6. 关键决策：这根绳子(seq_len)是不是比我记事本上的记录(self.max_seq_len_cached)还长？
        if seq_len > self.max_seq_len_cached:
            # 是！需要触发变形魔法！

            # a. 更新记事本
            self.max_seq_len_cached = seq_len
            # b. 计算本次变形的目标倍率
            scale = seq_len / self.original_max_position_embeddings
            # c. “动手改造！”—— 调用核心的 yarn() 魔法
            self.yarn(scale, x.device)

            # d. 给新形态的尺子拍照存档，覆盖旧照片
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)

        # 7. 使用当前最合适的尺子形态进行测量
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    # 附：共用的yarn()魔法实现，和静态版本几乎一样，只是scale是动态传入的
    def yarn(self, scale, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor
        
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.mscale = float(_yarn_get_mscale(scale) * self.attn_factor)
```
---

## 五、总结与对比：静态改造 vs 动态变形

现在，我们把两者放在一起，它们的区别就一目了然了。

| 特性 (Feature) | YaRN (静态) | Dynamic YaRN (动态) |
| :--- | :--- | :--- |
| **核心思想** | 对尺子进行**一次性**的精心改造，使其固定成一个更长、但仍保留部分精度的形态。 | 将“精心改造”的过程变成一个**可重复触发的技能**，按需变形。 |
| **`scale` (拉伸倍率)** | **固定的**。在`__init__`中设定一次。 | **动态的**。在`forward`中根据`seq_len`即时计算。 |
| **`yarn()` (改造/变形)** | 在`__init__`中**调用一次**，完成定型。 | 在`forward`中，当遇到**新的更长文本**时**被调用**，完成变形。 |
| **适用场景** | 明确知道要把上下文扩展到某个**固定长度**。 | 需要处理**任意可变长度**的文本，追求最大灵活性。 |
| **尺子比喻** | **改造车间**：把1米尺送进去，出来一把永久的4米特制尺。 | **变形金刚**：尺子平时是1米，遇到长绳子就喊“变形！”，变成最合适的长度。 |

**最终结论**:
*   **YaRN** 是一种先进的 **“改造技术”** ，它定义了如何智能地拉伸尺子。
*   **Dynamic YaRN** 是一种聪明的 **“使用策略”**，它决定了**何时**以及**如何**去使用这个“改造技术”。

两者相辅相成，共同实现了大语言模型超长上下文能力的优雅扩展。