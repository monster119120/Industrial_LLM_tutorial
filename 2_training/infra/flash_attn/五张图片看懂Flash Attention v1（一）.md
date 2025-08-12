# 五张图片看懂Flash Attention v1（一）

- FlashAttention 1/2/3
    - [五张图片看懂Flash Attention v1（一）](https://zhuanlan.zhihu.com/p/1936750158621676144)
    - [Flash Attention v2（一）](https://zhuanlan.zhihu.com/p/1936809531221972067)
    - [Flash Attention v3（一）](https://zhuanlan.zhihu.com/p/1936809729683861528)

笔者在学习Flash Attention时看了很多博客，感觉都很复杂，在此笔者希望能用五张图帮助同学们理解Flash Attention。

## 标准Attention计算

假如输入一段长度为`n`的序列，每两个token之间都需要计算注意力，我们以某个token对其他所有token的注意力计算为例。

![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic0.png)


## Flash Attention v1计算

标准Attention计算的问题在于，需要拿到所有token的注意力分数，然后再计算softmax，如果输入序列长度`n`非常大，显存将无法一次性的存储所有的注意力分数，只能反复的通过读写操作缓解计算存储的压力，最终导致计算时间非常慢。
Flash Attention v1的核心思想是，通过online的方式，把softmax的计算拆解为多个小的计算步骤，从而避免了反复的读写操作，提升计算速度。

### 第一步，初始化输出项和分母项
![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic1.png)


### 第二步，在线Softmax Attention计算
![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic2.png)

---

![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic3.png)

---

![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic4.png)

最终我们就获得了该行的最终输出。为了使计算更加稳定，Flash Attention v1中还引入了一个最大值`m`，避免了softmax计算时的数值溢出，计算逻辑如下：

$$ o = \text{softmax}( q K^T) V =  \displaystyle\sum_{j=0}^{N-1} \frac{e^{q K_j}}{\displaystyle\sum_{j=0}^{N-1} e^{q K_j}} V_j =  \displaystyle\sum_{j=0}^{N-1} \frac{e^{q K_j - m}}{\displaystyle\sum_{j=0}^{N-1} e^{q K_j - m}} V_j $$