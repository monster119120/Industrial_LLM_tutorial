# 五张图片看懂Flash Attention v1（一）

- FlashAttention 1/2/3
    - [五张图片看懂Flash Attention v1（一）](https://zhuanlan.zhihu.com/p/1936750158621676144)
    - [Flash Attention v2（一）](https://zhuanlan.zhihu.com/p/1936809531221972067)
    - [Flash Attention v3（一）](https://zhuanlan.zhihu.com/p/1936809729683861528)

笔者在学习Flash Attention时看了很多博客，感觉都很复杂，在此笔者希望能用五张图帮助同学们理解Flash Attention。

## 标准Attention计算
![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic0.png)


## Flash Attention v1计算

### 第一步，初始化输出项和分母项
![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic1.png)


### 第二步，在线Softmax Attention计算
![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic2.png)

---

![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic3.png)

---

![img](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/infra/flash_attn/flash_attn_v1_pic4.png)

最终我们就获得了该行的最终输出 

