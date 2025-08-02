# 前言

回看读博期间研究的大模型技术，不得不说视野非常小。自从进入大厂基座模型组，深感工业界已经领先学术界一大截。开此贴记录工业界真实大模型训练和推理的方方面面，希望能够对在读的同学一些帮助，也是对自己学习知识的总结，帖子会不断更新。

同步的Github更新链接为 [https://github.com/monster119120/Industrial_LLM_tutorial](https://github.com/monster119120/Industrial_LLM_tutorial)，感兴趣的小伙伴可以一起创作。

为了方便大家收到更新，已开知乎专栏[大模型全栈记录](https://www.zhihu.com/column/c_1934673782448062552)

同步的微信公众号
**大模型全栈开发**
欢迎关注。

个人经历记录记录到了[博士之路](https://www.zhihu.com/column/c_1934959737918697853)，欢迎关注。

# 摘要

工业界的大模型，涉及的部分主要有：**数据**、**训练**、**推理部署**、**大模型应用**四个方面。

---

## 1. 大模型数据

- 原始数据爬取
    - 网页
    - Arxiv
    - Github
    - 书籍
    - 等等

- 数据打分
    - 基于规则的打分
        - 正则表达式
    - 基于模型的打分
        - 训练质量模型，对每一条数据进行打分

- 数据分类
    - 基于规则分类
        - 关键词
    - 基于模型分类
        - 训练分类模型，对每一条数据进行分类

- 数据去重
  - 基于哈希去重
  - 基于语义相似性去重

- 数据采样
  - 不同domain数据源配比
  - 多样性采样平衡
  - 上采样
  - 下采样
  

- 训练数据合成
    - 短文预训练数据合成
    - 短文SFT数据合成
    - 短文RL数据合成
    - 长文预训练数据合成
    - 长文SFT数据合成
    - 长文RL数据合成

---

## 2. 大模型训练

- 大模型训练——算法
    - 长文
        - [大模型长文训练（一）位置编码基础理论](https://zhuanlan.zhihu.com/p/1933621399240569735)
        - [大模型长文训练（二）长度外推](https://zhuanlan.zhihu.com/p/1934347535641715830)
    - MoE
    - NSA (Native Sparse Attention)
        - [五张图片看懂Native Sparse Attention（一）](https://zhuanlan.zhihu.com/p/1934668007730290968)
    - Attention
    - Scaling Law
    - LoRA
    - Post-training
      - GRPO

- 大模型训练——infra
    - CP，TP，EP，SP，pipeline 等等
    - Megatron
        - [Megatron-LM（一）代码结构分析](https://zhuanlan.zhihu.com/p/1920265831364931803)
        - [Megatron-LM（二）代码运行流程](https://zhuanlan.zhihu.com/p/1920451187829900784)
    - Deepspeed
    - Torchtiton
    - FlashAttention 1/2/3
    - verl
    - 算子优化

---

## 3. 大模型推理部署

- 大模型推理——算法
    - KV cache 裁剪
    - 裁剪
    - 投机采样
    - 量化
    - RAG（Retrieval Augmented Generation）

- 大模型推理——infra
    - PD分离
    - Continues batching
    - Paged Attention
    - Cacheblend
    - PrefixCaching
    - Chunked prefill
    - SGLang
    - vLLM

---

## 4. 大模型应用

- Agent
- MCP
- Deep Research

---

> 本帖持续更新，欢迎关注交流！

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=monster119120/Industrial_LLM_tutorial&type=Date)](https://www.star-history.com/#monster119120/Industrial_LLM_tutorial&Date)