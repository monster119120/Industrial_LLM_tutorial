# Native Sparse Attention

本文简要介绍Native Sparse Attention (NSA) 的原理。

## 标准Self Attention计算流程

![normal_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/normal_attention.png)


## Native Sparse Attention (NSA) 的计算流程

### NSA = O1 + O2 + O3

![nsa_total](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/nsa_total.png)


### O1: Compressed Attention分支
![compress_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/compress_attention.png)


### O2: Top-n Attention分支
![top_n_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/top_n_attention.png)


### O3: Sliding Window Attention分支
![sliding_window_attention](https://github.com/monster119120/Industrial_LLM_tutorial/raw/main/2_training/algo/nsa/sliding_window_attention.png)
