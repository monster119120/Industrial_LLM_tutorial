
# Megatron-LM（三）代码调试指南

Nvidia的Megatron-LM是业界的主流大模型训练框架，然而这个框架非常复杂，为了能够实际上手代码学习，本帖记录了笔者在调试megatron时所用到的命令。


## 安装

笔者用了八张H100的单节点服务器，用vscode远程连接到服务器上。

这里使用的代码库并不是Nvidia官方的代码库，而是阿里打过补丁的，支持deepseek_v2的训练，链接为：https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v2/README.md。


运行下列代码克隆Pai-Megatron-Patch
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# set home dir
echo """
export Megatron_HOME=\"$(pwd)\"
""" >> ~/.bashrc && source ~/.bashrc
```


## 预训练数据集和模型下载


```bash
# ckpt download
cd $Megatron_HOME/examples/deepseek_v2/
mkdir deepseek-ckpts && cd deepseek-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-ckpts/DeepSeek-V2-Lite.tgz
tar -zxf DeepSeek-V2-Lite.tgz


# dataset download
cd $Megatron_HOME/examples/deepseek_v2/
mkdir deepseek-datasets && cd cd deepseek-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid-general.json
```

制作idxmap的脚本如下所示
```bash
cd $Megatron_HOME/toolkits/pretrain_data_preprocessing
sh run_make_pretraining_dataset_megatron.sh \
$Megatron_HOME/examples/deepseek_v2/deepseek-datasets/SlimPajama.json \
DeepSeekV2Tokenizer \
text \
$Megatron_HOME/examples/deepseek_v2/deepseek-datasets/ \
$Megatron_HOME/examples/deepseek_v2/deepseek-ckpts/DeepSeek-V2-Lite
```
为方便期间，我们也提供了已经处理好的idxmap数据集供后续测试使用
```bash
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
```


## Megatron-Core-MoE模型训练流程
### Megatron-Core-MoE模型格式转换
运行`hf2mcore_deepseek_v2_moe_convertor.sh`脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：A2.4B/A21B
SOURCE_CKPT_PATH=$2            # 源路径
TARGET_CKPT_PATH=$3            # 目标路径
TP=$4                          # 模型并行度, 当前只能设置为1
PP=$5                          # 流水并行度
EP=$6                          # 专家并行度
PR=$7                          # 转换精度
mg2hf=$8                       # 是否执行mcore2hf转换
HG_CKPT_PATH=$9                # HF的CKPT的路径
```
例如，使用下述脚本将checkpoint转换到MCore-MoE并检查输出。
注意对于A2.4B模型由于它有27层，所以需要执行非均匀切分策略设置`MP_PP0_LAYERS=6`。

```bash
export MP_PP0_LAYERS=6
cd $Megatron_HOME/toolkits/model_checkpoints_convertor/deepseek
bash hf2mcore_deepseek_v2_moe_convertor.sh \
A2.4B \
$Megatron_HOME/examples/deepseek_v2/deepseek-ckpts/DeepSeek-V2-Lite \
$Megatron_HOME/examples/deepseek_v2/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp4-ep2  \
1  \
4  \
2 \
fp32 \
false 
```


### Megatron-Core预训练

#### 预训练调试命令

将这段代码放到`Pai-Megatron-Patch/.vscode/launch.json`就可以愉快的打断点调试了：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Debug DeepSeek Pretrain (torchrun 8 GPUs)",
            "type": "python",
            "request": "launch",
            "python": "/usr/local/bin/python",  // 设置为你的python环境
            "module": "torch.distributed.run",
            "console": "integratedTerminal", // 在VS Code的集成终端中显示输出
            "subProcess": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/backends/megatron/Megatron-LM-241113",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "true",
                "MP_DATASET_TYPE": "idxmap",
                "MP_AC_LAYERS": "1",
                "MP_PP0_LAYERS": "6",
                "NVTE_FLASH_ATTN": "1",
                "NVTE_FUSED_ATTN": "0",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
            },
            // 设置当前工作目录为 'examples/deepseek_v2'，因为 'pretrain_deepseek.py' 在那里
            "cwd": "${workspaceFolder}/examples/deepseek_v2",
            "args": [
                "--nproc_per_node", "8",
                "--nnodes", "1",
                "--node_rank", "0",
                "--master_addr", "localhost",
                "--master_port", "27047",
                // 接下来是你的脚本 'pretrain_deepseek.py' 及其参数
                "pretrain_deepseek.py",
                "--save", "${workspaceFolder}/examples/deepseek_v2/output_mcore_deepseek_pretrain/checkpoint/pretrain-mcore-deepseek-v2-A2.4B-lr-1e-5-minlr-1e-6-bs-1-gbs-8-seqlen-1024-pr-bf16-tp-1-pp-4-cp-1-ac-sel-do-true-sp-true-ti-122070-wi-1",
                "--lr", "1e-5",
                "--min-lr", "1e-6",
                "--lr-decay-style", "cosine",
                "--weight-decay", "0.1",
                "--adam-beta1", "0.9",
                "--adam-beta2", "0.95",
                "--clip-grad", "1.0",
                "--init-method-std", "0.008",
                "--attention-dropout", "0.0",
                "--hidden-dropout", "0.0",
                "--lr-decay-iters", "122070",
                "--lr-warmup-iters", "1",
                "--train-iters", "122070",
                "--micro-batch-size", "1",
                "--global-batch-size", "8",
                "--num-layers", "27",
                "--hidden-size", "2048",
                "--num-attention-heads", "16",
                "--ffn-hidden-size", "10944",
                "--seq-length", "1024",
                "--max-position-embeddings", "1024",
                "--max-padding-length", "1024",
                "--log-interval", "1",
                "--log-throughput",
                "--eval-interval", "10000",
                "--eval-iters", "10",
                "--save-interval", "100000",
                "--tensorboard-queue-size", "1",
                "--tensorboard-dir", "${workspaceFolder}/examples/deepseek_v2/output_mcore_deepseek_pretrain/tensorboard/pretrain-mcore-deepseek-v2-A2.4B-lr-1e-5-minlr-1e-6-bs-1-gbs-8-seqlen-1024-pr-bf16-tp-1-pp-4-cp-1-ac-sel-do-true-sp-true-ti-122070-wi-1_2025.08.14-15.28.28",
                "--log-timers-to-tensorboard",
                "--log-validation-ppl-to-tensorboard",
                "--tensor-model-parallel-size", "1",
                "--pipeline-model-parallel-size", "4",
                "--context-parallel-size", "1",
                "--no-load-optim",
                "--no-load-rng",
                "--num-workers", "8",
                "--extra-vocab-size", "2400",
                "--patch-tokenizer-type", "DeepSeekV2Tokenizer",
                "--swiglu",
                "--normalization", "RMSNorm",
                "--norm-epsilon", "1e-6",
                "--use-rotary-position-embeddings",
                "--no-bias-swiglu-fusion",
                "--no-rope-fusion",
                "--position-embedding-type", "rope",
                "--untie-embeddings-and-output-weights",
                "--disable-bias-linear",
                "--rotary-base", "10000",
                "--rotary-scaling-factor", "40",
                "--no-save-optim",
                "--kv-channels", "128",
                "--qk-layernorm",
                "--multi-latent-attention",
                "--ckpt-format", "torch",
                "--data-path", "${workspaceFolder}/examples/deepseek_v2/deepseek-datasets/mmap_deepseekv2_datasets_text_document",
                "--split", "99,1,0",
                "--dataset", "MMAP",
                "--bf16",
                "--load", "${workspaceFolder}/examples/deepseek_v2/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp4-ep2",
                "--transformer-impl", "transformer_engine",
                "--recompute-activations",
                "--use-distributed-optimizer",
                "--moe-ffn-hidden-size", "1408",
                "--moe-router-topk", "6",
                "--num-experts", "64",
                "--moe-layer-freq", "1",
                "--moe-aux-loss-coeff", "1e-2",
                "--moe-shared-expert-intermediate-size", "2816",
                "--expert-model-parallel-size", "2",
                "--kv-lora-rank", "512",
                "--qk-head-dim", "128",
                "--qk-pos-emb-head-dim", "64",
                "--v-head-dim", "128",
                "--moe-router-load-balancing-type", "aux_loss",
                "--train-mode", "pretrain",
                "--decoder-first-pipeline-num-layers", "6"
            ],
            "justMyCode": false // 允许调试第三方库代码，重要！
        }
    ]
}
```



#### 预训练原始命令

在DeepSeek-V2中，我们已将预训练和微调整合到`run_mcore_deepseek.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: A2.4B，A21B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度，当前只能设置为1
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
EP=${13}                        # 专家并行度
SP=${14}                        # 是否使用序列并行: true, false
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否优先使用Flash Attention: true, false
SFT=${17}                       # 是否执行微调训练: true, false
AC=${18}                        # 激活检查点模式: sel, full, offload, none
OPTIMIZER_OFFLOAD=${19}         # 是否启用Offload optimizer: false
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
VALID_DATASET_PATH=${22}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${23}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${24}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${25}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${26}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对Deepseek-V2-MoE的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
export MP_PP0_LAYERS=6
cd $Megatron_HOME/examples/deepseek_v2
sh run_mcore_deepseek.sh  \
dsw  \
A2.4B   \
1    \
8 \
1e-5   \
1e-6   \
1024  \
1024  \
bf16  \
1   \
4  \
1 \
2 \
true \
true   \
true \
false \
sel   \
false \
100000  \
$Megatron_HOME/examples/deepseek_v2/deepseek-datasets/mmap_deepseekv2_datasets_text_document   \
$Megatron_HOME/examples/deepseek_v2/deepseek-datasets/mmap_deepseekv2_datasets_text_document   \
$Megatron_HOME/examples/deepseek_v2/deepseek-ckpts/DeepSeek-V2-Lite-ruliu-to-mcore-tp1-pp4-ep2  \
1000000000  \
10000   \
$Megatron_HOME/examples/deepseek_v2/output_mcore_deepseek_pretrain
```