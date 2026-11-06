# 在另一台机器运行实验

当前实验使用 `SCALE_relaxed/`，模型为 **DeepSeek-V3.2**，默认 S8/M64。
数据准备可在普通 CPU 电脑运行；模型推理需要 Linux、CUDA 和足够的 GPU 显存。
当前服务器使用 TP=8；仅下载 GitHub 代码并不会下载模型权重。

## 1. 下载代码和准备数据环境

```bash
git clone https://github.com/amy-77/Scale.git
cd Scale
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r scripts/data/requirements.txt
```

数据环境与 GPU serving 环境分开：前者按 RULER 生成器使用 Transformers 4.x，
当前 serving 环境使用 Transformers 5.x。不要把数据依赖覆盖安装到正在运行的服务环境。
下载使用标准 `HTTPS_PROXY` / `HTTP_PROXY`；Hugging Face 下载支持 `HF_ENDPOINT`。
普通网络可直连时无需设置代理。公司网络的证书配置需保持正常的 TLS 校验。

## 2. 下载 LongBench v2

```bash
# 官方全集：503 题
python scripts/data/download_longbench_v2.py \
  --subset full --output data/longbench_v2_full.json

# 与当前论文实验一致的 401 题；逐题校验内容哈希
python scripts/data/download_longbench_v2.py \
  --subset paper401 --output data/longbench_v2_paper401.json
```

使用 [官方 LongBench v2 数据](https://huggingface.co/datasets/zai-org/LongBench-v2)
的固定 revision；原 `THUDM/LongBench-v2` 地址目前重定向到该地址。
下载缓存自动复用。已有官方 `data.json` 可加 `--input-file /path/to/data.json`。
`paper401` 按已发布的 ID 清单筛选，不重新随机抽题。

### 401 题具体如何选择

这套数据来自官方 503 题，去掉原数据收集脚本的 `train` 组 96 题和
`validation` 组 6 题，剩下 **503 − 96 − 6 = 401 题**。
这里的 `train` / `validation` 是项目内部划分；Hugging Face 上完整的 503 题
都放在名为 `train` 的官方数据 split 中，两者不是同一个概念。

具体规则沿用 [`select_stratified`](../SCALE_relaxed/runner/evaluators/collect_longbench_v2.py)：

1. 按难度 `easy/hard` × 长度 `short/medium/long` 分成 6 层。
2. 每层再按 `domain` 分组；每个领域内部按题目 `_id` 的 SHA256 升序排列。
3. 将领域名排序，按领域轮转取题，得到该层的确定性顺序。
4. 每层前 16 题划入 `train`，第 17 题划入 `validation`，共排除 17 题。
5. 从官方全集中去掉这 102 个 ID，保留其余题目的原始顺序。

该规则不读取模型预测、答案正确与否或得分；它也不按实际 token 数是否超过
128K 来筛题。输入截断属于后续评测配置，与这一步选题无关。

| 难度 | 长度 | 官方全集 | 排除 | 保留 |
| --- | --- | ---: | ---: | ---: |
| easy | short | 59 | 17 | 42 |
| easy | medium | 88 | 17 | 71 |
| easy | long | 45 | 17 | 28 |
| hard | short | 121 | 17 | 104 |
| hard | medium | 127 | 17 | 110 |
| hard | long | 63 | 17 | 46 |
| 合计 | | 503 | 102 | 401 |

固定清单及完整排除 ID 见
[`longbench_heldout_manifest.json`](../experiments/20261107/longbench_heldout_manifest.json)。
下载脚本会重新计算上述规则，校验 401 个 ID 和顺序，再逐题校验内容 SHA256。
本次核对得到的集合、顺序和全部 401 条内容均与服务器当前数据一致。
因此公司电脑执行上面的 `--subset paper401` 命令即可得到相同的评测输入。

历史口径中，`held-out` 特指排除了 `train` 和 `validation`：旧收集脚本另标记的
6 条 `calibration` 和 6 条 `test` 仍包含在这 401 条中。本脚本保留这一既有口径。

## 3. 生成 RULER

脚本固定 [NVIDIA/RULER](https://github.com/NVIDIA/RULER) 到原始生成版本
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`，自动准备 NLTK、Paul Graham、
SQuAD 和 HotpotQA 输入。默认 seed 为原实验的 `20260823`。
需要对应模型的 tokenizer 文件，无需加载模型权重。

```bash
# 先用 2 条 2K 样本检查生成环境；这不是正式评测集
python scripts/data/generate_ruler.py \
  --tokenizer /models/DeepSeek-V3.2 --output data/ruler_smoke \
  --tasks niah_single_1 --lengths 2048 --samples 2

# 原实验的选样规则：每个任务生成 20 条，保留第 7–20 条
# 13 任务 × 2 长度 × 14 条 = 364 条
python scripts/data/generate_ruler.py \
  --tokenizer /models/DeepSeek-V3.2 --output data/ruler_paper364 \
  --lengths 32768 131072 --samples 20 --paper-subset --verify-paper

# 新实验可扩大样本量；与旧的 364 条结果分开记录
python scripts/data/generate_ruler.py \
  --tokenizer /models/DeepSeek-V3.2 --output data/ruler_100 \
  --lengths 32768 131072 --samples 100
```

原始完整 20 条位于输出目录旁的 `*_raw20/`，不会被评测器重复读入。
同参数重跑可跳过已有的完整任务；若参数/依赖/语料发生变化会要求使用新输出目录。
`--verify-paper` 对照 [固定数据哈希](../experiments/20261107/ruler_manifest.json)。
相同 seed 不足以保证跨依赖版本/语料更新的字节一致；校验失败会明确报错，不能把
未匹配的数据标成原论文子集。当前服务器上的 26 个文件已核实等于原始数据 `[6:20]`。
`wonderwords==3.0.1` 的词表版本已固定；使用 2.2.0 会改变针检索样本。
本次实际重生成了 `niah_single_1` 的 32K/14 条子集，通过原数据字节哈希校验。

## 4. GPU 环境和 HISA 基线

若已有兼容的 serving 环境，可直接使用。仓库提供当前 H20 环境的 Docker 构建配方：

```bash
docker build --network host -f scripts/env/Dockerfile -t scale-experiments:20261107 .
docker run --rm -it --gpus all --network host --shm-size 64g \
  -v "$PWD:/workspace/Scale" \
  -v /path/to/models:/models:ro \
  scale-experiments:20261107
```

基础镜像固定 digest；Torch 2.11.0+cu130、CUDA toolkit 13.0.2、FlashInfer 0.6.13
等版本与正在运行的 H20 质量实验一致。需兼容 CUDA 13 的 NVIDIA 驱动。
配方根据已运行环境整理，尚未完整重建该镜像；不同 GPU/驱动仍需先跑小样本验证。
依赖及 CUDA/FlashInfer 修正记录见 [provenance.json](../experiments/20261107/provenance.json)。

准备固定块基线（只运行 SCALE 时可省略）：

```bash
cp -a sglang HISA_current
git -C HISA_current init -q
git -C HISA_current apply --check ../experiments/20261107/hisa_current.patch
git -C HISA_current apply ../experiments/20261107/hisa_current.patch
```

如需 4-token 固定块，继续应用小粒度扩展：

```bash
git -C HISA_current apply --check ../experiments/20261107/hisa_block4.patch
git -C HISA_current apply ../experiments/20261107/hisa_block4.patch
python scripts/experiments/validate_hisa_fine_blocks.py --hisa-source HISA_current
```

该补丁扩展现有 grouped Triton 路径到 K=4，并增加对应的调度配置；K=8/16 的
计算路径保持不变。H20 上已通过 36 项 GPU 参考检查（池化、非连续物理页访问、
尾块、Top-k）；完整 4-token 端到端精度结果仍须等待评测完成。
SCALE 需要 `tilelang==0.1.11` 和 `apache-tvm-ffi==0.1.11`：0.1.9 在当前代码的
增量摘要内核编译时会失败。Docker 配方已固定这两个版本。

`git init` 将复制目录设为独立 Git 根目录，确保补丁路径相对于该目录应用。
基线补丁已校验：还原后的 Python 源码 2,635 个文件与当前实验快照一致。
历史仓库含 `README.md` / `readme.md` 同名大小写文件，完整推理环境请使用 Linux
区分大小写的文件系统。

## 5. 运行一个配置

先使用 `--max-records 2` 做小样本检查，正式运行时去掉它并选择新的输出目录。
以下命令在 GPU serving 环境中运行；脚本会启动模型、等待就绪、评测并关闭自身启动的服务。

```bash
# SCALE S8/M64，关闭 dense-final，RULER
python scripts/experiments/run_experiment.py \
  --method scale --dense-final 0 --dataset ruler \
  --data data/ruler_paper364 --model /models/DeepSeek-V3.2 \
  --output results/scale64_no_dense_ruler --resume

# SCALE S8/M64，开启 dense-final，LongBench v2 的 401 条固定子集
python scripts/experiments/run_experiment.py \
  --method scale --dense-final 1 --dataset longbench-v2 \
  --data data/longbench_v2_paper401.json --model /models/DeepSeek-V3.2 \
  --output results/scale64_dense_longbench401 --resume

# HISA 16-token 固定块，关闭 dense-final，候选预算 8192
python scripts/experiments/run_experiment.py \
  --method hisa --hisa-block 16 --dense-final 0 --dataset ruler \
  --data data/ruler_paper364 --model /models/DeepSeek-V3.2 \
  --output results/hisa16_no_dense_ruler --resume
```

细粒度对比使用 `--hisa-block 4/8/16`（K=4 先应用上面的补丁）。
SCALE 对应 `--scale-average 4/8/16`，默认中间 split divisor 分别为 1/1/2；
64/128 的默认 split divisor 都为 8。可用 `--scale-split` 明确指定并记入配置。
SCALE 的 average 是目标平均区间长度，并非固定块长度。

两种方法必须统一 `--dense-final`、候选预算、数据及生成参数。当前新增成对队列
统一 `--dense-final 1`；此前的 `dense-final 0` 结果单列，不能混成同协议比较。
先完成同候选预算的质量–速度点，再根据实测延迟补预算扫描；相同候选预算不等于
相同延迟。`--method dsa` 使用原始 DSA 路径。
`--tp`、`--port`、`--mem-fraction` 可调整；默认分别是 8、31946、0.82。
`--dry-run` 只打印命令。已有服务可加 `--server http://127.0.0.1:31946`；此时
脚本不修改已有服务配置，调用者应核对其算法/模型，实际 server info 会保存到结果目录。
运行前自行分配空闲 GPU，可通过 `CUDA_VISIBLE_DEVICES` 指定。

每个配置使用独立输出目录，包含逐题 `predictions.jsonl`、汇总 `summary.json`、
`run_config.json` 与服务日志。`--resume` 跳过已成功完成的题；配置或数据发生变化时拒绝混跑。
LongBench 默认将输入按头尾截断到 131072 tokens，生成上限 128 tokens；这与无限上下文
评测不同。脚本使用当前 DeepSeek 提示模板，不能直接把模型路径替换成 GLM 并声称协议相同。

完整队列及性能测试入口见 [实验快照说明](../experiments/20261107/README.md)。
`--dense-final 0/1` 必须在比较表中单独标明；共享 GPU 时的请求耗时仅供诊断。
