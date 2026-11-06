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
git -C HISA_current apply --check ../experiments/20261107/hisa_current.patch
git -C HISA_current apply ../experiments/20261107/hisa_current.patch
```

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

`--hisa-block 8` 对应更细固定块，`--method dsa` 使用原始 DSA 路径。
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
