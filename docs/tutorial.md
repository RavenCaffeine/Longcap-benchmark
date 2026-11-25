# 使用教程：在 LongCapBench-TAC-Long 上跑 LongCap baseline

这份教程从零开始，带你走完一整套流程：在 GPU 集群上搭好 LongCap、跑长音频推理、用本仓库的评分器打分，最后把结果写进论文。

- [0. 整体流程](#0-整体流程)
- [1. 背景：任务、benchmark 和指标](#1-背景任务benchmark-和指标)
- [2. 在 GPU 集群上安装 LongCap](#2-在-gpu-集群上安装-longcap)
- [3. 下载模型并启动服务器](#3-下载模型并启动服务器)
- [4. 获取本仓库和 benchmark](#4-获取本仓库和-benchmark)
- [5. 配置 LLM 接口（.env）](#5-配置-llm-接口env)
- [6. 校验评分器：给官方预测打分](#6-校验评分器给官方预测打分)
- [7. 准备音频](#7-准备音频)
- [8. 运行 LongCap 推理](#8-运行-longcap-推理)
- [9. 给自己的推理结果打分](#9-给自己的推理结果打分)
- [10. 读懂 report.json](#10-读懂-reportjson)
- [11. 写进论文时要说明的内容](#11-写进论文时要说明的内容)
- [12. 实现细节：本仓库自行决定的地方](#12-实现细节本仓库自行决定的地方)
- [13. 常见问题](#13-常见问题)
- [附录：命令行参数](#附录命令行参数)

---

## 0. 整体流程

```text
 ┌──────────────┐   ┌─────────────────┐   ┌───────────────────────┐
 │ 安装 LongCap  │──▶│ 下载模型、启动服务器 │──▶│ 准备音频 → longcap caption │──┐
 └──────────────┘   └─────────────────┘   └───────────────────────┘  │ predictions.jsonl
                                                                      ▼
 ┌──────────────────────┐   ┌──────────────────────────────┐   ┌──────────────┐
 │ 下载 benchmark 标注      │──▶│ score_tac.py（LLM 分组 + 打分） │──▶│ report.json   │
 └──────────────────────┘   └──────────────────────────────┘   └──────────────┘
```

第 4 到 6 步（校验评分器）**不需要 GPU，也不需要音频**，可以先做，和第 2、3 步并行推进。

---

## 1. 背景：任务、benchmark 和指标

### 1.1 TAC 任务

**Timestamped Audio Captioning（TAC）**：给定一段音频（可以长达数小时），输出一条事件时间线。每个事件包含：

- `start` 和 `end`：从录音开头算起的秒数，事件之间可以重叠。
- `attributes`：结构化属性。`type` 是 `speech`、`music` 或 `sound_event` 之一，其余是从 62 个专家设计的 key 里选出的适用项，比如转写、说话人性别、乐器、声音事件等。key 名和取值用中文，转写保留原语言。

LongCap 的做法是把音频切成 8/16/24/32 秒的小块，**逐块读入、逐块输出**这一块内结束的事件，之前的音频和文字都留在上下文里（最长 1M token）。

### 1.2 LongCapBench-TAC-Long

| key | 录音 | 语言 | 时长 | 事件数 |
|---|---|---|---:|---:|
| `01_movie_en` | *A Star Is Born*（1937） | 英文 | 110.7 min | 1,892 |
| `01_movie_zh` | 《私人订制》 | 中文 | 117.1 min | 1,956 |
| `08_standup_variety_en` | 第 92 届奥斯卡颁奖典礼 | 英文 | 93.6 min | 1,187 |
| `08_standup_variety_zh` | 《吐槽大会》S3E05 | 中文 | 101.5 min | 1,485 |

合计 7.05 小时、6,520 个人工标注事件。**数据集不包含音频**，需要按 `sources.jsonl` 里的链接自己获取。数据集里还附带了 LongCap-35B-A3B-FP8 的官方预测结果。

### 1.3 三个指标

| 指标 | 含义 | 方向 |
|---|---|---|
| **TER**（Timeline Error Rate） | 把 DER 的算法用在"语义组"上：漏检、虚警、混淆的总时长除以参考时长 | ↓ 越低越好 |
| **GER**（Granularity Error Rate） | 事件被拆分、合并或重复造成的粒度错误，计算公式是 (T_overlap − T_match + T_dup) / 参考事件总时长 | ↓ 越低越好 |
| **tcpCapSim** | 时间约束下的属性语义相似度。属性值由 LLM 按 {0, .25, .5, .75, 1} 打分，最终分数是精确率和召回率的平均 | ↑ 越高越好 |

**语义组**：一般的声音事件不像说话人那样有身份标签，所以要由 LLM 判断参考里的哪些事件和预测里的哪些事件描述的是"同一件事"，把它们分成一组。论文让 LLM 做 3 次分组（3 个 proposal），每个指标各自取最好的那一次。

官方预测的论文分数：**TER 44.84%、GER 11.46%、tcpCapSim 50.85%**。

---

## 2. 在 GPU 集群上安装 LongCap

> 如果你的机器是 Ubuntu 22.04 或更新的系统，直接按[官方 quickstart](https://github.com/jctian98/longcap-code/blob/main/docs/quickstart.md) 的选项 A 安装就行，可以跳过本节的容器部分。

### 2.1 为什么要用容器

LongCap 的依赖（比如 `llguidance==1.7.6`）只提供 `manylinux_2_31` 的 wheel，要求系统的 **glibc ≥ 2.31**。很多集群跑的是 Rocky/RHEL 8，glibc 只有 2.28，直接安装会报这样的错：

```text
ERROR: Could not find a version that satisfies the requirement llguidance==1.7.6 (from versions: ... 1.6.1)
```

先检查一下 glibc 版本：

```bash
ldd --version | head -1
```

如果低于 2.31，就要用 **Apptainer**（以前叫 Singularity）容器。它自带一套较新的系统库（Debian 12，glibc 2.36），不需要 root 权限，HPC 集群上一般都有。

### 2.2 准备容器

下面的 `$WORK` 换成你自己的 scratch 目录。缓存要放在 scratch 里，home 目录的配额通常不够用。

**设置工作目录和缓存位置：**

```bash
export WORK=/srv/scratch/<你的目录>
```

```bash
export APPTAINER_CACHEDIR=$WORK/apptainer_cache PIP_CACHE_DIR=$WORK/pip_cache
```

**确认集群有 Apptainer：**

```bash
which apptainer singularity
```

**拉取 Python 3.12 镜像**（只需做一次）：

```bash
cd $WORK && apptainer pull python312.sif docker://python:3.12-bookworm
```

**进入容器。** `--nv` 会把宿主机的 NVIDIA 驱动挂进容器，进去以后提示符会变成 `Apptainer>`：

```bash
apptainer shell --nv python312.sif
```

> ⚠️ 提示符是 `Apptainer>` 就说明**已经在容器里了**，不要再运行 `apptainer shell`。容器里没有这个命令，会报 `command not found`。

### 2.3 在容器里安装 LongCap

**克隆仓库，建 venv：**

```bash
cd $WORK && git clone https://github.com/jctian98/longcap-code.git && cd longcap-code
```

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

**依次安装依赖：**

```bash
python -m pip install --only-binary=:all: --require-hashes -r requirements/bootstrap.lock
```

```bash
python -m pip install --only-binary=:all: --require-hashes -r requirements/server.lock
```

```bash
python -m pip install --no-build-isolation --no-deps .
```

> ⚠️ `.venv` 一定要**在容器里**创建。在宿主机上建的 venv 链接的是宿主机的库，在容器里不能用。反过来，在容器里装好的 venv 也只能在容器里用。

**检查一下：**

```bash
ldd --version | head -1; nvidia-smi -L; longcap --version
```

应该看到 glibc 2.36 和你的 GPU 型号。

### 2.4 硬件要求

| 检查点 | 是否满足 |
|---|---|
| `nvidia-smi` 里显示的 CUDA 版本 ≥ 13.0 | 驱动满足要求 |
| FP8 模型 | 单卡 H100 80GB 或更大的卡。H200 也可以：同为 Hopper 架构，显存 141 GB |
| BF16 模型 | 官方配置是 2 张 H100。单张 H200 理论上放得下，但官方没有测过 |

---

## 3. 下载模型并启动服务器

以下命令都在**容器里、激活了 venv、位于 `longcap-code` 目录**的情况下运行。

**下载并校验模型**（FP8 版本约 38 GB）：

```bash
export MODEL=LongCap-35B-A3B-FP8
```

```bash
hf download JinchuanTian/$MODEL --local-dir ./models/$MODEL && longcap verify-model --model ./models/$MODEL
```

**启动服务器。只运行一次！**

```bash
nohup longcap serve --model ./models/$MODEL > server.log 2>&1 &
```

**等服务器启动好。** 第一次启动大约要 2 到 5 分钟，因为要读权重、编译 kernel：

```bash
until curl -sf http://127.0.0.1:20000/health > /dev/null; do sleep 10; done; echo ready
```

日志里出现下面这样的输出，就说明启动成功了：

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:20000
```

需要注意的几点：

- **不要重复启动。** 启动前先用 `ps aux | grep "longcap serve" | grep -v grep` 或 `jobs` 查一下。如果启动了多个，它们会抢同一张卡，彼此拖慢，甚至 OOM。全部停掉的方法是 `pkill -f "longcap serve"`。
- **不要退出容器 shell。** 服务器是在容器里放到后台的，退出容器后它可能会被一起结束。要长时间运行，建议在 `tmux` 里进入容器。
- **服务器只监听 `127.0.0.1`。** 客户端要在同一台机器上运行，可以另开一个终端，ssh 到同一个节点再进入容器。
- **TAC 长音频要用默认的 `long` profile。** 不要加 `--profile short`。

---

## 4. 获取本仓库和 benchmark

**在集群上克隆本仓库：**

```bash
cd $WORK && git clone https://github.com/RavenCaffeine/Longcap-benchmark.git && cd Longcap-benchmark
```

**下载 benchmark 的标注**（约 12.6 MB，18 个文件）：

```bash
hf download JinchuanTian/LongCapBench-TAC-Long --repo-type dataset --local-dir data/tac_long
```

**检查下载结果。** 两个 jsonl 都应该是 4 行，一行一条录音：

```bash
wc -l data/tac_long/references.jsonl data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl
```

> 下载时出现 `Still waiting to acquire lock` 或 `unauthenticated requests` 都没关系，前者是正常的锁文件提示，后者只是提醒你没登录 HF 账号。

---

## 5. 配置 LLM 接口（.env）

评分要调用一个 OpenAI 兼容的 chat 接口。论文用的是 **DeepSeek-V4-Flash**，并且开启了 thinking。

**在仓库根目录复制配置模板**，并把权限设成只有自己能读：

```bash
cp .env.example .env && chmod 600 .env
```

然后编辑 `.env`：

```bash
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-...
LLM_MODEL=<服务商给出的模型 id>
# 可选：合并进每个请求的额外 JSON，例如开启 thinking 的参数
# LLM_EXTRA={}
```

读取规则：

- 依次查找**当前目录**和 **`tac_eval/` 目录**下的 `.env`，也可以用 `--env-file 路径` 指定。
- 支持 `#` 注释、`export` 前缀和引号。
- 已经 export 的环境变量优先，不会被 `.env` 覆盖。命令行参数 `--model`、`--base-url` 的优先级最高。
- 运行开头会打印 `[env] .env: loaded ...`，表示读取成功。

> 🔒 `.env` 已经写进 `.gitignore`，**不要提交到 git，也不要发给别人**。还要注意，打分时 benchmark 的标注和模型预测会发给你配置的 API 服务。

---

## 6. 校验评分器：给官方预测打分

这一步不需要 GPU，也不需要音频，在任何能联网的机器上都能跑。

### 6.1 离线冒烟测试（不调用 LLM，几秒钟跑完）

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/smoke --grouper heuristic --judge exact
```

预期输出大致如下：

```text
recording                        TER%     GER%  tcpCapSim%
01_movie_en                     49.21    10.75       20.91
...
MICRO                           47.99    11.39       20.83
```

离线模式**不是论文的评测方法**：分组用的是"同类型、时间重叠最多"的启发式规则，属性打分用的是精确字符串匹配。所以它只能用来检查流程是否跑通。tcpCapSim 偏低是正常的。

### 6.2 LLM 小规模试验（只评第一条录音的前 10 分钟）

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/trial --keys 01_movie_en --max-seconds 600
```

运行时注意观察这几点：

- 是否打印了 `[env] .env: loaded LLM_BASE_URL, LLM_API_KEY, LLM_MODEL`。
- 最后一行的 `diagnostics` 里，`group_fallback_windows` 和 `judge_failed_items` 是否接近 0。这两个数偏高，说明模型回复的 JSON 解析不了，需要换模型或者调整 `LLM_EXTRA`。
- 打开 `runs/trial/groups_01_movie_en_m0.json`，抽查几组分组是否合理。

### 6.3 正式打分

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/score_official
```

- **调用量**：分组约 420 个窗口 × 3 个 proposal ≈ 1,300 次，再加上几百次属性打分。
- **续跑**：所有 LLM 回复都缓存在 `runs/score_official/llm_cache.jsonl`。中断以后重新运行同一条命令，已经完成的请求不会重复调用。
- **并发**：默认 8 个线程，可以用 `--workers` 调整。遇到限流就调小。

把结果和论文对照：**TER 44.84 / GER 11.46 / tcpCapSim 50.85**。两者之间的差距就是"复现误差"，要写进论文。

---

## 7. 准备音频

### 7.1 获取并转换音频

查看每条录音的来源：

```bash
python -c "import json;[print(r['key'], r['source_url']) for r in map(json.loads, open('data/tac_long/sources.jsonl'))]"
```

按链接获取原始媒体（要遵守各来源的使用条款），然后转换成 **16 kHz、单声道、PCM16 的 WAV**，文件名用 `<key>.wav`，放在 `audio/` 目录下：

```bash
mkdir -p audio && ffmpeg -i source.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le -fflags +bitexact -map_metadata -1 audio/01_movie_en.wav
```

### 7.2 校验音频并生成 manifest

```bash
python tac_eval/prepare_audio.py manifest --sources data/tac_long/sources.jsonl --audio-dir audio --out data/manifest_full.jsonl --segment 600 --segment-dir audio_segments --segment-manifest data/manifest_segments.jsonl
```

输出示例：

```text
[verified] 01_movie_en: 16000 Hz, 1 ch, 6640.0 s (header ok, sha256 ok)
wrote 4 recordings to data/manifest_full.jsonl
wrote 45 segments to data/manifest_segments.jsonl
```

- **`[verified]`**：你的音频和官方评测用的音频逐字节一致。
- **`[MISMATCH]`**：来源被重新编码或剪辑过，时间轴可能对不上。默认会跳过这条录音。加上 `--allow-mismatch` 可以强制保留，但**论文里必须写明哪几条对不上**。
- **`--segment 600`**：同时切出 10 分钟的分段，对应论文 Fig. 1 的分段评测设置。只做整条录音评测的话，可以不加这个参数。

---

## 8. 运行 LongCap 推理

在**另一个终端**进入容器、激活 `longcap-code` 的 venv，再 `cd` 到本仓库目录。服务器保持运行。

### 8.1 整条录音单次推理（主要的 baseline）

```bash
longcap caption --manifest data/manifest_full.jsonl --model $WORK/longcap-code/models/LongCap-35B-A3B-FP8 --output runs/full_fp8
```

- 7 小时的音频要跑很久，建议放在 `tmux` 里运行。
- **退出码**：0 表示全部完成；2 表示部分录音不完整，具体看 `runs/full_fp8/predictions/conversion.jsonl`；1 表示请求失败或者连不上服务器。
- 中断后重新运行**同一条命令**会自动续跑。换了输入、模型或解码参数，就要用新的 `--output` 目录。
- 结果在 `runs/full_fp8/predictions/predictions.jsonl`。

### 8.2 10 分钟分段推理（可选，和论文 Fig. 1 的设置一致）

**推理：**

```bash
longcap caption --manifest data/manifest_segments.jsonl --model $WORK/longcap-code/models/LongCap-35B-A3B-FP8 --output runs/seg_fp8
```

**把分段结果拼回整条录音的时间轴**，每个事件的时间会加上所在分段的偏移量：

```bash
python tac_eval/prepare_audio.py stitch --segment-manifest data/manifest_segments.jsonl --predictions runs/seg_fp8/predictions/predictions.jsonl --out runs/seg_fp8/stitched.jsonl
```

---

## 9. 给自己的推理结果打分

**整条录音：**

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp runs/full_fp8/predictions/predictions.jsonl --workdir runs/score_full
```

**10 分钟分段：**

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp runs/seg_fp8/stitched.jsonl --workdir runs/score_seg
```

以后你自己方法的输出，只要整理成同样的格式（每行一个 `{"key", "duration", "events": [{"start", "end", "attributes": {...}}]}`），就可以用同一条命令打分，和 baseline 公平比较。

---

## 10. 读懂 report.json

`<workdir>/report.json` 的主要字段：

| 字段 | 内容 |
|---|---|
| `micro` | **主结果**：在每条录音上分别取最好的 proposal，再把 4 条录音做 micro 汇总后得到的 TER、GER、tcpCapSim，还有 tcpCapSim 的 precision 和 recall |
| `recordings.<key>` | 每条录音的分数，`per_proposal` 里是 3 个 proposal 各自的分数 |
| `ter_components` | `miss`、`fa`、`conf`、`ref`（单位秒），可以看出 TER 主要来自漏检、虚警还是混淆 |
| `ger_components` | `overlap`、`match`、`dup`、`ref_events`（单位秒） |
| `per_proposal_micro` | 按 proposal 整体汇总的分数，用来看 LLM 分组的波动有多大 |
| `diagnostics` | `group_fallback_windows`（分组回复解析失败、退回启发式规则的窗口数）、`judge_failed_items`（打分失败的条目数）、`reversed_hyp`（end < start 的预测事件数） |
| `settings` | 本次运行的参数，不包含 key |

workdir 里还有 `groups_<key>_m<k>.json`，记录每个 proposal 的分组结果，内容是事件下标，可以用来做误差分析。

---

## 11. 写进论文时要说明的内容

- [ ] 模型检查点（FP8 或 BF16）、`longcap --version` 的版本号、服务器 profile（`long`）、贪心解码。
- [ ] 硬件型号。比如用的是 H200，官方只在 H100 上验证过。
- [ ] 音频校验结果：4 条录音是否都是 `verified`。
- [ ] 评分器是**自己复现的**：用了哪个 LLM 和版本、做了几个 proposal、grouping 的 temperature，以及**在官方预测上复现的分数和论文分数的差距**。
- [ ] LongCap 的输出不是逐比特可复现的，batch 大小、并发数和硬件都会影响结果。建议跑 2 次，报告波动范围。
- [ ] 部分录音可能是 `partial`（太多块需要重试，提前停了）。如实报告，不要剔除。

---

## 12. 实现细节：本仓库自行决定的地方

论文附录 A.2 给出了公式和总体流程，但没有交代下面这些细节，由本仓库自行决定：

1. **分组 prompt**：给 LLM 看的是事件 id、完整的绝对时间区间、类型和少量关键属性（转写、说话人、声音事件、乐器、歌词等），见 `score_tac.py` 里的 `GROUPING_KEYS`。prompt 原文见 `GROUP_SYSTEM`。
2. **滑窗**：第 k 个窗口是 `[60k − 15, 60k + 60)` 秒，和窗口有重叠的事件都会给 LLM 看。不同窗口、同一窗口里共享事件的组，通过 union-find 合并成录音级别的分组。没有被分进任何组的事件，各自成为单独的一组。
3. **解析失败**：某个窗口连续 3 次回复都无法解析时，这个窗口退回启发式分组，并记录次数。
4. **3 个 proposal 的多样性**：通过 `--group-temperature 0.7` 实现。如果模型忽略 temperature，比如某些 thinking 模型，3 次 proposal 可能完全一样。
5. **"取最好的 proposal"按录音分别取**，然后再做 micro 汇总。
6. **tcpCapSim**：同一组同一侧的同名 key 会合并成一个值（用 ` | ` 连接），每组每侧只计一次。排除 `caption` 和 `speaker_id`，保留 `type`。归一化后完全相同的值直接记 1，不调用 LLM。LLM 的打分会被吸附到 {0, .25, .5, .75, 1} 中最近的一档。时间约束是组内至少有一对参考和预测区间重叠，或者相距不超过 5 秒。
7. **区间**：采用 `[start, end)`。`end < start` 的预测事件按长度 0 处理并计数，不修改输入文件。TER 和 GER 不使用 collar。
8. **匹配**：GER 的 T_match 用最大权一对一匹配（匈牙利算法）计算，权重是两个事件的时间重叠长度。

这些选择都会影响分数。改动其中任何一项，都要重新给官方预测打分，确认和论文的差距。

---

## 13. 常见问题

**Q：`pip install` 报 `No matching distribution found for llguidance==1.7.6`。**
A：系统 glibc 低于 2.31，见 [2.1 节](#21-为什么要用容器)，需要在 Apptainer 容器里安装。可以用 `ldd --version | head -1` 确认版本。如果在容器里还是报这个错，多半是你其实不在容器里：提示符不是 `Apptainer>`，或者 `echo $APPTAINER_CONTAINER` 输出为空。

**Q：`bash: apptainer: command not found`。**
A：你已经在容器里了，不需要再进一次。

**Q：`setlocale: LC_ALL: cannot change locale`。**
A：没有影响，容器里没有宿主机的 locale。想去掉这个警告，可以运行 `export LC_ALL=C.UTF-8`。

**Q：`ps` 里有好几个 `longcap serve` 进程。**
A：vLLM 会派生子进程，所以有几个进程是正常的。但如果 `jobs` 显示有多个后台任务，就说明重复启动了，用 `pkill -f "longcap serve"` 全部停掉，再只启动一次。

**Q：服务器启动后显存一直是 0。**
A：还在从磁盘读权重。在网络文件系统上，读 35 GB 可能要几分钟，进程状态是 `D` 的话就是在等磁盘。

**Q：评分时报 `LLM mode needs LLM_API_KEY and --model/LLM_MODEL`。**
A：没有读到 `.env`。确认你是在仓库根目录运行的，或者用 `--env-file` 指定路径，并检查开头有没有打印 `[env] ... loaded ...`。

**Q：`group_fallback_windows` 很高。**
A：LLM 的回复里没有合法的 JSON。可以换一个对 JSON 输出更稳定的模型，检查 `LLM_EXTRA`，或者打开 `llm_cache.jsonl` 看看原始回复。

**Q：想重新打分，但不想重复调用 LLM。**
A：用同一个 `--workdir` 再跑一次就行，会全部命中缓存。如果改了 prompt 或模型，缓存的 key 会变，相应的请求会重新调用。

---

## 附录：命令行参数

### `tac_eval/score_tac.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ref` | 必填 | 参考标注 `references.jsonl` |
| `--hyp` | 必填 | 预测结果，LongCap 输出格式 |
| `--workdir` | 必填 | 缓存、分组结果和报告的输出目录 |
| `--env-file` | `./.env` → `tac_eval/.env` | LLM 配置文件 |
| `--keys` | 全部 | 只评这几条录音 |
| `--max-seconds` | 不限制 | 只评开始时间早于这个秒数的事件，用于小规模试验 |
| `--grouper` | `llm` | 设成 `heuristic` 是离线启发式分组 |
| `--judge` | `llm` | 设成 `exact` 是离线精确字符串匹配 |
| `--proposals` | 3 | 分组 proposal 的数量 |
| `--group-temperature` | 0.7 | 分组请求的 temperature |
| `--judge-batch` | 40 | 每次打分请求包含的条目数 |
| `--max-value-chars` | 2000 | 每个属性值发给 LLM 时最多保留的字符数 |
| `--workers` | 8 | 并发请求数 |
| `--base-url` / `--model` / `--llm-extra` | 来自环境变量 | 分别覆盖 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_EXTRA` |

### `tac_eval/prepare_audio.py`

| 子命令 | 参数 | 说明 |
|---|---|---|
| `manifest` | `--sources`、`--audio-dir`、`--out` | 校验 `<key>.wav` 并生成整条录音的 manifest |
| | `--segment 600`、`--segment-dir`、`--segment-manifest` | 同时切出 10 分钟分段，生成对应的 manifest |
| | `--skip-sha` | 只检查 WAV 头，不算 sha256，速度更快 |
| | `--allow-mismatch` | 保留和官方音频不一致的录音 |
| `stitch` | `--segment-manifest`、`--predictions`、`--out` | 把分段预测结果拼回整条录音的时间轴 |
