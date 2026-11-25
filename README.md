# Longcap-benchmark

把 [LongCap](https://github.com/jctian98/longcap-code)（arXiv:2611.11125）当作**长音频带时间戳音频描述（TAC）**的 baseline，在 [LongCapBench-TAC-Long](https://huggingface.co/datasets/JinchuanTian/LongCapBench-TAC-Long) 上做推理和评测。

官方仓库只提供推理代码，**没有提供 TER / GER / tcpCapSim 的评分代码**。本仓库按论文 §2.2 和附录 A.2 重新实现了评分，并提供了准备音频、切分段、拼接结果用的工具。

📖 **完整教程：[docs/tutorial.md](docs/tutorial.md)**，从集群环境搭建一直讲到打分和结果解读。

## 目录结构

```text
Longcap-benchmark/
├── README.md
├── .env.example          # LLM 接口配置模板，复制为 .env 后填写
├── docs/
│   └── tutorial.md       # 使用教程
└── tac_eval/
    ├── score_tac.py      # 评分入口：LLM 分组 + LLM 打分 → report.json
    ├── tac_metrics.py    # TER / GER / tcpCapSim 的纯计算部分（确定性）
    ├── prepare_audio.py  # 校验音频、生成 manifest、10 分钟分段、拼接分段结果
    └── test_metrics.py   # 单元测试（包含论文附录 A.2 的例子）
```

## 快速开始

只依赖 Python 3.9+ 的标准库。如果装了 `scipy`，会自动用它加速匹配。评分不需要 GPU。

**1. 下载 benchmark**，只有标注和官方预测，不含音频：

```bash
hf download JinchuanTian/LongCapBench-TAC-Long --repo-type dataset --local-dir data/tac_long
```

**2. 配置 LLM 接口**，填入 `LLM_API_KEY` 和 `LLM_MODEL`：

```bash
cp .env.example .env && chmod 600 .env
```

**3. 检查评分器是否正常。** 先跑离线冒烟测试，不调用 LLM：

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/smoke --grouper heuristic --judge exact
```

再用 LLM 跑一个小规模试验，只评第一条录音的前 10 分钟：

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/trial --keys 01_movie_en --max-seconds 600
```

**4. 正式给官方预测打分**，然后和论文数字对照：

```bash
python tac_eval/score_tac.py --ref data/tac_long/references.jsonl --hyp data/tac_long/predictions/longcap-35b-a3b-fp8.jsonl --workdir runs/score_official
```

官方预测的论文分数：**TER 44.84%、GER 11.46%、tcpCapSim 50.85%**。

自己跑推理和打分的完整流程见[教程](docs/tutorial.md)。

## 单元测试

```bash
cd tac_eval && python -m unittest test_metrics -v
```

## 引用

如果使用了 LongCap 或 LongCapBench，请引用原论文：

```bibtex
@misc{tian2026longcap,
  title = {LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing},
  author = {Tian, Jinchuan and Chen, Sen and Huang, Wenbin and Weng, Chao and Huang, Zhengwei and Shi, Jiatong and Lu, Yichen and Wang, Haoran and Watanabe, Shinji and Zhou, Cong},
  year = {2026},
  eprint = {2611.11125},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

本仓库的评分代码是独立复现的，不是官方实现。分数可能和论文有出入，使用时请说明这一点。
