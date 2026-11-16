# 回声（Echo）

中文长期陪伴对话的**分层记忆系统**。从 NovaEcho 的 `novaecho/memory` 抽离而来，保留了完整的检索链与改过底层的三个组件（beam / hierarchical / mnemosyne）。

> 不是聊天机器人框架，只是记忆层：给它 `remember_turn()`，它给你 `agent_search()` / `get_roleplay_context()`。

## 架构

```
Tier 0  短期      最近若干轮原文
Tier 1  层级      tier1_hierarchical：会话摘要 + 逐层下钻（L1/L2）
Tier 2  拾忆      tier2_beam：向量检索，后端 = vendored Mnemosyne（FTS5 兜底）
Tier 3  核心      tier3_core：稳定偏好 / 边界
Facts   事实      tier_facts：时间片摘要 + 原文回取 + 事件卡 + 原文轮 BM25 + 到期/实体通道
```

检索入口：

- `MemoryService.agent_search(query, max_results)` —— 面向 Agent 的深度检索（Facts + 事件 + 实体/到期开关）；
- `MemoryService.search(query, ...)` / `get_roleplay_context()` —— 面向角色扮演的注入；
- `FactsMemory.search_segments()` / `search_events()` / `search_sources()` —— 可单独调用。

## 目录

```
echo/
  memory/
    service.py           MemoryService 门面
    settings.py          全部开关（env 前缀 MEMORY_）
    tier0_short_term.py
    tier1_hierarchical.py   改过底层：摘要分层与下钻规则
    tier2_beam.py            改过底层：Mnemosyne 适配、中文长问句门控补偿
    tier3_core.py
    tier_facts.py            时间片 / 事件卡 / 到期 / 实体 / 原文轮 BM25
    zh_segment.py            中文分词与 BM25 词项
    zh_synonyms.py
    consolidator.py / evolution.py / idle_scheduler.py / token_tracker.py
  models.py / llm_usage.py
vendor/
  mnemosyne/              改过底层的 Mnemosyne 3.10（向量、召回、facts 抽取）
models/
  *.lock                  模型占位清单（权重不入库）
scripts/
  fetch_models.py         按 lock 下载（HF / 魔搭）
  smoke_memory.py         冒烟：写入两轮 + 检索一次
```

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_models.py                 # 默认 HuggingFace
# 国内网络：
#   python scripts/fetch_models.py --source modelscope
#   或 HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_models.py

python scripts/smoke_memory.py
```

冒烟输出应类似：

```
Model loaded succeed
检索到 4 条：
 - 我前天把田恬的软木瑜伽砖放猫爬架旁边，被汤团抓了好几道印子，还没敢跟她说。
 ...
stats: {'vector_search': 'on'}
```

## 模型与精度

| 模型 | 用途 | 运行时 | 精度 |
|---|---|---|---|
| `BAAI/bge-small-zh-v1.5` | 向量索引 + 时间片向量 | fastembed（ONNX） | ONNX 默认（无需 fp16） |
| `BAAI/bge-reranker-v2-m3` | 候选重排（可选） | transformers | **CUDA 可用时 float16**，CPU 回退 float32 |

`models/*.lock` 只是清单，权重不进仓库。`fetch_models.py` 读取清单下载；重排模型落在 `models/<name>/`，向量模型落在 fastembed 缓存目录。

## 关键开关（环境变量，前缀 `MEMORY_`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_BEAM_BACKEND` | `auto` | `mnemosyne` / `fallback` / `auto` |
| `MEMORY_VECTOR_ENABLED` | `1` | 向量检索；关闭则只用 FTS5 |
| `MEMORY_FACTS_ENABLED` | `1` | 时间片 / 事件 / 原文证据 |
| `MEMORY_FACTS_EVENTS` | `1` | 事件卡通道（动作词 → 时间片原文） |
| `MEMORY_FACTS_DUE` | `0` | 到期日历（默认关：评测里净收益为负） |
| `MEMORY_FACTS_SEGMENT_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 时间片向量模型 |
| `MNEMOSYNE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Tier2 向量模型，中文建议设成 `BAAI/bge-small-zh-v1.5` |
| `MNEMOSYNE_EMBEDDING_DIM` | 模型默认 | 与所选模型一致（bge-small-zh 为 512） |

完整列表见 `echo/memory/settings.py`。

## 关于 vendored 组件

- `echo/memory/tier1_hierarchical.py`、`tier2_beam.py` 是改过底层的实现：摘要分层、下钻预算、Mnemosyne 适配与中文长问句的词法门控补偿都在这里；
- `vendor/mnemosyne` 是 Mnemosyne 3.10.0 的本地修改版（含 beam/recall/facts 抽取的补丁）。`echo/__init__.py` 会把 `vendor/` 插到 `sys.path` 最前面，所以**不要**再 `pip install mnemosyne-memory`，以免装回原版覆盖修改。

## 评测

开发集评测（Bench 2.2，短对话 809 题 + 自然用记忆榜）在本仓库发布时跑过一轮，结果与成本记录在 `RESULTS.md`。
