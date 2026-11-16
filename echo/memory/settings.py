from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import LLMMessage

if TYPE_CHECKING:
    from ..providers.openai_compatible import OpenAICompatibleSettings


# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------

DEFAULT_WORKING_DIR = Path(".novaecho") / "memory"
DEFAULT_MAX_INPUT_LENGTH = 128 * 1024

# Tier 0 — 短期
DEFAULT_SHORT_TERM_SIZE = 50

# Tier 1 — hmem 分层
DEFAULT_HIERARCHICAL_L1_MAX = 80       # 系统提示注入的最大 L1 摘要数
DEFAULT_HIERARCHICAL_DB = "hierarchical.db"

# Tier 1 检索打分
DEFAULT_TIER1_WEIGHT_RELEVANCE = 0.60
DEFAULT_TIER1_WEIGHT_IMPORTANCE = 0.25
DEFAULT_TIER1_WEIGHT_RECENCY = 0.15
DEFAULT_TIER1_BM25_K = 1.0
DEFAULT_TIER1_RECENCY_HALFLIFE_HOURS = 720.0   # 30 天
DEFAULT_TIER1_MIN_SCORE = 0.10

# Tier 3 检索打分
DEFAULT_TIER3_WEIGHT_COVERAGE = 0.70
DEFAULT_TIER3_WEIGHT_EXACT = 0.30

# Agent 检索各层融合权重与标定参考分
DEFAULT_FUSION_TIER0_WEIGHT = 0.8
DEFAULT_FUSION_TIER1_WEIGHT = 0.8
DEFAULT_FUSION_TIER2_WEIGHT = 1.0
DEFAULT_FUSION_TIER3_WEIGHT = 0.9
DEFAULT_FUSION_TIER0_REFERENCE = 1.5
DEFAULT_FUSION_TIER1_REFERENCE = 0.8
DEFAULT_FUSION_TIER2_REFERENCE = 0.6
DEFAULT_FUSION_TIER3_REFERENCE = 0.9
DEFAULT_FUSION_TIER1_MAX_ITEMS = 5

# Tier 2 — 拾忆（底层兼容 BEAM/Mnemosyne）
DEFAULT_BEAM_DIR = "beam"
DEFAULT_BEAM_TOP_K = 5
DEFAULT_BEAM_BACKEND = "auto"

# Tier 3 — 核心
DEFAULT_CORE_DB = "core_facts.db"
DEFAULT_CORE_FACTS_MAX = 30            # 系统提示注入的最大核心事实数

# 记忆整合 (Consolidation)
DEFAULT_TOKEN_DB = "token_usage.db"
DEFAULT_EVOLUTION_DB = "evolution.db"
DEFAULT_CONSOLIDATION_SIMILARITY = 0.55   # TF-IDF + Jaccard 聚类阈值
DEFAULT_CONSOLIDATION_MIN_CLUSTER = 2     # 最小聚类大小
DEFAULT_CONSOLIDATION_MIN_CHUNKS = 5      # 触发整合的最小记忆片段数

# 空闲调度器
DEFAULT_IDLE_THRESHOLD = 300              # 空闲 5 分钟后触发
DEFAULT_CONSOLIDATION_INTERVAL = 1800     # 整合最小间隔 30 分钟
DEFAULT_EVOLUTION_INTERVAL = 3600         # 进化分析最小间隔 1 小时
DEFAULT_TOKEN_REPORT_INTERVAL = 7200      # Token 报告间隔 2 小时

# 隔离作用域记忆（渠道人格会话：按对话对象/群聊分目录，只启用 Tier0+Tier1）
DEFAULT_ISOLATED_DIR = "isolated"
DEFAULT_ISOLATED_MAX_SCOPES = 64


# ---------------------------------------------------------------------------
# MemoryPreparationResult (保持兼容)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MemoryPreparationResult:
    """记忆压缩结果（保持与旧接口兼容）。"""
    messages: list[LLMMessage]
    compressed_summary: str


# ---------------------------------------------------------------------------
# MemorySettings
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MemorySettings:
    """记忆系统配置 — 4 层架构。

    层级结构:
      Tier 0: ShortTermMemory           (纯内存, Roleplay 即时上下文)
      Tier 1: HierarchicalMemory        (hmem 5级懒加载, 跨会话偏好)
      Tier 2: 拾忆（BeamMemory）         (Mnemosyne 后端，Agent 深度检索)
      Tier 3: CoreMemory                (固化事实, 全局注入)
      tier_facts: FactsMemory            (结构化事件事实与原文证据, 按需检索)
    """

    working_dir: Path

    # Tier 0 — 短期记忆
    short_term_max_size: int = DEFAULT_SHORT_TERM_SIZE

    # Tier 1 — hmem 风格分层
    hierarchical_l1_max: int = DEFAULT_HIERARCHICAL_L1_MAX
    hierarchical_db_name: str = DEFAULT_HIERARCHICAL_DB
    tier1_weight_relevance: float = DEFAULT_TIER1_WEIGHT_RELEVANCE
    tier1_weight_importance: float = DEFAULT_TIER1_WEIGHT_IMPORTANCE
    tier1_weight_recency: float = DEFAULT_TIER1_WEIGHT_RECENCY
    tier1_bm25_k: float = DEFAULT_TIER1_BM25_K
    tier1_recency_halflife_hours: float = DEFAULT_TIER1_RECENCY_HALFLIFE_HOURS
    tier1_min_score: float = DEFAULT_TIER1_MIN_SCORE

    # Tier 2 — Mnemosyne BEAM
    beam_dir_name: str = DEFAULT_BEAM_DIR
    beam_top_k: int = DEFAULT_BEAM_TOP_K
    beam_backend: str = DEFAULT_BEAM_BACKEND
    vector_enabled: bool = True
    gpu_enabled: bool = False

    # 独立事实证据索引
    facts_enabled: bool = False
    facts_db_name: str = "facts.db"
    facts_batch_messages: int = 2
    facts_defer_extraction: bool = False
    # 按时间分段归档（一次调用整理一段），替代逐消息抽取原子事实
    facts_segment_mode: bool = True
    facts_segment_gap_seconds: int = 900
    facts_segment_max_messages: int = 40
    facts_segment_max_lines: int = 10
    # 时间片召回的词法增强（BM25 / 同义词 / 伪相关反馈），不增加 LLM 调用
    facts_segment_bm25: bool = True
    facts_segment_synonyms: bool = True
    facts_segment_rm3: bool = False
    # 归档段复用 Tier2 的中文向量模型，与词法做 RRF 融合
    facts_segment_vectors: bool = True
    facts_segment_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 事件卡通道：规则从用户原话抽动作事件，命中后注入所在时间片原文
    facts_events: bool = True
    facts_event_rows: int = 1
    # 到期日历：今天该主动提起的约定（还钱/考试/取件/生日等）
    # 默认关闭：809 全量上开着反而少 2 题（同一天有对照题被判“编造今天有约定”）
    facts_due: bool = False
    facts_due_rows: int = 2

    # Tier 3 — 核心固化
    core_db_name: str = DEFAULT_CORE_DB
    core_facts_max: int = DEFAULT_CORE_FACTS_MAX

    # Agent 检索融合
    fusion_tier0_weight: float = DEFAULT_FUSION_TIER0_WEIGHT
    fusion_tier1_weight: float = DEFAULT_FUSION_TIER1_WEIGHT
    fusion_tier2_weight: float = DEFAULT_FUSION_TIER2_WEIGHT
    fusion_tier3_weight: float = DEFAULT_FUSION_TIER3_WEIGHT
    fusion_tier0_reference: float = DEFAULT_FUSION_TIER0_REFERENCE
    fusion_tier1_reference: float = DEFAULT_FUSION_TIER1_REFERENCE
    fusion_tier2_reference: float = DEFAULT_FUSION_TIER2_REFERENCE
    fusion_tier3_reference: float = DEFAULT_FUSION_TIER3_REFERENCE
    fusion_tier1_max_items: int = DEFAULT_FUSION_TIER1_MAX_ITEMS

    # 压缩策略
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH
    language: str = "zh"

    # LLM (仅用于摘要压缩, 非必须)
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # 记忆整合 (Consolidation)
    token_db_name: str = DEFAULT_TOKEN_DB
    evolution_db_name: str = DEFAULT_EVOLUTION_DB
    consolidation_similarity: float = DEFAULT_CONSOLIDATION_SIMILARITY
    consolidation_min_cluster: int = DEFAULT_CONSOLIDATION_MIN_CLUSTER
    consolidation_min_chunks: int = DEFAULT_CONSOLIDATION_MIN_CHUNKS
    consolidation_use_llm: bool = True

    # 空闲调度器
    idle_threshold_seconds: int = DEFAULT_IDLE_THRESHOLD
    consolidation_interval_seconds: int = DEFAULT_CONSOLIDATION_INTERVAL
    evolution_interval_seconds: int = DEFAULT_EVOLUTION_INTERVAL
    token_report_interval_seconds: int = DEFAULT_TOKEN_REPORT_INTERVAL
    idle_consolidation_enabled: bool = True

    # 渠道人格会话的隔离记忆（每个对话对象/群聊一个子目录，只启用 Tier0+Tier1）
    isolated_working_dir_name: str = DEFAULT_ISOLATED_DIR
    isolated_max_scopes: int = DEFAULT_ISOLATED_MAX_SCOPES

    # -- 工厂方法 ------------------------------------------------------------

    def for_isolated_scope(self, directory_name: str) -> "MemorySettings":
        """派生某个隔离作用域专用的配置（独立目录，只启用 Tier0+Tier1）。"""
        return replace(
            self,
            working_dir=self.working_dir / self.isolated_working_dir_name / directory_name,
        )

    @classmethod
    def from_provider_settings(
        cls,
        workspace: str | Path,
        provider_settings: "OpenAICompatibleSettings",
    ) -> "MemorySettings":
        workspace_path = Path(workspace).resolve()
        env = os.environ
        working_dir = _resolve_path(
            env.get("MEMORY_WORKING_DIR"),
            base=workspace_path,
            default=workspace_path / DEFAULT_WORKING_DIR,
        )

        return cls(
            working_dir=working_dir,
            short_term_max_size=_env_int(env, "MEMORY_SHORT_TERM_SIZE", 50, 10),
            hierarchical_l1_max=_env_int(env, "MEMORY_HIERARCHICAL_L1_MAX", 80, 10),
            beam_top_k=_env_int(env, "MEMORY_BEAM_TOP_K", 5, 1),
            beam_backend=_env_choice(env, "MEMORY_BEAM_BACKEND", DEFAULT_BEAM_BACKEND, {"auto", "mnemosyne", "fallback"}),
            vector_enabled=_env_bool(env, "MEMORY_VECTOR_ENABLED", True),
            gpu_enabled=cls._detect_gpu(env),
            facts_enabled=_env_bool(env, "MEMORY_FACTS_ENABLED", False),
            facts_segment_mode=_env_bool(env, "MEMORY_FACTS_SEGMENT_MODE", True),
            facts_segment_gap_seconds=_env_int(env, "MEMORY_FACTS_SEGMENT_GAP", 900, 60),
            facts_segment_max_messages=_env_int(env, "MEMORY_FACTS_SEGMENT_MAX_MESSAGES", 40, 4),
            facts_segment_max_lines=_env_int(env, "MEMORY_FACTS_SEGMENT_MAX_LINES", 10, 1),
            facts_segment_bm25=_env_bool(env, "MEMORY_FACTS_SEGMENT_BM25", True),
            facts_segment_synonyms=_env_bool(env, "MEMORY_FACTS_SEGMENT_SYNONYMS", True),
            facts_segment_rm3=_env_bool(env, "MEMORY_FACTS_SEGMENT_RM3", False),
            facts_segment_vectors=_env_bool(env, "MEMORY_FACTS_SEGMENT_VECTORS", True),
            facts_segment_embedding_model=(
                env.get("MEMORY_FACTS_SEGMENT_EMBEDDING_MODEL") or "BAAI/bge-small-zh-v1.5").strip(),
            facts_events=_env_bool(env, "MEMORY_FACTS_EVENTS", True),
            facts_event_rows=_env_int(env, "MEMORY_FACTS_EVENT_ROWS", 1, 0),
            facts_due=_env_bool(env, "MEMORY_FACTS_DUE", False),
            facts_due_rows=_env_int(env, "MEMORY_FACTS_DUE_ROWS", 2, 0),
            core_facts_max=_env_int(env, "MEMORY_CORE_FACTS_MAX", 30, 5),
            language=(env.get("MEMORY_LANGUAGE") or "zh").strip() or "zh",
            llm_api_key=provider_settings.api_key,
            llm_base_url=provider_settings.base_url,
            llm_model=provider_settings.model,
            # 记忆整合
            consolidation_similarity=_env_float(env, "MEMORY_CONSOLIDATION_SIMILARITY", 0.55, 0.1, 0.95),
            consolidation_min_cluster=_env_int(env, "MEMORY_CONSOLIDATION_MIN_CLUSTER", 2, 2),
            consolidation_min_chunks=_env_int(env, "MEMORY_CONSOLIDATION_MIN_CHUNKS", 5, 2),
            consolidation_use_llm=_env_bool(env, "MEMORY_CONSOLIDATION_USE_LLM", True),
            # 空闲调度器
            idle_threshold_seconds=_env_int(env, "MEMORY_IDLE_THRESHOLD", 300, 30),
            consolidation_interval_seconds=_env_int(env, "MEMORY_CONSOLIDATION_INTERVAL", 1800, 60),
            evolution_interval_seconds=_env_int(env, "MEMORY_EVOLUTION_INTERVAL", 3600, 300),
            token_report_interval_seconds=_env_int(env, "MEMORY_TOKEN_REPORT_INTERVAL", 7200, 300),
            idle_consolidation_enabled=_env_bool(env, "MEMORY_IDLE_CONSOLIDATION_ENABLED", True),
            isolated_max_scopes=_env_int(env, "MEMORY_ISOLATED_MAX_SCOPES", 64, 1),
        )

    @staticmethod
    def _detect_gpu(env: dict[str, str]) -> bool:
        val = env.get("MEMORY_GPU_ENABLED", "").strip().lower()
        if val in {"1", "true", "yes"}:
            return True
        if val in {"0", "false", "no"}:
            return False
        # 自动检测
        cuda_path = os.environ.get("CUDA_PATH", "")
        return bool(cuda_path and os.path.isdir(cuda_path))


# -- 公开工具函数 -----------------------------------------------------------

def default_memory_working_dir(workspace: str | Path) -> str:
    """获取默认记忆存储路径。"""
    return str(Path(workspace).resolve() / DEFAULT_WORKING_DIR)


# -- 私有工具函数 -----------------------------------------------------------

def _resolve_path(value: str | None, *, base: Path, default: Path) -> Path:
    if not value or not value.strip():
        return default
    p = Path(value.strip())
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _env_bool(source: dict[str, str], name: str, default: bool) -> bool:
    v = source.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(source: dict[str, str], name: str, default: int, minimum: int) -> int:
    v = source.get(name)
    if v is None or not v.strip():
        return default
    try:
        return max(int(v), minimum)
    except ValueError:
        return default


def _env_float(source: dict[str, str], name: str, default: float, min_: float, max_: float) -> float:
    v = source.get(name)
    if v is None or not v.strip():
        return default
    try:
        return min(max(float(v), min_), max_)
    except ValueError:
        return default


def _env_choice(source: dict[str, str], name: str, default: str, choices: set[str]) -> str:
    value = (source.get(name) or "").strip().lower()
    return value if value in choices else default
