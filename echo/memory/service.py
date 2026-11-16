from __future__ import annotations

import asyncio
import os
import re
import time
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..models import LLMMessage
from .scope import active_memory_scope, scope_directory_name
from .tier0_short_term import ShortTermMemory
from .tier1_hierarchical import HierarchicalMemory
from .tier2_beam import BeamMemory, NAMESPACE_MEMORY, NAMESPACE_LOG
from .tier3_core import CoreMemory
from .tier_facts import FactsMemory
from .token_tracker import TokenTracker
from .consolidator import MemoryConsolidator
from .evolution import EvolutionAnalyzer
from .idle_scheduler import IdleConsolidationScheduler, SchedulerConfig
from .settings import MemorySettings, MemoryPreparationResult

if TYPE_CHECKING:
    from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _split_context_chunks(messages: list[LLMMessage], token_budget: int) -> list[list[LLMMessage]]:
    chunks: list[list[LLMMessage]] = []
    current: list[LLMMessage] = []
    used = 0
    for message in messages:
        cost = _approx_tokens(str(message.content))
        if current and used + cost > token_budget:
            chunks.append(current)
            current = []
            used = 0
        current.append(message)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def _compress_context_text(text: str, char_budget: int) -> str:
    if len(text) <= char_budget:
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:char_budget]
    selected: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > char_budget:
            break
        selected.append(line)
        used += len(line) + 1
    return "\n".join(selected) or text[:char_budget]


# ---------------------------------------------------------------------------
# MemoryService — 统一 Facade
# ---------------------------------------------------------------------------

class MemoryService:
    """NovaEcho 统一记忆服务

    tier_facts 独立保存带引文的事件事实，在 agent_search 中核对并返回原文证据。

    架构:
    ┌────────────────────────────────────────────────────────────────────┐
    │                         MemoryService                             │
    ├─────────┬──────────┬───────────┬──────────┬──────────┬───────────┤
    │  Tier 0 │  Tier 1  │  Tier 2   │  Tier 3  │Consolid. │ Evolution │
    │  Short  │ Hierarch │  拾忆     │  Core    │ 整合引擎 │ 进化分析  │
    │  Term   │ (hmem)   │(Mnemosyne)│ (Facts)  │          │           │
    ├─────────┴──────────┴───────────┴──────────┴──────────┴───────────┤
    │  注入路径:                                                        │
    │  Roleplay层  → Tier0 recent() + Tier1 L1 + Tier3 facts           │
    │  Agent层     → Tier2 search() + Tier1 drill_down()                │
    │  Decision层  → Tier0 recent(3)                                    │
    │  跨会话       → Tier1 + Tier2 持久化                               │
    │  空闲巩固     → Consolidator + Evolution + TokenTracker           │
    └───────────────────────────────────────────────────────────────────┘

    隔离作用域:
    渠道人格会话（如 qq-mimic）通过 ``novaecho.memory.scope`` 激活作用域上下文，
    此时读写会被路由到该对话对象/群聊专属的轻量实例 —— 独立目录、只启用
    Tier0+Tier1，因此既不会读到主项目记忆，也不会写脏主项目记忆。
    """

    def __init__(self, settings: MemorySettings, *, lightweight: bool = False) -> None:
        self._settings = settings
        self._initialized = False
        self._llm_provider: LLMProvider | None = None
        self._lightweight = lightweight
        self.facts_provider: LLMProvider | None = None
        self.facts = (
            FactsMemory(settings.working_dir / settings.facts_db_name, settings.facts_batch_messages,
                        segment_mode=settings.facts_segment_mode,
                        segment_gap_seconds=settings.facts_segment_gap_seconds,
                        segment_max_messages=settings.facts_segment_max_messages,
                        segment_max_lines=settings.facts_segment_max_lines,
                        segment_bm25=settings.facts_segment_bm25,
                        segment_synonyms=settings.facts_segment_synonyms,
                        segment_rm3=settings.facts_segment_rm3,
                        segment_vectors=settings.facts_segment_vectors,
                        embedding_model=settings.facts_segment_embedding_model)
            if settings.facts_enabled and not lightweight else None
        )
        # 作用域键 → 轻量实例（仅主服务使用）
        self._scoped_services: OrderedDict[str, MemoryService] = OrderedDict()
        # 后台写入任务（Tier2/Tier3）。close()/flush 时必须等待，
        # 否则退出时任务仍在写 sqlite，可能丢数据或关闭已关闭的连接。
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._core_extraction_lock = asyncio.Lock()
        # 初始化 4 层记忆系统及记忆巩固子系统
        wd = settings.working_dir

        self.short_term = ShortTermMemory(
            max_size=settings.short_term_max_size,
        )

        self.hierarchical = HierarchicalMemory(
            db_path=wd / settings.hierarchical_db_name,
            weight_relevance=settings.tier1_weight_relevance,
            weight_importance=settings.tier1_weight_importance,
            weight_recency=settings.tier1_weight_recency,
            bm25_k=settings.tier1_bm25_k,
            recency_halflife_hours=settings.tier1_recency_halflife_hours,
        )

        if lightweight:
            # 隔离作用域只激活 Tier0 + Tier1：不构建 BEAM / Core / 巩固 / 进化 /
            # 空闲调度器，天然不会产生跨会话污染。
            self.beam: BeamMemory | None = None
            self.core: CoreMemory | None = None
            self.token_tracker: TokenTracker | None = None
            self.consolidator: MemoryConsolidator | None = None
            self.evolution: EvolutionAnalyzer | None = None
            self.idle_scheduler: IdleConsolidationScheduler | None = None
            return

        self.beam = BeamMemory(
            data_dir=wd / settings.beam_dir_name,
            backend=settings.beam_backend,
        )

        self.core = CoreMemory(
            db_path=wd / settings.core_db_name,
        )

        # 记忆巩固子系统
        self.token_tracker = TokenTracker(
            db_path=wd / settings.token_db_name,
        )

        self.consolidator = MemoryConsolidator(
            llm_provider=self._llm_provider,
            token_tracker=self.token_tracker,
            similarity_threshold=settings.consolidation_similarity,
            min_cluster_size=settings.consolidation_min_cluster,
            consolidation_min_chunks=settings.consolidation_min_chunks,
        )

        self.evolution = EvolutionAnalyzer(
            llm_provider=self._llm_provider,
            token_tracker=self.token_tracker,
            db_path=wd / settings.evolution_db_name,
        )

        self.idle_scheduler = IdleConsolidationScheduler(
            consolidator=self.consolidator,
            evolution=self.evolution,
            token_tracker=self.token_tracker,
            config=SchedulerConfig(
                idle_threshold_seconds=settings.idle_threshold_seconds,
                consolidation_interval_seconds=settings.consolidation_interval_seconds,
                evolution_interval_seconds=settings.evolution_interval_seconds,
                token_report_interval_seconds=settings.token_report_interval_seconds,
                enabled=settings.idle_consolidation_enabled,
                use_llm=settings.consolidation_use_llm,
            ),
        )
        self.idle_scheduler.set_consolidation_callback(self.consolidate_now)

    def set_llm_provider(self, provider: LLMProvider) -> None:
        """注入主 LLM provider，启用深度归纳能力。"""
        self.configure_consolidation_providers(primary=provider, local=None, mode="1")

    def configure_consolidation_providers(
        self,
        *,
        primary: LLMProvider | None,
        local: LLMProvider | None,
        mode: str,
    ) -> None:
        """统一配置记忆压缩的主 LLM、本地 GGUF 和工作模式。"""
        self._llm_provider = primary
        self.facts_provider = primary or local
        self.consolidator.configure_providers(
            primary=primary,
            local=local,
            mode=mode,
        )
        self.evolution._llm = primary or local

    @property
    def working_dir(self) -> Path:
        return self._settings.working_dir

    # -- 生命周期 ------------------------------------------------------------

    async def ensure_started(self) -> None:
        """初始化所有层。"""
        if self._initialized:
            return

        results = await self._initialize_tiers()
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError("Memory tier initialization failed") from failures[0]
        self._initialized = True

        # 打印摘要
        l1_count = len(await self.hierarchical.get_l1_summaries(max_items=1))
        if self._lightweight:
            logger.info(
                "Scoped memory ready — Tier0(short_term) Tier1(hierarchical) "
                "[L1=%d] @ %s",
                l1_count, self.working_dir,
            )
            return
        facts = await self.core.get_active_facts(max_items=1)
        logger.info(
            "MemoryService ready — "
            "Tier0(short_term) Tier1(hierarchical) Tier2(beam) Tier3(core) "
            "[L1=%d, facts=%d]",
            l1_count, len(facts),
        )

    async def _initialize_tiers(self) -> list[Any]:
        if self._lightweight:
            return list(await asyncio.gather(
                self.hierarchical.initialize(),
                return_exceptions=True,
            ))

        results = await asyncio.gather(
            self.hierarchical.initialize(),
            self.beam.initialize(),
            self.core.initialize(),
            self.token_tracker.initialize(),
            self.evolution.initialize(),
            return_exceptions=True,
        )

        if self.facts is not None:
            try:
                await self.facts.initialize()
            except Exception as exc:
                results.append(exc)

        # 注册消息源回调，使 Evolution 空闲分析能获取对话数据
        self.idle_scheduler.register_message_source(
            lambda: self.short_term.recent(50)
        )

        # 启动空闲调度器
        await self.idle_scheduler.start()
        return results

    async def notify_activity(self) -> None:
        """通知记忆系统有用户活动（重置空闲计时器）。"""
        if self.idle_scheduler is not None:
            self.idle_scheduler.notify_user_activity()

    @staticmethod
    def is_available() -> bool:
        """MemoryService 始终可用 — 没有硬性外部依赖。"""
        return True

    # ====================================================================
    #  注入接口 (供 NovaEcho 三层架构使用)
    # ====================================================================

    # -- Roleplay 层注入 --------------------------------------------------

    def _resolve_scope_target(self) -> "MemoryService":
        """当前若处于隔离作用域，返回该作用域的轻量记忆实例。

        轻量实例自身不再二次路由（``_lightweight`` 直接返回 self）。
        """
        if self._lightweight:
            return self
        scope = active_memory_scope.get()
        if not scope:
            return self
        return self._scoped_service(scope)

    def _scoped_service(self, scope: str) -> "MemoryService":
        """按作用域取（或创建）轻量记忆实例，超出上限时淘汰最久未用者。"""
        directory = scope_directory_name(scope)
        existing = self._scoped_services.get(directory)
        if existing is not None:
            self._scoped_services.move_to_end(directory)
            return existing

        service = MemoryService(
            self._settings.for_isolated_scope(directory),
            lightweight=True,
        )
        self._scoped_services[directory] = service
        limit = max(int(self._settings.isolated_max_scopes), 1)
        while len(self._scoped_services) > limit:
            _, evicted = self._scoped_services.popitem(last=False)
            asyncio.ensure_future(MemoryService._close_quietly(evicted))
        logger.debug("Scoped memory ready for '%s' @ %s", scope, service.working_dir)
        return service

    @staticmethod
    async def _close_quietly(service: "MemoryService") -> None:
        try:
            await service.close()
        except Exception as exc:
            logger.debug("Scoped memory close: %s", exc)

    async def get_roleplay_context(self, max_recent: int = 10) -> str:
        """为 Roleplay 层生成上下文注入字符串（含隔离作用域路由）。"""
        target = self._resolve_scope_target()
        return await target._get_roleplay_context_local(max_recent)

    async def _get_roleplay_context_local(self, max_recent: int = 10) -> str:
        """为 Roleplay 层生成上下文注入字符串。

        包含:
        - Tier 0: 最近对话 (最后 max_recent 条)
        - Tier 1: L1 摘要 (最高重要性)
        - Tier 3: 核心事实 (最高权重, 隔离作用域没有该层)

        返回: 格式化文本，直接注入 Roleplay 系统提示
        """
        await self.ensure_started()

        recent_msgs = await self.short_term.recent(max_recent)
        l1_summaries = await self.hierarchical.get_l1_summaries(
            max_items=self._settings.hierarchical_l1_max,
        )
        core_facts = (
            await self.core.get_active_facts(max_items=self._settings.core_facts_max)
            if self.core is not None
            else []
        )

        parts: list[str] = []

        # P1-3: 添加"取最新值"规则说明
        parts.append("## 记忆系统说明")
        parts.append("当以下记忆中对同一事实有多个值时，优先采用时间戳最新的那个。")
        parts.append("")

        # Tier 3: 核心事实
        if core_facts:
            parts.append("## 核心记忆 (不可遗忘)")
            for f in core_facts:
                parts.append(f"- [{f['category']}] {f['content']}")

        # Tier 1: L1 跨会话摘要
        if l1_summaries:
            parts.append("## 跨会话记忆摘要")
            for s in l1_summaries[:30]:  # 最多 30 条 L1
                parts.append(f"- [{s['type']}/{s['scope']}] {s['l1']}")

        # Tier 0: 最近对话
        if recent_msgs:
            parts.append("## 刚刚的对话")
            for m in recent_msgs:
                parts.append(f"[{m['role']}] {m['content']}")

        return "\n".join(parts)

    async def get_roleplay_messages(self, max_recent: int = 10) -> list[LLMMessage]:
        """为 Roleplay 层返回 LLMMessage 列表。"""
        return await self.short_term.as_messages(max_recent)

    # -- Decision 层注入 ---------------------------------------------------

    async def get_decision_context(self) -> str:
        """为 Decision 层生成轻量上下文（最后 3 条消息）。"""
        recent = await self.short_term.recent(3)
        lines = [f"[{m['role']}] {m['content']}" for m in recent]
        return "\n".join(lines)

    # -- Agent 层接口 -----------------------------------------------------

    async def expand_query(self, query: str) -> str:
        """P0-2: Query expansion — 将代词/模糊时间词展开为具体搜索词。

        例如:
          "她喜欢什么颜色" → "墨墨喜欢什么颜色 用户偏好"
          "上次说的项目"   → "作品集网站 Next.js 星穹互动 项目"
          "那个咖啡馆"     → "晨曦咖啡 永康路 咖啡馆"
        """
        if not self._llm_provider:
            return query
        try:
            from ..models import LLMMessage
            prompt = (
                "你是一个搜索 query 改写器。给定的 query 可能含有代词 "
                "('她/他/它/那个/这个') 或模糊时间词 ('上次/之前/最近/上周')。"
                "重写成更具体的搜索 query，把代词换成可能的具体名词，"
                "把模糊时间词去掉或换成更具体的描述。"
                "保持 1-2 句话，只输出重写后的 query，不解释。\n\n"
                f"原 query: {query}\n重写:"
            )
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self._llm_provider.generate(
                messages, temperature=0.1, max_tokens=64,
            )
            expanded = self._response_text(response)
            if not expanded and getattr(response, "finish_reason", None) == "length":
                # 推理型模型（如 deepseek-flash）会先消耗 token 思考，
                # 64 token 常常只够推理、正文为空且 finish_reason=length。
                # 给一次更大预算重试，避免 query expansion 静默失效。
                response = await self._llm_provider.generate(
                    messages, temperature=0.1, max_tokens=1024,
                )
                expanded = self._response_text(response)
            if expanded:
                logger.debug("Query expansion: '%s' → '%s'", query, expanded)
                return expanded
        except Exception as exc:
            logger.debug("Query expansion failed (fallback to original): %s", exc)
        return query

    @staticmethod
    def _decompose_query(query: str) -> list[str]:
        """把长中文问句拆成短子句，供词法门控严格的记忆后端使用。

        Mnemosyne 3.10 的检索对连续中文按整段 token 做词法门控：一个长问句
        与记忆条目的字符重合率不够时整题 0 命中，而同一个问句拆成子句后
        能正常召回。子句仍走原生检索入口与同一后端，不是第二套检索。
        """
        parts = [p.strip() for p in re.split(r"[，。？！、；：,.?!;:—（）()]", query)]
        clauses = [p for p in parts if 2 <= len(p) <= 24]
        if len(clauses) < 2 and len(query) > 24:
            clauses = [query[i:i + 12] for i in range(0, len(query), 12)]
        return clauses[:6]

    async def _collect_tiers(self, search_query: str, max_results: int,
                             min_score: float,
                             query_time: Any | None = None) -> tuple[list, list, list, list]:
        results = await asyncio.gather(
            self.short_term.search(search_query, max_results),
            self.hierarchical.search(search_query, max_results, min_score,
                                     query_time=query_time),
            self.beam.search(search_query, top_k=max_results, min_score=min_score,
                             namespaces=(NAMESPACE_MEMORY,), query_time=query_time),
            self.core.search(search_query, max_results),
            return_exceptions=True,
        )
        return tuple(r if isinstance(r, list) else [] for r in results)

    def _merge_tiers(self, tier0: list, tier1: list, tier2: list, tier3: list,
                     max_results: int, min_score: float = 0.1) -> list[dict[str, Any]]:
        if max_results <= 0:
            return []
        layers = (
            (tier0, self._settings.fusion_tier0_weight, self._settings.fusion_tier0_reference, "short_term"),
            (tier1, self._settings.fusion_tier1_weight, self._settings.fusion_tier1_reference, "hierarchical"),
            (tier2, self._settings.fusion_tier2_weight, self._settings.fusion_tier2_reference, "beam"),
            (tier3, self._settings.fusion_tier3_weight, self._settings.fusion_tier3_reference, "core"),
        )
        candidates = []
        for rows, weight, reference, tier_name in layers:
            for row in rows:
                raw = float(row.get("score", 0.0))
                if tier_name == "short_term":
                    raw *= 2.0 if row.get("role") == "user" else 1.5
                if raw < min_score:
                    continue
                candidates.append({
                    **row, "raw_score": round(raw, 4), "tier_weight": weight,
                    "score": round(min(1.0, raw / max(0.01, reference)) * weight, 4),
                    "tier": row.get("tier", tier_name),
                })
        seen = set()
        merged = []
        tier1_count = 0
        tier1_cap = min(self._settings.fusion_tier1_max_items, max(1, max_results // 2 + 1))
        for row in sorted(candidates, key=lambda r: (r["score"], r["raw_score"], r.get("timestamp", "")), reverse=True):
            key = row.get("id", "") or "".join(str(row.get("content", ""))[:80].split())
            if key in seen:
                continue
            if row["tier"] == "hierarchical":
                if tier1_count >= tier1_cap:
                    continue
                tier1_count += 1
            seen.add(key)
            merged.append(row)
            if len(merged) >= max_results:
                break
        return merged

    async def agent_search(self, query: str, max_results: int = 5,
                           min_score: float = 0.1,
                           *, query_time: Any | None = None,
                           expand: bool = True) -> list[dict[str, Any]]:
        """Agent 深度检索 (Tier 1 + Tier 2 + Tier 3 混合加权)。

        query_time 是逻辑“现在”；expand=False 时不调用 LLM 扩写，
        供回放/评测做纯检索回归。
        """
        await self.ensure_started()
        if max_results <= 0:
            return []

        # P0-2: Query expansion — 展开代词/模糊时间词
        if expand:
            expanded_query = await self.expand_query(query)
            search_queries = [expanded_query]
            # 扩展 query 只应提升召回，不能把原 query 里的关键实体（人名、地点、
            # 数字）洗掉。实测扩写后偶发 0 命中，因此同时用原 query 检索并合并。
            if expanded_query.strip() != query.strip():
                search_queries.append(query)
        else:
            search_queries = [query]

        candidate_limit = max_results
        tiers: list[list[dict[str, Any]]] = [[], [], [], []]
        for search_query in search_queries:
            for index, rows in enumerate(
                    await self._collect_tiers(
                        search_query, candidate_limit, min_score, query_time=query_time)):
                tiers[index].extend(rows)

        merged = self._merge_tiers(*tiers, max_results, min_score)

        # 长中文问句在 Mnemosyne 的词法门控下会整题 0 命中；召回不足时
        # 再按子句检索同一后端，补足结果。仅在命中不足时触发，正常短问句不受影响。
        if len(merged) < max_results and len(query) > 12:
            clauses = self._decompose_query(query)
            for clause in clauses:
                if clause in search_queries:
                    continue
                search_queries.append(clause)
                for index, rows in enumerate(
                        await self._collect_tiers(
                            clause, candidate_limit, min_score, query_time=query_time)):
                    tiers[index].extend(rows)
            merged = self._merge_tiers(*tiers, max_results, min_score)
            logger.debug("Query decomposed into %d clauses: %s", len(clauses), clauses)

        if self.facts is not None:
            if self._settings.facts_segment_mode:
                # 词法门控会把几乎所有问题判成“已确认”（80题里77题从未查归档），
                # 改为始终查一次归档，只注入得分最高的1条，避免挤占混合检索席位。
                archive = await self.facts.search_segments(
                    query, max_results, query_time=query_time)
                if archive:
                    selected = ([max(archive, key=lambda row: row["score"])]
                                if merged else archive)
                    seen_content = {row["content"] for row in selected}
                    merged = (selected
                              + [row for row in merged if row.get("content") not in seen_content]
                              )[:max_results]
            else:
                fact_rows = await self.facts.search(
                    query, max_results, provider=(self.facts_provider or self._llm_provider) if expand else None,
                    query_time=query_time,
                )
                # 为经原文核对的事实保留一半席位，余下沿用原有融合排序。
                selected = fact_rows[:max(1, max_results // 2)] if max_results > 0 else []
                seen_content = {row["content"] for row in selected}
                merged = (selected + [row for row in merged if row.get("content") not in seen_content])[:max_results]
            if self._settings.facts_events and self._settings.facts_event_rows > 0:
                # 事件卡通道：查询命中原话事件时，补一条事件所在时间片的原文。
                # 只占固定少量席位，避免长原文挤掉融合检索结果。
                events = await self.facts.search_events(
                    query, self._settings.facts_event_rows, query_time=query_time)
                if events:
                    seen_content = {row["content"] for row in events}
                    merged = (events
                              + [row for row in merged if row.get("content") not in seen_content]
                              )[:max_results]
            if self._settings.facts_due and self._settings.facts_due_rows > 0:
                # 到期日历：今天到期的约定优先放在最前，提醒口径由回答层决定。
                due = await self.facts.search_due(query_time, self._settings.facts_due_rows)
                if due:
                    seen_content = {row["content"] for row in due}
                    merged = (due
                              + [row for row in merged if row.get("content") not in seen_content]
                              )[:max_results]
        return merged

    # -- Agent 工具日志检索 / 写入 ---------------------------------------

    async def log_search(self, query: str, top_k: int = 5,
                         min_score: float = 0.1) -> list[dict[str, Any]]:
        """检索 Agent 工具日志层 (namespace="log")。

        供 Agent 自省、技能总结、进化分析使用。
        不会返回对话记忆层的内容。
        """
        await self.ensure_started()
        return await self.beam.search(
            query, top_k=top_k, min_score=min_score,
            namespaces=(NAMESPACE_LOG,),
        )

    async def store_log(self, content: str, *,
                        source: str = "tool",
                        importance: float = 0.3,
                        scope: str = "global") -> str:
        """存储 Agent 工具执行日志到 log 命名空间。

        log 命名空间的内容不会被 roleplay 层和 agent_search 看到，
        仅供 Agent 自省和进化分析使用。
        """
        await self.ensure_started()
        return await self.beam.store_log(
            content, source=source,
            importance=importance, scope=scope,
        )

    # ====================================================================
    #  记忆写入 (各层自动分流)
    # ====================================================================

    async def remember_turn(self, messages: list[LLMMessage], *,
                            event_time: Any | None = None,
                            defer_beam: bool = False) -> None:
        """记录一轮对话（含隔离作用域路由）。

        event_time 是调用方传入的逻辑对话时间；不传则使用当前时间。
        defer_beam=True 时跳过 Tier2 写入，供调用方后续按批写入（评测回放）。
        """
        target = self._resolve_scope_target()
        await target._remember_turn_local(
            messages, event_time=event_time, defer_beam=defer_beam,
        )

    async def remember_beam_batch(self, items: list[dict[str, Any]]) -> None:
        """按批写入 Tier2，供评测回放使用；不改变常规逐轮写入路径。"""
        if self._lightweight or self.beam is None or not items:
            return
        await self.beam.store_messages_batch(items)

    async def _remember_turn_local(self, messages: list[LLMMessage], *,
                                   event_time: str | Any | None = None,
                                   defer_beam: bool = False) -> None:
        """记录一轮对话，自动分流到各层。

        分流规则:
        - Tier 0: 所有消息原文（纯内存，立即返回）
        - Tier 1: 用户偏好/决策类消息 → L1-L5 分层存储
        - Tier 2: 所有消息 → BEAM 语义索引（隔离作用域不写入）
        - 通知空闲调度器有新活动（隔离作用域没有调度器）
        """
        await self.ensure_started()

        # 通知调度器有用户活动
        if self.idle_scheduler is not None:
            self.idle_scheduler.notify_user_activity()

        # Tier 0: 即时写入（纯内存，不阻塞）
        await self.short_term.add_messages(messages)

        fact_sources = []
        # Tier 1: 识别高重要性消息写入 hmem 风格分层
        for msg in messages:
            content = self._extract_text(msg)
            if not content.strip():
                continue

            importance, mtype = self._classify_memory(msg, content)

            # P0-4: 所有消息都进 Tier 1（去除 0.3 阈值），按重要性决定 scope
            evidence_id = await self.hierarchical.store(
                content,
                importance=importance,
                memory_type=mtype,
                scope="cross_session" if importance >= 0.7 else "session",
                source=msg.role or "user",
                tags=self._extract_tags(content),
                event_time=event_time,
            )

            if self.facts is not None:
                from datetime import datetime, timezone
                stamp = (
                    event_time.isoformat() if hasattr(event_time, "isoformat")
                    else str(event_time or datetime.now(timezone.utc).isoformat())
                )
                fact_sources.append({"id": evidence_id, "role": msg.role or "user",
                                     "content": content, "timestamp": stamp})

        if self.facts is not None and os.environ.get(
                "NOVAECHO_MEMORY_FACTS_READONLY") != "1":
            await self.facts.add_sources(fact_sources)
            if not self._settings.facts_defer_extraction:
                self._spawn_background(self._facts_write())

        if self._lightweight:
            # 隔离作用域止步于 Tier 1：不做偏好固化，也不写 BEAM。
            return

        # 偏好提取 → Tier 3 CoreMemory（后台任务）
        self._spawn_background(self._extract_user_preferences(messages))

        # Tier 2: BEAM 存储（后台任务，不阻塞阻塞路径）
        if not defer_beam:
            self._spawn_background(self._store_beam_async(messages, event_time=event_time))

    def _spawn_background(self, coro: Any) -> None:
        """启动后台写入任务并登记，便于 close()/flush 等待其完成。"""
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def flush_background(self) -> None:
        """等待当前所有后台写入落库（测试、答题前快照、退出时使用）。"""
        pending = [task for task in self._background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.facts is not None:
            await self._facts_write(force=True)

    async def _facts_write(self, *, force: bool = False) -> None:
        """按配置写事实索引：分段归档或逐消息抽取原子事实。"""
        if os.environ.get("NOVAECHO_MEMORY_FACTS_READONLY") == "1":
            return
        provider = self.facts_provider or self._llm_provider
        if self._settings.facts_segment_mode:
            await self.facts.consolidate_pending(provider, force=force)
        else:
            await self.facts.extract_pending(provider, force=force)

    async def _store_beam_async(self, messages: list[LLMMessage], *,
                                event_time: Any | None = None) -> None:
        """后台 BEAM 存储。"""
        try:
            await self.beam.store_messages(
                messages, scope="session", event_time=event_time,
            )
        except Exception as exc:
            logger.debug("Beam async store: %s", exc)

    # -- 偏好提取 → Tier 3 CoreMemory -------------------------------------

    _PREFERENCE_PATTERNS: list[tuple[str, str]] = [
        (r"(?:叫我|叫我为|称呼我为?|喊我|叫我一声)\s*(.+?)(?:$|[，。！？\s])", "nickname"),
        (r"(?:不要|别再|不准|别)\s*(?:叫(?:我)?|喊|称呼)\s*(?:我)?\s*(.+?)(?:$|[，。！？\s])", "nickname_exclude"),
        (r"(?:记住|请记住|以后|以后都)\s*(?:叫|称呼|喊|说我?是)\s*(?:我)?\s*(.+?)(?:$|[，。！？\s])", "nickname"),
        # 必须要求"是/叫"，否则 "我的名字。" 会把句号当成昵称。
        (r"(?:我的(?:名字|称呼|称谓))(?:是|叫)\s*(.+?)(?:$|[，。！？\s])", "nickname"),
        (r"(?:我叫|我叫做)\s*(.+?)(?:$|[，。！？\s])", "nickname"),
        (r"(?:叫我|我喜欢你叫我)\s*(.+?)(?:$|[，。！？\s])", "nickname"),
    ]

    @staticmethod
    def _clean_preference_value(value: str) -> str:
        """清洗并校验偏好值，丢弃纯标点/过长/明显的非称呼内容。"""
        import re
        cleaned = value.strip().strip("，。！？、,.;:：!?（）()[]【】\"'“”‘’ ")
        if not cleaned or len(cleaned) > 24:
            return ""
        # 至少包含一个中文字、字母或数字
        if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", cleaned):
            return ""
        # 排除把整句动作误当称呼的情况（如"我叫你把门关上"）
        if any(token in cleaned for token in ("把", "请", "帮", "给", "去", "来", "你", "它", "他", "她", "了", "没", "不", "一声", "两声", "三声")):
            return ""
        return cleaned

    _CORE_PATTERNS = [
        (r"我的生日是\s*(.+)", "birthday", "用户生日: ", "identity"),
        (r"我(?:长期居住在|常住在)\s*(.+)", "residence", "用户常住地: ", "identity"),
        (r"我的职业是\s*(.+)", "occupation", "用户职业: ", "identity"),
        (r"我(?:对话|回复)(?:时)?(?:更)?喜欢\s*(.+)", "response_style", "用户回复偏好: ", "user_preference"),
        (r"我(?:习惯|希望你以后|希望你一直)\s*(.+)", "habit", "用户长期偏好: ", "user_preference"),
        (r"我(?P<negative>不喜欢|喜欢)\s*(?P<value>.+)", "interest", "", "user_preference"),
        (r"(?:请记住|记住)[，,:：]?\s*((?:本项目|这个项目|项目中).+)", "project_rule", "项目约定: ", "project_rule"),
    ]

    @classmethod
    def _iter_core_preferences(cls, text: str) -> list[tuple[str, str, str, tuple[str, ...]]]:
        """只接受完整、直接的用户陈述，不从引文或条件句提升核心事实。"""
        facts = []
        for sentence in re.split(r"[。；;\n]+", text):
            if re.search(r"[?？\"“”‘’`]|如果|假如|假设|比如|例如|也许|可能|听说|他说|她说|今天|这次|暂时|以前|曾经|不再", sentence):
                continue
            for clause in re.split(r"[，,]+", sentence):
                clause = clause.strip()
                if not clause:
                    continue
                excluded = re.fullmatch(r"(?:不要|别再|不准|别)\s*(?:叫我|喊我|称呼我)\s*(.+)", clause)
                nickname = re.fullmatch(r"(?:请记住[：:]?\s*)?(?:以后都?|请)?(?:叫我为?|称呼我为?|喊我|我的名字是|我的名字叫|我叫做?|我喜欢你叫我)\s*(.+)", clause)
                if excluded or nickname:
                    value = cls._clean_preference_value((excluded or nickname).group(1))
                    if value:
                        positive = f"用户偏好称呼: {value}"
                        negative = f"用户不希望被称呼为: {value}"
                        facts.append((f"nickname_exclude:{value}" if excluded else "nickname",
                                      negative if excluded else positive, "user_preference",
                                      (positive,) if excluded else (negative,)))
                    continue
                for pattern, slot, prefix, category in cls._CORE_PATTERNS:
                    match = re.fullmatch(pattern, clause)
                    if not match:
                        continue
                    value = match.groupdict().get("value") or match.group(1)
                    value = value.strip()
                    if not value or len(value) > 96:
                        break
                    conflicts: tuple[str, ...] = ()
                    if slot == "interest":
                        # 每个兴趣独立更新，否定同一兴趣时替换相反事实。
                        slot = f"interest:{value}"
                        positive, negative = f"用户喜欢: {value}", f"用户不喜欢: {value}"
                        content = negative if match.group("negative") == "不喜欢" else positive
                        conflicts = (positive if content == negative else negative,)
                    else:
                        content = prefix + value
                        if slot in {"habit", "project_rule"}:
                            slot += ":" + value
                    facts.append((slot, content, category, conflicts))
                    break
        return facts

    async def _extract_user_preferences(self, messages: list[LLMMessage]) -> None:
        """将明确的稳定用户事实串行更新到核心层，后续改口覆盖同字段。"""
        if self.core is None:
            return
        async with self._core_extraction_lock:
            for msg in messages:
                if msg.role != "user":
                    continue
                for slot, content, category, conflicts in self._iter_core_preferences(self._extract_text(msg)):
                    await self.core.set_auto_fact(
                        slot, content, category=category, weight=0.9 if slot == "nickname" else 0.85,
                        conflicts=conflicts,
                    )

    # ====================================================================
    #  兼容旧接口
    # ====================================================================

    async def search(self, query: str, max_results: int = 5,
                     min_score: float = 0.1) -> dict[str, Any]:
        """统一搜索接口。"""
        results = await self.agent_search(query, max_results, min_score)
        return {
            "content": "\n\n".join(r.get("content", "") for r in results),
            "results": results,
            "count": len(results),
        }

    async def compact_history(self, messages: list[LLMMessage],
                              system_prompt: str | None = None,
                              compressed_summary: str = "") -> MemoryPreparationResult:
        """压缩对话历史（兼容旧接口）。"""
        from .settings import DEFAULT_MAX_INPUT_LENGTH

        budget = self._settings.max_input_length
        keep_count = self._settings.short_term_max_size
        chunks = _split_context_chunks(messages, max(1, budget // 4))

        if len(chunks) <= 1:
            return MemoryPreparationResult(
                messages=messages,
                compressed_summary=compressed_summary or "",
            )
        kept: list[LLMMessage] = []
        kept_tokens = 0
        for message in reversed(chunks[-1]):
            cost = _approx_tokens(self._extract_text(message))
            if kept and kept_tokens + cost > max(1, budget // 4):
                break
            kept.insert(0, message)
            kept_tokens += cost
        if not kept and chunks[-1]:
            message = chunks[-1][-1]
            text = self._extract_text(message)
            kept = [LLMMessage(role=message.role, content=text[: max(1, budget)])]
        prior = "\n".join(
            f"[{m.role}] {self._extract_text(m)}"
            for chunk in chunks[:-1]
            for m in chunk
        )
        summary = compressed_summary.strip()
        if prior:
            summary = "\n".join(part for part in (summary, _compress_context_text(prior, max(1, budget // 8))) if part)
        return MemoryPreparationResult(
            messages=kept,
            compressed_summary=summary or "[auto-compressed]",
        )

    @staticmethod
    def build_summary_message(compressed_summary: str) -> str:
        """构建摘要消息（兼容旧接口）。"""
        if compressed_summary:
            return f"[Previous conversation summary: {compressed_summary}]"
        return ""

    async def close(self) -> str:
        """关闭所有层（含该服务下挂的隔离作用域实例）。"""
        for service in list(self._scoped_services.values()):
            await self._close_quietly(service)
        self._scoped_services.clear()

        # 等待后台写入（Tier2/Tier3）完成，避免关闭连接时仍在写。
        await self.flush_background()

        # 先停空闲调度器
        if self.idle_scheduler is not None:
            await self.idle_scheduler.stop()

        for tier in [self.facts, self.hierarchical, self.beam, self.core,
                     self.token_tracker, self.evolution]:
            if tier is None:
                continue
            try:
                await tier.close()
            except Exception as exc:
                logger.debug("Tier close: %s", exc)
        self._initialized = False
        logger.info("MemoryService closed")
        return "MemoryService closed"

    # ====================================================================
    #  内部工具
    # ====================================================================

    @staticmethod
    def _response_text(response: Any) -> str:
        """提取 LLM 响应正文（兼容 str/list/None）。"""
        content = getattr(getattr(response, "message", None), "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                seg.get("text", "") for seg in content if isinstance(seg, dict)
            ).strip()
        return str(content).strip()

    @staticmethod
    def _extract_text(msg: LLMMessage) -> str:
        if msg.content is None:
            return ""
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            texts = [seg.get("text", "") for seg in msg.content if isinstance(seg, dict)]
            return " ".join(texts)
        return str(msg.content)

    def _classify_memory(self, msg: LLMMessage, content: str) -> tuple[float, str]:
        """分类消息并返回 (importance, memory_type)。"""
        role = (msg.role or "").lower()
        content_lower = content.lower()

        # 高重要性: 用户偏好/决策/project
        if role == "user" and any(kw in content_lower for kw in
                                  ["prefer", "喜欢", "want", "需要", "decide", "决定",
                                   "always", "永远", "never", "don't"]):
            return (0.85, "preference")
        if any(kw in content_lower for kw in
               ["project", "项目", "feature", "功能", "rule", "规则",
                "important", "重要", "remember", "记住"]):
            return (0.75, "decision")
        if role == "user" and len(content) > 50:
            return (0.5, "conversation")
        if role == "assistant":
            importance = 0.4 if len(content) > 100 else 0.2
            return (importance, "response")
        if role == "tool":
            return (0.3, "tool_result")

        return (0.3, "conversation")

    @staticmethod
    def _extract_tags(content: str) -> list[str]:
        """从内容中提取标签（简单实现）。"""
        tags = []
        content_lower = content.lower()
        topic_keywords = {
            "programming": ["code", "programming", "function", "api", "bug", "debug",
                           "python", "javascript", "typescript"],
            "writing": ["write", "article", "document", "note", "memo"],
            "project": ["project", "feature", "milestone", "deploy"],
            "personal": ["prefer", "like", "want", "need", "hate"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(topic)
        return tags[:3]

    # ====================================================================
    #  记忆巩固 & 进化分析接口
    # ====================================================================

    async def consolidate_now(self, *, use_llm: bool | None = None) -> Any:
        """手动触发记忆整合。"""
        await self.ensure_started()
        result = await self.consolidator.run_full_consolidation(
            hierarchical=self.hierarchical,
            beam=self.beam,
            use_llm=use_llm if use_llm is not None else self.idle_scheduler._config.use_llm,
        )
        self.idle_scheduler._state.last_consolidation = time.monotonic()
        self.idle_scheduler._state.total_consolidations += 1
        return result

    async def evolve_now(self) -> Any:
        """手动触发进化分析。"""
        await self.ensure_started()
        # 注入对话历史源
        self.idle_scheduler.register_message_source(
            lambda: self.short_term.recent(50)
        )
        return await self.idle_scheduler.trigger_evolution_now()

    async def get_token_usage(self) -> dict[str, Any]:
        """获取 token 消耗统计。"""
        await self.ensure_started()
        return await self.token_tracker.get_session_summary()

    async def get_memory_stats(self) -> dict[str, Any]:
        await self.ensure_started()
        stats = await self.beam.get_stats()
        if self.facts is not None:
            stats["tier_facts"] = self.facts.get_stats()
        return stats

    async def get_token_usage_historical(
        self, operation: str | None = None, since_hours: int = 24
    ) -> list[dict[str, Any]]:
        """获取历史 token 消耗。"""
        await self.ensure_started()
        return await self.token_tracker.get_historical_summary(
            operation=operation, since_hours=since_hours,
        )

    async def get_token_usage_total(self) -> dict[str, Any]:
        """获取全历史 token 总量。"""
        await self.ensure_started()
        return await self.token_tracker.get_total_all_time()

    async def get_user_patterns(
        self, pattern_type: str | None = None, min_confidence: float = 0.0
    ) -> list[dict[str, Any]]:
        """获取用户模式。"""
        await self.ensure_started()
        return await self.evolution.get_patterns(
            pattern_type=pattern_type, min_confidence=min_confidence,
        )

    async def get_scheduler_state(self) -> dict[str, Any]:
        """获取空闲调度器状态。"""
        return self.idle_scheduler.get_state()
