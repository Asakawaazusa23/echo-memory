"""MemoryConsolidator — 记忆整合引擎。

核心职责:
1. 向量聚类: 基于文本相似度 (TF-IDF + Jaccard) 将相似记忆分组
2. LLM 深度归纳: 对高相似度记忆组，调用 LLM 生成整合摘要
3. 整合写回: 替换原始碎片记忆为整合后的摘要
4. 跨时间压缩: 旧的、低重要性的碎片记忆自动合并

设计原则:
- 不依赖外部 embedding 模型（Mnemosyne 管理自己的向量）
- 使用轻量 TF-IDF + Jaccard 做聚类，纯 Python 实现
- LLM 调用通过注入的 provider 完成
- 所有操作异步，不阻塞主路径
- Token 消耗通过 TokenTracker 追踪
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import LLMProvider
    from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MemoryChunk:
    """待整合的记忆片段。"""
    entry_id: str
    content: str
    importance: float
    memory_type: str
    scope: str
    source: str
    timestamp: str
    tier: str  # "hierarchical", "beam", "core"

    @property
    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class Cluster:
    """一个相似记忆聚类。"""
    cluster_id: int
    chunks: list[MemoryChunk] = field(default_factory=list)
    centroid_tfidf: dict[str, float] = field(default_factory=dict)
    avg_importance: float = 0.0

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def combined_text(self) -> str:
        parts = []
        for c in self.chunks:
            parts.append(f"[{c.source}/{c.memory_type}] {c.content[:300]}")
        return "\n".join(parts)


@dataclass
class ConsolidationResult:
    """一次整合操作的结果。"""
    clusters_found: int
    memories_merged: int
    memories_removed: int
    summaries_generated: int
    token_usage: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TF-IDF 轻量实现
# ---------------------------------------------------------------------------

_CN_STOP_WORDS = frozenset(
    "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 "
    "会 着 没有 看 好 自己 这 他 她 它 们 那 被 从 把 让 给 对 但 而 但 "
    "如果 因为 所以 虽然 可以 这个 那个 什么 怎么 如何 些 还 呢 吧 啊 呀 "
    "the a an is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "under again further then once here there when where why how all both "
    "each few more most other some such no nor not only own same so than "
    "too very just don now".split()
)


def _parse_timestamp(value: Any) -> float | None:
    """把记忆条目的 timestamp 解析成 epoch 秒，无法解析时返回 None。

    Tier2 两个后端的 timestamp 形态不一致：fallback 写 ISO 字符串，
    mnemosyne 可能是 ISO 字符串或数值，因此这里统一容错解析。
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _tokenize(text: str) -> list[str]:
    """中英文混合分词: 英文按空格, 中文按字符对。"""
    text = text.lower()
    # 提取英文单词
    en_words = re.findall(r"[a-z0-9_]+", text)
    # 提取中文双字组合 (bigram)
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cn_bigrams = [cn_chars[i] + cn_chars[i + 1]
                  for i in range(len(cn_chars) - 1)]
    # 单个中文字符也保留
    tokens = en_words + cn_bigrams + cn_chars
    return [t for t in tokens if t not in _CN_STOP_WORDS and len(t) > 1]


def _compute_tfidf(texts: list[str]) -> list[dict[str, float]]:
    """计算一组文本的 TF-IDF 向量。"""
    n = len(texts)
    if n == 0:
        return []

    # 分词
    tokenized = [_tokenize(t) for t in texts]

    # 文档频率
    df: dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    # TF-IDF for each doc
    results = []
    for tokens in tokenized:
        if not tokens:
            results.append({})
            continue
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = len(tokens)
        idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in tf}
        tfidf = {t: (tf[t] / total) * idf[t] for t in tf}
        # 归一化
        norm = math.sqrt(sum(v * v for v in tfidf.values())) or 1.0
        tfidf = {t: v / norm for t, v in tfidf.items()}
        results.append(tfidf)

    return results


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """两个稀疏向量的余弦相似度。"""
    if not a or not b:
        return 0.0
    # 只遍历较小的字典
    smaller, larger = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * larger.get(k, 0) for k, v in smaller.items())
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (norm_a * norm_b)


def _jaccard_sim(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Jaccard 相似度。"""
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# 聚类算法
# ---------------------------------------------------------------------------

def _cluster_chunks(
    chunks: list[MemoryChunk],
    *,
    similarity_threshold: float = 0.45,
    min_cluster_size: int = 2,
) -> list[Cluster]:
    """基于文本相似度的贪心聚类。

    算法:
    1. 计算所有 chunks 的 TF-IDF 向量
    2. 贪心合并: 遍历每个 chunk，找到最相似的已有聚类
    3. 如果相似度 > threshold，加入该聚类；否则创建新聚类
    """
    if len(chunks) < 2:
        return []

    texts = [c.text for c in chunks]
    tfidf_vectors = _compute_tfidf(texts)
    token_lists = [_tokenize(t) for t in texts]

    clusters: list[Cluster] = []

    for i, chunk in enumerate(chunks):
        best_cluster_idx = -1
        best_sim = 0.0

        for ci, cluster in enumerate(clusters):
            # 与聚类质心的相似度
            sim_tfidf = _cosine_sim(tfidf_vectors[i], cluster.centroid_tfidf)
            # 与聚类内最近成员的 Jaccard
            max_jaccard = max(
                (_jaccard_sim(token_lists[i], token_lists[cj])
                 for cj in range(len(chunks))
                 if any(cc.entry_id == chunks[cj].entry_id for cc in cluster.chunks)),
                default=0.0,
            )
            # 综合相似度
            sim = 0.6 * sim_tfidf + 0.4 * max_jaccard
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = ci

        if best_sim >= similarity_threshold and best_cluster_idx >= 0:
            # 加入现有聚类
            cluster = clusters[best_cluster_idx]
            cluster.chunks.append(chunk)
            # 更新质心 (增量平均)
            n = len(cluster.chunks)
            old_weight = (n - 1) / n
            new_weight = 1.0 / n
            new_centroid: dict[str, float] = defaultdict(float)
            all_keys = set(cluster.centroid_tfidf) | set(tfidf_vectors[i])
            for k in all_keys:
                new_centroid[k] = (cluster.centroid_tfidf.get(k, 0) * old_weight
                                   + tfidf_vectors[i].get(k, 0) * new_weight)
            cluster.centroid_tfidf = dict(new_centroid)
            cluster.avg_importance = (
                (cluster.avg_importance * (n - 1) + chunk.importance) / n
            )
        else:
            # 创建新聚类
            clusters.append(Cluster(
                cluster_id=len(clusters),
                chunks=[chunk],
                centroid_tfidf=tfidf_vectors[i].copy(),
                avg_importance=chunk.importance,
            ))

    # 过滤太小的聚类
    return [c for c in clusters if c.size >= min_cluster_size]


# 高重要性保护阈值 — 高于此值的单条不参与聚类整合
_PROTECTED_IMPORTANCE_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# LLM 归纳 Prompt
# ---------------------------------------------------------------------------

_CONSOLIDATION_SYSTEM_PROMPT = """你是一个记忆整合助手。你的任务是将多条相关的记忆片段整合为一条精炼的摘要。

规则:
1. 保留所有关键事实和偏好信息
2. 去除重复内容
3. 用简洁的中文总结
4. 如果存在矛盾信息，保留最新的/更具体的
5. 输出长度控制在 100-200 字以内
6. 保持客观，不添加记忆中没有的信息"""


def _build_consolidation_prompt(cluster: Cluster) -> str:
    """构建整合归纳的 user prompt。"""
    lines = [f"以下是 {cluster.size} 条相关记忆，请整合为一条摘要:\n"]
    for i, chunk in enumerate(cluster.chunks, 1):
        lines.append(f"--- 记忆 {i} (重要性:{chunk.importance:.1f}, "
                     f"类型:{chunk.memory_type}) ---")
        lines.append(chunk.content[:500])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """记忆整合引擎 — 记忆的"睡眠巩固"。

    工作流:
    1. collect() — 从各层收集待整合的记忆片段
    2. cluster() — 基于 TF-IDF + Jaccard 聚类
    3. consolidate() — 调用 LLM 对每个聚类生成整合摘要
    4. apply() — 删除原始碎片，写入整合后的新记忆
    5. report() — 返回整合结果和 token 消耗

    触发时机:
    - 对话间隙 (idle consolidation)
    - 手动触发
    - 定期定时触发
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider | None = None,
        local_provider: LLMProvider | None = None,
        token_tracker: TokenTracker | None = None,
        similarity_threshold: float = 0.45,
        min_cluster_size: int = 2,
        max_consolidation_batch: int = 100,
        consolidation_min_chunks: int = 2,
        memory_choice: str = "1",
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._llm = llm_provider
        self._local_llm = local_provider
        self._token_tracker = token_tracker
        self._similarity_threshold = similarity_threshold
        self._min_cluster_size = min_cluster_size
        self._max_batch = max_consolidation_batch
        self._min_chunks = consolidation_min_chunks
        self._memory_choice = str(memory_choice).strip() or "1"
        self._progress_callback = progress_callback

        # 内存中的待整合片段
        self._pending_chunks: list[MemoryChunk] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._stop_event = asyncio.Event()

    def configure_providers(
        self,
        *,
        primary: LLMProvider | None,
        local: LLMProvider | None,
        mode: str,
    ) -> None:
        """配置记忆整合所使用的主模型、本地模型和模式。"""
        self._llm = primary
        self._local_llm = local
        self._memory_choice = str(mode).strip() or "1"

    def stop(self) -> None:
        """请求停止当前正在进行的整合任务。"""
        self._stop_event.set()

    def _reset_stop(self) -> None:
        """重置停止标志（开始新任务前调用）。"""
        self._stop_event.clear()

    def _check_stop(self, msg: str = "consolidation stopped by user") -> None:
        """检查是否收到停止请求，若收到则抛出异常中断流程。"""
        if self._stop_event.is_set():
            raise asyncio.CancelledError(msg)

    def _report_progress(self, stage: str, extra: dict | None = None) -> None:
        """向外部报告当前阶段。"""
        if self._progress_callback:
            try:
                self._progress_callback(stage, extra or {})
            except Exception:
                pass

    # -- 噪声过滤 -----------------------------------------------------------

    def _filter_noise(self, chunks: list[MemoryChunk]) -> list[MemoryChunk]:
        """预过滤低价值记忆片段，减少后续 LLM 调用量。

        过滤规则:
        1. 内容过短 (< 8 字符)
        2. 纯寒暄 / 无意义 ack
        3. 与上一条内容高度重复 (Jaccard > 0.85)
        """
        kept: list[MemoryChunk] = []
        last_text = ""
        for c in chunks:
            text = c.content.strip()
            if len(text) < 8:
                continue
            if self._is_greeting_or_ack(text):
                continue
            if last_text:
                sim = _jaccard_sim(_tokenize(text), _tokenize(last_text))
                if sim > 0.85:
                    continue
            kept.append(c)
            last_text = text
        if len(kept) < len(chunks):
            logger.info("Noise filter removed %d chunks (%d -> %d)",
                        len(chunks) - len(kept), len(chunks), len(kept))
        return kept

    @staticmethod
    def _is_greeting_or_ack(text: str) -> bool:
        """判断是否为寒暄或无意义确认。"""
        # 中文常见寒暄
        cn_greetings = {
            "你好", "您好", "哈喽", "嗨", "在吗", "在的", "嗯", "嗯嗯", "哦", "哦哦",
            "好的", "好", "行", "可以", "知道了", "明白", "了解", "谢谢", "多谢",
            "不客气", "没事", "没关系", "再见", "拜拜", "回头见",
        }
        # 英文常见寒暄
        en_greetings = {
            "hello", "hi", "hey", "ok", "okay", "yes", "yeah", "no", "nope",
            "thanks", "thank you", "sorry", "bye", "goodbye", "see you",
        }
        lowered = text.lower().strip("。？！.,!?~ ")
        if lowered in cn_greetings or lowered in en_greetings:
            return True
        # 过短且主要是标点/语气词
        if len(lowered) <= 3 and not re.search(r"[\u4e00-\u9fff]", lowered):
            return True
        return False

    # -- 激进清理阶段（直接删除低价值碎片） ---------------------------------

    async def aggressive_cleanup(
        self,
        *,
        hierarchical: Any | None = None,
        beam: Any | None = None,
        cleanup_min_importance: float = 0.30,
        cleanup_protected_types: tuple = ("preference", "role_knowledge"),
        max_age_hours: int = 168,  # 默认 7 天内才清理
        tier2_memory_max_age_hours: int = 0,  # 默认不主动删 Tier2 memory（由聚类处理）
    ) -> int:
        """激进清理低价值碎片，直接批量删除（无需 LLM）。

        删除条件:
        1. Tier 2 namespace='log' 的临时日志（默认全部删除）
        2. Tier 2 namespace='memory' 超过 tier2_memory_max_age_hours 的旧对话
        3. Tier 1 importance < cleanup_min_importance 的碎片
           (排除 protected_types: preference / role_knowledge)
        4. Tier 1 内容为空或极短（< 8 字符）
        5. Tier 1 内容纯重复 ack/寒暄

        Returns: 删除的条数。
        """
        removed = 0

        # 1. 清空 Tier 2 namespace=log（这些是临时日志）
        # 通过 BeamMemory 的后端无关接口操作，mnemosyne / fallback 均生效。
        if beam is not None and hasattr(beam, "list_entries"):
            try:
                entries = await beam.list_entries(limit=100000)
                log_ids = [
                    e["id"] for e in entries
                    if (e.get("namespace") or "memory") == "log"
                ]
                if log_ids:
                    deleted = await beam.delete_entries(log_ids)
                    removed += deleted
                    logger.info("Aggressive cleanup: removed %d Tier2 log entries", deleted)

                # b) 删除 namespace='memory' 超过 N 天的旧对话历史
                if tier2_memory_max_age_hours > 0:
                    cutoff = time.time() - tier2_memory_max_age_hours * 3600
                    stale_ids = []
                    for e in entries:
                        if (e.get("namespace") or "memory") != "memory":
                            continue
                        ts = _parse_timestamp(e.get("timestamp"))
                        if ts is not None and ts < cutoff:
                            stale_ids.append(e["id"])
                    if stale_ids:
                        deleted = await beam.delete_entries(stale_ids)
                        removed += deleted
                        logger.info(
                            "Aggressive cleanup: removed %d Tier2 memory entries "
                            "older than %d hours",
                            deleted, tier2_memory_max_age_hours,
                        )
            except Exception as exc:
                logger.warning("Failed to cleanup Tier2: %s", exc)

        # 2. 清理 Tier 1 低价值记忆
        if hierarchical and hierarchical._conn is not None:
            try:
                placeholders = ",".join("?" * len(cleanup_protected_types))
                # a) 低重要性 + 非受保护类型 + 较旧
                sql_low_imp = f"""
                    DELETE FROM hierarchical_entries
                    WHERE importance < ?
                      AND memory_type NOT IN ({placeholders})
                      AND (julianday('now') - julianday(timestamp)) > ?
                """
                loop = asyncio.get_running_loop()
                max_age_days = max_age_hours / 24
                cursor = await loop.run_in_executor(
                    None,
                    lambda: hierarchical._conn.execute(
                        sql_low_imp,
                        (cleanup_min_importance,
                         *cleanup_protected_types,
                         max_age_days),
                    ),
                )
                hierarchical._conn.commit()
                removed += cursor.rowcount
                logger.info("Aggressive cleanup: removed %d Tier1 low-importance entries",
                            cursor.rowcount)

                # b) 内容极短或为空
                sql_empty = """
                    DELETE FROM hierarchical_entries
                    WHERE length(trim(coalesce(l1_summary, ''))) < 4
                       OR (coalesce(l3_full, '') = ''
                           AND coalesce(l1_summary, '') = ''
                           AND coalesce(l2_detail, '') = '')
                """
                cursor = await loop.run_in_executor(
                    None,
                    lambda: hierarchical._conn.execute(sql_empty),
                )
                hierarchical._conn.commit()
                removed += cursor.rowcount
                logger.info("Aggressive cleanup: removed %d empty/short Tier1 entries",
                            cursor.rowcount)

            except Exception as exc:
                logger.warning("Failed to cleanup Tier1: %s", exc)

        return removed

    # -- 类型分组整合（不依赖相似度）----------------------------------------

    async def type_grouped_consolidation(
        self,
        chunks: list[MemoryChunk],
        *,
        hierarchical: Any | None = None,
        beam: Any | None = None,
        time_window_minutes: int = 30,
        min_group_size: int = 3,
    ) -> tuple[list[MemoryChunk], list[tuple[Cluster, str]]]:
        """按 memory_type + 时间窗口分组整合相似记忆。

        策略: 同一 memory_type 在 time_window_minutes 内的多条记忆，
        即使文本相似度不够高，也按时间聚类后合并为 L1 摘要。

        Returns: (剩余未整合的 chunks, [(cluster, summary), ...])
        """
        if not chunks or hierarchical is None:
            return chunks, []

        # 按 (tier, memory_type) 分组
        groups: dict[tuple, list[MemoryChunk]] = defaultdict(list)
        for c in chunks:
            groups[(c.tier, c.memory_type)].append(c)

        merged_ids: set[str] = set()
        summaries: list[tuple[Cluster, str]] = []
        kept_chunks: list[MemoryChunk] = []

        for (tier, mtype), group in groups.items():
            # 跳过受保护类型
            if mtype in ("preference", "role_knowledge"):
                kept_chunks.extend(group)
                continue
            # 太少不分
            if len(group) < min_group_size:
                kept_chunks.extend(group)
                continue

            # 按时间窗口进一步分桶
            buckets: list[list[MemoryChunk]] = []
            sorted_group = sorted(
                group,
                key=lambda c: c.timestamp or "",
            )
            current_bucket: list[MemoryChunk] = []
            last_ts: float | None = None
            for c in sorted_group:
                try:
                    ts = datetime.fromisoformat(c.timestamp.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = time.time() if last_ts is None else last_ts

                if last_ts is None or (ts - last_ts) <= time_window_minutes * 60:
                    current_bucket.append(c)
                else:
                    if current_bucket:
                        buckets.append(current_bucket)
                    current_bucket = [c]
                last_ts = ts
            if current_bucket:
                buckets.append(current_bucket)

            # 对每个 bucket 做整合
            for bucket in buckets:
                if len(bucket) < min_group_size:
                    kept_chunks.extend(bucket)
                    continue
                # 生成 group 摘要
                summary_text = self._make_group_summary(bucket, mtype)
                cluster = Cluster(
                    cluster_id=len(summaries),
                    chunks=bucket,
                    avg_importance=sum(c.importance for c in bucket) / len(bucket),
                )
                summaries.append((cluster, summary_text))
                for c in bucket:
                    merged_ids.add(c.entry_id)

        # 把未参与的 chunk 保留
        kept_chunks = [c for c in chunks if c.entry_id not in merged_ids]

        return kept_chunks, summaries

    @staticmethod
    def _make_group_summary(bucket: list[MemoryChunk], memory_type: str) -> str:
        """为同类型时间窗口的分组生成轻量摘要（无 LLM）。"""
        # 取每条 L1 摘要的关键句
        parts: list[str] = []
        seen: set[str] = set()
        for c in bucket:
            text = c.content[:80].strip() if c.content else ""
            if not text:
                continue
            # 简单去重
            key = text[:30]
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
            if len(parts) >= 5:
                break

        if not parts:
            return f"[{memory_type}] {len(bucket)} 条记录已合并"

        header = f"[{memory_type}×{len(bucket)}]"
        body = " | ".join(parts)
        return f"{header} {body}"[:400]

    # -- 收集待整合记忆 -----------------------------------------------------

    async def collect_from_hierarchical(
        self, hierarchical: Any,  # HierarchicalMemory
        *,
        max_age_hours: int = 48,
        min_importance: float = 0.0,
        max_importance: float = 0.75,
        max_entries: int = 2000,
    ) -> list[MemoryChunk]:
        """从 Tier 1 收集待整合的记忆。

        策略: 收集中等重要性 (0.0-0.75) 的近期记忆，这些最适合整合。
        高重要性的记忆 (>0.75) 通常需要保留原始形式。
        """
        chunks = []
        async with hierarchical._lock:
            if hierarchical._conn is None:
                return chunks
            rows = hierarchical._conn.execute(
                """SELECT entry_id, l1_summary, l3_full, importance,
                          memory_type, scope, source, timestamp
                   FROM hierarchical_entries
                   WHERE importance >= ? AND importance <= ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (min_importance, max_importance, max_entries),
            ).fetchall()

        for r in rows:
            content = r[2] or r[1]  # 优先用完整内容，fallback 到摘要
            chunks.append(MemoryChunk(
                entry_id=r[0],
                content=content,
                importance=r[3],
                memory_type=r[4],
                scope=r[5],
                source=r[6],
                timestamp=r[7],
                tier="hierarchical",
            ))

        logger.debug("Collected %d chunks from hierarchical", len(chunks))
        return chunks

    async def collect_from_beam(
        self, beam: Any,  # BeamMemory
        *,
        max_entries: int = 2000,
        exclude_log: bool = True,
    ) -> list[MemoryChunk]:
        """从 Tier 2 收集待整合的记忆。

        exclude_log: 是否排除 namespace='log' 的临时日志记忆（这些不参与整合，直接清空）。
        """
        chunks: list[MemoryChunk] = []
        if beam is None or not hasattr(beam, "list_entries"):
            return chunks

        # 走 BeamMemory 的后端无关接口：mnemosyne 与 FTS5 fallback 都能正常枚举。
        # （历史实现直接读 _fallback._conn，mnemosyne 生效后 _fallback 为 None，
        #  会导致整个 Tier2 被静默跳过。）
        try:
            entries = await beam.list_entries(
                limit=max_entries,
                exclude_namespace="log" if exclude_log else None,
            )
        except Exception as exc:
            logger.warning("Failed to collect from beam: %s", exc)
            return chunks

        for e in entries:
            chunks.append(MemoryChunk(
                entry_id=str(e["id"]),
                content=e.get("content") or "",
                importance=e.get("importance", 0.5),
                memory_type="conversation",
                scope=e.get("scope") or "session",
                source=e.get("source") or "agent",
                timestamp=e.get("timestamp") or "",
                tier="beam",
            ))

        logger.debug("Collected %d chunks from beam (exclude_log=%s)",
                     len(chunks), exclude_log)
        return chunks

    def add_manual_chunk(self, chunk: MemoryChunk) -> None:
        """手动添加待整合的记忆片段。"""
        self._pending_chunks.append(chunk)

    # -- 聚类 ----------------------------------------------------------------

    async def cluster(self, chunks: list[MemoryChunk] | None = None) -> list[Cluster]:
        """对记忆片段进行聚类。"""
        target = chunks or self._pending_chunks
        if len(target) < self._min_chunks:
            logger.debug("Not enough chunks to cluster (%d < %d)",
                         len(target), self._min_chunks)
            return []

        return _cluster_chunks(
            target,
            similarity_threshold=self._similarity_threshold,
            min_cluster_size=self._min_cluster_size,
        )

    # -- LLM 深度归纳 -------------------------------------------------------

    async def consolidate_with_llm(
        self, clusters: list[Cluster]
    ) -> list[tuple[Cluster, str]]:
        """对每个聚类生成整合摘要，支持三种 LLM 模式。

        - choice=1: 仅主 LLM
        - choice=2: 本地 GGUF 草稿 + 主 LLM 精修（高重要性/低置信度才精修）
        - choice=3: 仅本地 GGUF

        返回: [(cluster, summary), ...]
        """
        results: list[tuple[Cluster, str]] = []
        batch = clusters[:self._max_batch]
        total = len(batch)

        for i, cluster in enumerate(batch):
            self._check_stop()
            self._report_progress(
                "summarizing",
                {
                    "cluster_current": i + 1,
                    "cluster_total": total,
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size,
                    "message": f"正在生成第 {i + 1}/{total} 个聚类摘要 (cluster_id={cluster.cluster_id}, size={cluster.size})",
                },
            )
            try:
                summary = await self._summarize_cluster(cluster)
                if summary:
                    results.append((cluster, summary))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("LLM consolidation failed for cluster %d: %s",
                               cluster.cluster_id, exc)

        return results

    async def _summarize_cluster(self, cluster: Cluster) -> str:
        """根据当前 memory_choice 选择摘要策略。"""
        if self._memory_choice == "3":
            # 纯本地 GGUF
            if self._local_llm is None:
                logger.warning("No local provider available for pure-local consolidation")
                return ""
            return await self._summarize_cluster_local(cluster)

        if self._memory_choice == "2":
            # 本地先过一轮
            if self._local_llm is None:
                logger.warning("No local provider available for local-first consolidation")
                return await self._summarize_cluster_remote(cluster)
            local_summary = await self._summarize_cluster_local(cluster)
            if not local_summary:
                # 本地没出字（模型加载失败/只输出了推理块）时退回主 LLM，
                # 否则这一簇会被静默丢弃。
                return await self._summarize_cluster_remote(cluster)
            if await self._should_refine_cluster(cluster, local_summary):
                refined = await self._summarize_cluster_remote(
                    cluster, draft=local_summary)
                # 主 LLM 不可用或精修失败时保留本地草稿，不能白丢。
                return refined or local_summary
            return local_summary

        # 默认：主 LLM
        if self._llm is None:
            logger.warning("No LLM provider available for consolidation")
            return ""
        return await self._summarize_cluster_remote(cluster)

    async def _summarize_cluster_remote(
        self,
        cluster: Cluster,
        *,
        draft: str | None = None,
    ) -> str:
        """调用主 LLM 归纳单个聚类，可传入本地草稿作为参考。"""
        from ..models import LLMMessage

        if self._llm is None:
            return ""

        user_prompt = _build_consolidation_prompt(cluster)
        if draft:
            user_prompt = (
                f"本地模型已生成参考摘要：\n{draft}\n\n"
                f"请基于以下原始记忆进行精修，保留关键事实，去除冗余：\n\n{user_prompt}"
            )

        messages = [
            LLMMessage(role="system", content=_CONSOLIDATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        start = time.monotonic()
        response = await self._llm.generate(
            messages,
            temperature=0.3,
            max_tokens=512,
        )
        duration_ms = (time.monotonic() - start) * 1000

        # 记录 token 消耗
        if self._token_tracker:
            await self._token_tracker.record_from_response(
                "consolidation",
                response.usage,
                model=response.model,
                duration_ms=duration_ms,
                detail=f"cluster_id={cluster.cluster_id} size={cluster.size} mode=remote",
            )

        content = self._response_text(response)
        if not content and getattr(response, "finish_reason", None) == "length":
            # 推理型模型先消耗 token 思考，512 预算可能只出推理不出正文。
            # 这种情况下重试一次更大预算，否则该聚类会被静默丢弃。
            retry_start = time.monotonic()
            response = await self._llm.generate(
                messages,
                temperature=0.3,
                max_tokens=1536,
            )
            retry_duration = (time.monotonic() - retry_start) * 1000
            if self._token_tracker:
                await self._token_tracker.record_from_response(
                    "consolidation",
                    response.usage,
                    model=response.model,
                    duration_ms=retry_duration,
                    detail=(
                        f"cluster_id={cluster.cluster_id} size={cluster.size} "
                        "mode=remote retry"
                    ),
                )
            content = self._response_text(response)
        return content

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

    async def _summarize_cluster_local(self, cluster: Cluster) -> str:
        """调用本地 GGUF 模型生成草稿摘要。"""
        from ..models import LLMMessage

        if self._local_llm is None:
            return ""

        messages = [
            LLMMessage(role="system", content=_CONSOLIDATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=_build_consolidation_prompt(cluster)),
        ]

        start = time.monotonic()
        try:
            response = await self._local_llm.generate(
                messages,
                temperature=0.0,
                # 本地小模型（MiniCPM/Qwen）会先输出 <think> 推理块，
                # 预算太小会导致推理没写完就截断、正文一个字都没有。
                # provider 会剥离推理块，这里只需给足预算。
                max_tokens=768,
            )
        except Exception as exc:
            logger.warning("Local summarization failed for cluster %d: %s",
                           cluster.cluster_id, exc)
            return ""
        duration_ms = (time.monotonic() - start) * 1000

        # 记录 token 消耗（本地模型也可能有 usage）
        if self._token_tracker and response.usage:
            await self._token_tracker.record_from_response(
                "consolidation_local",
                response.usage,
                model=response.model or "local",
                duration_ms=duration_ms,
                detail=f"cluster_id={cluster.cluster_id} size={cluster.size} mode=local",
            )

        content = response.message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [seg.get("text", "") for seg in content if isinstance(seg, dict)]
            return " ".join(texts).strip()
        return str(content).strip()

    async def _should_refine_cluster(self, cluster: Cluster, local_summary: str) -> bool:
        """判断本地摘要是否需要主 LLM 精修。

        精修条件:
        1. 聚类平均重要性较高
        2. 本地摘要过短或过长
        3. 本地摘要包含不确定表述
        """
        # 高重要性聚类需要精修
        if cluster.avg_importance >= 0.65:
            return True
        # 摘要过短或过长
        if len(local_summary) < 40 or len(local_summary) > 300:
            return True
        # 包含不确定表述
        uncertain = ["不知道", "不清楚", "无法", "不能确定", "maybe", "not sure", "unclear"]
        lowered = local_summary.lower()
        if any(u in lowered for u in uncertain):
            return True
        return False

    # -- 轻量规则整合 (无 LLM) -----------------------------------------------

    async def consolidate_with_rules(
        self, clusters: list[Cluster]
    ) -> list[tuple[Cluster, str]]:
        """不使用 LLM 的规则整合: 截断 + 优先级排序。

        适合 LLM 不可用时的降级方案。
        """
        results: list[tuple[Cluster, str]] = []
        for cluster in clusters:
            self._check_stop()
            # 按重要性降序排列
            sorted_chunks = sorted(cluster.chunks,
                                   key=lambda c: c.importance, reverse=True)
            # 取最重要的几条内容拼合
            parts = []
            seen = set()
            for chunk in sorted_chunks[:5]:
                text = chunk.content[:200].strip()
                # 简单去重
                fingerprint = text[:40]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    parts.append(text)

            summary = " | ".join(parts) if parts else cluster.chunks[0].content[:200]
            results.append((cluster, summary))

        return results

    # -- 整合写回 -----------------------------------------------------------

    async def apply(
        self,
        summaries: list[tuple[Cluster, str]],
        *,
        hierarchical: Any | None = None,
        core: Any | None = None,
        beam: Any | None = None,
    ) -> ConsolidationResult:
        """将整合结果写回记忆层。

        流程:
        1. 对每个聚类的摘要，写入新记忆（到 Tier1）
        2. 删除原始碎片（Tier1 + Tier2）
        """
        result = ConsolidationResult(
            clusters_found=len(summaries),
            memories_merged=sum(c.size for c, _ in summaries),
            memories_removed=0,
            summaries_generated=0,
        )

        total = len(summaries)
        for i, (cluster, summary) in enumerate(summaries):
            self._check_stop()
            self._report_progress(
                "applying",
                {
                    "apply_current": i + 1,
                    "apply_total": total,
                    "cluster_id": cluster.cluster_id,
                    "message": f"正在写回第 {i + 1}/{total} 个整合摘要",
                },
            )
            # 写入整合后的摘要（到 Tier1）
            if hierarchical:
                eid = await hierarchical.store(
                    summary,
                    importance=max(cluster.avg_importance, 0.5),
                    memory_type="consolidated",
                    scope="cross_session",
                    source="consolidator",
                    tags=["consolidated"],
                )
                result.summaries_generated += 1
                result.details.append(
                    f"cluster_{cluster.cluster_id}: "
                    f"{cluster.size}条→{eid[:8]}"
                )

            # 删除原始碎片（按 tier 分别处理）
            for chunk in cluster.chunks:
                if chunk.tier == "hierarchical" and hierarchical:
                    await self._remove_hierarchical_entry(
                        hierarchical, chunk.entry_id
                    )
                    result.memories_removed += 1
                elif chunk.tier == "beam" and beam:
                    if await self._remove_beam_entry(beam, chunk.entry_id):
                        result.memories_removed += 1

        # 更新 pending 列表: 移除已整合的
        merged_ids = {
            chunk.entry_id
            for cluster, _ in summaries
            for chunk in cluster.chunks
        }
        self._pending_chunks = [
            c for c in self._pending_chunks
            if c.entry_id not in merged_ids
        ]

        return result

    @staticmethod
    async def _remove_hierarchical_entry(hierarchical: Any, entry_id: str) -> None:
        """安全删除一条 hierarchical 记忆。"""
        try:
            async with hierarchical._lock:
                if hierarchical._conn:
                    hierarchical._conn.execute(
                        "DELETE FROM hierarchical_entries WHERE entry_id = ?",
                        (entry_id,),
                    )
                    hierarchical._conn.commit()
        except Exception as exc:
            logger.debug("Failed to remove entry %s: %s", entry_id, exc)

    @staticmethod
    async def _remove_beam_entry(beam: Any, entry_id: str) -> bool:
        """安全删除一条 beam (Tier 2) 记忆。

        entry_id 由 BeamMemory.list_entries 给出：fallback 后端是 "fb_<数字>"，
        mnemosyne 后端是原始字符串 id。删除统一交给 delete_entries 处理，
        由后端各自解析 id 形态。
        """
        try:
            if beam is None or not hasattr(beam, "delete_entries"):
                return False
            removed = await beam.delete_entries([entry_id])
            return removed > 0
        except Exception as exc:
            logger.debug("Failed to remove beam entry %s: %s", entry_id, exc)
            return False

    # -- 一站式整合 (收集→聚类→归纳→写回) -----------------------------------

    async def run_full_consolidation(
        self,
        *,
        hierarchical: Any | None = None,
        beam: Any | None = None,
        use_llm: bool = True,
    ) -> ConsolidationResult:
        """执行完整的记忆整合流程。

        多阶段流水线（按顺序执行，每阶段可独立完成压缩目标）:
        0. aggressive_cleanup  — 直接批量删除低价值碎片（无 LLM）
        1. collect             — 从各层收集待整合的记忆片段
        2. filter              — 噪声预过滤
        3. type_grouped        — 按 memory_type + 时间窗口分组整合
        4. cluster             — TF-IDF + Jaccard 相似度聚类
        5. summarize           — LLM 归纳 / 规则整合
        6. apply               — 删除原始碎片，写入整合后的新记忆

        这是空闲时调用的主入口。
        """
        if self._running:
            logger.debug("Consolidation already running, skipping")
            return ConsolidationResult(
                clusters_found=0, memories_merged=0,
                memories_removed=0, summaries_generated=0,
            )

        self._running = True
        self._reset_stop()
        start = time.monotonic()

        try:
            # 0. 激进清理阶段（直接删除低价值碎片，无需 LLM）
            self._check_stop()
            self._report_progress("aggressive_cleanup")
            removed_directly = await self.aggressive_cleanup(
                hierarchical=hierarchical,
                beam=beam,
            )
            logger.info("Aggressive cleanup removed %d low-value entries",
                        removed_directly)

            # 1. 收集
            self._check_stop()
            self._report_progress("collecting")
            all_chunks: list[MemoryChunk] = []
            if hierarchical:
                all_chunks.extend(await self.collect_from_hierarchical(hierarchical))
            if beam:
                all_chunks.extend(await self.collect_from_beam(beam))

            # 1.1 噪声预过滤
            self._check_stop()
            self._report_progress("filtering", {"before": len(all_chunks)})
            all_chunks = self._filter_noise(all_chunks)

            if len(all_chunks) < self._min_chunks:
                logger.debug("Not enough chunks for consolidation (%d)",
                             len(all_chunks))
                result = ConsolidationResult(
                    clusters_found=0, memories_merged=0,
                    memories_removed=removed_directly,
                    summaries_generated=0,
                    details=[f"insufficient chunks: {len(all_chunks)}",
                             f"direct_removed: {removed_directly}"],
                )
                result.duration_ms = (time.monotonic() - start) * 1000
                return result

            # 2. 类型分组整合（先于相似度聚类，吸收大量同质碎片）
            self._check_stop()
            self._report_progress("type_grouping", {"chunks": len(all_chunks)})
            all_chunks, type_summaries = await self.type_grouped_consolidation(
                all_chunks,
                hierarchical=hierarchical,
            )

            # P1-1: 重要单条白名单 — importance > 0.8 的不参与聚类
            self._check_stop()
            protected = [c for c in all_chunks if c.importance > _PROTECTED_IMPORTANCE_THRESHOLD]
            clusterable = [c for c in all_chunks if c.importance <= _PROTECTED_IMPORTANCE_THRESHOLD]
            if protected:
                logger.info("Protected %d high-importance chunks from consolidation",
                            len(protected))

            # 3. 聚类（仅对可聚类部分）
            self._check_stop()
            self._report_progress("clustering", {"chunks": len(clusterable)})
            clusters = await self.cluster(clusterable)

            # 合并 type_grouped 摘要与相似度聚类
            summaries: list[tuple[Cluster, str]] = list(type_summaries)
            if not clusters and not type_summaries:
                # 即使没有聚类，也写入 protected 作为 pinned 条目
                if hierarchical and protected:
                    for chunk in protected:
                        await hierarchical.store(
                            chunk.content,
                            importance=chunk.importance,
                            memory_type="pinned",
                            scope="cross_session",
                            source=chunk.source,
                            tags=["pinned", "high_importance"],
                        )
                result = ConsolidationResult(
                    clusters_found=0, memories_merged=0,
                    memories_removed=removed_directly,
                    summaries_generated=0,
                    details=[f"protected_only: {len(protected)} items pinned",
                             f"direct_removed: {removed_directly}"],
                )
                result.duration_ms = (time.monotonic() - start) * 1000
                return result

            # 4. 归纳（仅对相似度聚类部分）
            if clusters:
                self._check_stop()
                self._report_progress("summarizing", {"clusters": len(clusters)})
                if use_llm and (self._llm or self._local_llm):
                    llm_summaries = await self.consolidate_with_llm(clusters)
                else:
                    llm_summaries = await self.consolidate_with_rules(clusters)
                summaries.extend(llm_summaries)

            # 5. 写回
            self._check_stop()
            self._report_progress("applying", {"summaries": len(summaries)})
            result = await self.apply(
                summaries,
                hierarchical=hierarchical,
                beam=beam,
            )
            result.memories_removed += removed_directly
            result.details.append(f"direct_removed: {removed_directly}")
            result.details.append(
                f"type_grouped: {len(type_summaries)} groups merged"
            )

            # P1-1: 写回被保护的 high-importance 单条作为 pinned 条目
            if hierarchical and protected:
                for chunk in protected:
                    await hierarchical.store(
                        chunk.content,
                        importance=chunk.importance,
                        memory_type="pinned",
                        scope="cross_session",
                        source=chunk.source,
                        tags=["pinned", "high_importance"],
                    )
                result.details.append(f"pinned: {len(protected)} high-importance items")

            result.duration_ms = (time.monotonic() - start) * 1000

            logger.info(
                "Consolidation complete: %d clusters, %d merged, %d removed "
                "(%d direct cleanup), %d summaries in %.1fs",
                result.clusters_found, result.memories_merged,
                result.memories_removed, removed_directly,
                result.summaries_generated,
                result.duration_ms / 1000,
            )
            return result

        except asyncio.CancelledError as exc:
            logger.info("Consolidation cancelled: %s", exc)
            raise
        finally:
            self._running = False
            self._stop_event.clear()

    @property
    def is_running(self) -> bool:
        return self._running
