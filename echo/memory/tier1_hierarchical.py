from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import (
    DEFAULT_TIER1_BM25_K,
    DEFAULT_TIER1_RECENCY_HALFLIFE_HOURS,
    DEFAULT_TIER1_WEIGHT_IMPORTANCE,
    DEFAULT_TIER1_WEIGHT_RECENCY,
    DEFAULT_TIER1_WEIGHT_RELEVANCE,
)
from .zh_segment import index_text, segment

logger = logging.getLogger(__name__)


_L1_MAX_CHARS = 64   # 单行摘要 (~17 tokens)
_L2_MAX_CHARS = 200  # 段落详情 (~50 tokens)
_L3_MAX_CHARS = 2000 # 完整上下文 (~500 tokens)

# 人名/称呼识别：用于“核心检索”，避免实体命中被近因与重要度挤出。
# 只匹配像具体称呼的片段（阿梅 / 邵师傅 / 罗姐），再由库内文档频率过滤掉泛词。
_ENTITY_PATTERNS = (
    re.compile(r"[阿小老][\u4e00-\u9fa5]{1,2}"),
    re.compile(r"[\u4e00-\u9fa5]{1,3}(?:师傅|老师|医生|大夫|经理|老板|姐|哥|叔|姨)"),
)
_ENTITY_MAX_DF = 8      # 库内出现次数超过该值视为泛词，不做核心检索
_ENTITY_MAX_TERMS = 3
_ENTITY_RESERVED = 3    # 最多为实体命中的条目前置保留几个名额
_ENTITY_STOP = {
    "小票", "小时", "小区", "小米", "小学", "小心", "小声", "小组", "小吃", "小店",
    "老师", "老家", "老实", "老是", "老板", "老公", "老婆", "大夫",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HierarchicalEntry:
    """hmem 风格分层记忆条目。"""
    entry_id: str
    l1_summary: str          # Level 1: 单行摘要 (始终加载)
    l2_detail: str | None    # Level 2: 段落详情
    l3_full: str | None      # Level 3: 完整内容
    l4_extended: str | None  # Level 4: 扩展内容
    l5_raw: str | None       # Level 5: 原始数据
    importance: float        # 0.0 ~ 1.0
    memory_type: str         # "preference", "decision", "project", "role_knowledge", "conversation"
    scope: str               # "session", "cross_session", "global"
    source: str              # "roleplay", "agent", "user"
    timestamp: str           # ISO 8601
    tags: list[str] = field(default_factory=list)

    def to_l1_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry_id,
            "l1": self.l1_summary,
            "importance": self.importance,
            "type": self.memory_type,
            "scope": self.scope,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Tier 1: HierarchicalMemory — hmem 风格 5 级懒加载
# ---------------------------------------------------------------------------

class HierarchicalMemory:
    """Tier 1 — 跨会话语义摘要（hmem 风格）

    用途: Roleplay 层系统提示注入 (L1) + Agent 层按需钻取 (L2-L5)

    特征:
    - 5 级懒加载: 始终只加载 L1 摘要 (< 5K tokens for 300 entries)
    - L2-L5 存储在 SQLite，按需加载
    - 通过重要性 + 类型 + 时间混合排序决定哪些 L1 出现在系统提示中
    - 灵感来自 hmem (its-over-9k) 的分层设计
    """

    def __init__(
        self,
        db_path: Path,
        *,
        weight_relevance: float = DEFAULT_TIER1_WEIGHT_RELEVANCE,
        weight_importance: float = DEFAULT_TIER1_WEIGHT_IMPORTANCE,
        weight_recency: float = DEFAULT_TIER1_WEIGHT_RECENCY,
        bm25_k: float = DEFAULT_TIER1_BM25_K,
        recency_halflife_hours: float = DEFAULT_TIER1_RECENCY_HALFLIFE_HOURS,
    ) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._l1_cache: dict[str, HierarchicalEntry] = OrderedDict()
        self._cache_max = 500
        self._weight_relevance = max(0.0, weight_relevance)
        self._weight_importance = max(0.0, weight_importance)
        self._weight_recency = max(0.0, weight_recency)
        self._bm25_k = max(0.1, bm25_k)
        self._recency_halflife_hours = max(1.0, recency_halflife_hours)

    # -- 初始化 ------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化 SQLite 存储 + FTS5 全文索引。"""
        async with self._lock:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS hierarchical_entries (
                entry_id TEXT PRIMARY KEY,
                l1_summary TEXT NOT NULL,
                l2_detail TEXT,
                l3_full TEXT,
                l4_extended TEXT,
                l5_raw TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                memory_type TEXT NOT NULL DEFAULT 'conversation',
                scope TEXT NOT NULL DEFAULT 'session',
                source TEXT NOT NULL DEFAULT 'user',
                timestamp TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            self._conn.execute("""CREATE INDEX IF NOT EXISTS idx_hier_scope
                ON hierarchical_entries(scope, importance DESC)""")
            self._conn.execute("""CREATE INDEX IF NOT EXISTS idx_hier_type
                ON hierarchical_entries(memory_type, importance DESC)""")
            # FTS5 for legacy/ASCII search（保留旧表，不再每次启动全量重建）
            try:
                self._conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS hierarchical_fts
                    USING fts5(entry_id UNINDEXED, l1_summary, l2_detail, l3_full, l4_extended, l5_raw)""")
            except sqlite3.OperationalError:
                pass

            # 预分词检索索引：中文由外部分词器切成空格分隔的 token，
            # unicode61 才能按词/ngram 建索引，避免整段中文被当成一个 token。
            self._conn.execute("""CREATE TABLE IF NOT EXISTS hierarchical_seg (
                entry_id TEXT PRIMARY KEY,
                tokens TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            try:
                self._conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS hierarchical_seg_fts
                    USING fts5(entry_id UNINDEXED, tokens, tokenize='unicode61')""")
            except sqlite3.OperationalError:
                pass

            self._conn.commit()
            self._backfill_seg_index()
            logger.info("HierarchicalMemory (hmem-style) initialized @ %s", self._db_path)

    def _backfill_seg_index(self) -> None:
        """为缺少预分词索引的旧条目补建索引。"""
        if self._conn is None:
            return
        rows = self._conn.execute(
            """SELECT e.entry_id, e.l1_summary, e.l2_detail, e.l3_full
               FROM hierarchical_entries e
               LEFT JOIN hierarchical_seg s ON s.entry_id = e.entry_id
               WHERE s.entry_id IS NULL
               LIMIT 400"""
        ).fetchall()
        if not rows:
            return
        for entry_id, l1, l2, l3 in rows:
            self._write_seg(entry_id, l1 or "", l2 or "", l3 or "")
        self._conn.commit()
        logger.info("HierarchicalMemory: backfilled %d seg indexes", len(rows))

    def _write_seg(self, entry_id: str, l1: str, l2: str, l3: str) -> None:
        """写入/更新一条预分词索引（调用方负责 commit）。"""
        if self._conn is None:
            return
        # L1+L2 已覆盖摘要与段落；L3 与 L2 高度重复，不重复索引以控制写入开销。
        tokens = index_text(" ".join(part for part in (l1, l2) if part))
        self._conn.execute(
            "INSERT OR REPLACE INTO hierarchical_seg(entry_id, tokens, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (entry_id, tokens),
        )
        try:
            self._conn.execute("DELETE FROM hierarchical_seg_fts WHERE entry_id = ?", (entry_id,))
            self._conn.execute(
                "INSERT INTO hierarchical_seg_fts(entry_id, tokens) VALUES (?, ?)",
                (entry_id, tokens),
            )
        except sqlite3.OperationalError:
            pass

    def _delete_seg(self, entry_id: str) -> None:
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM hierarchical_seg WHERE entry_id = ?", (entry_id,))
        try:
            self._conn.execute("DELETE FROM hierarchical_seg_fts WHERE entry_id = ?", (entry_id,))
        except sqlite3.OperationalError:
            pass

    # -- 写入 --------------------------------------------------------------

    async def store(self, content: str, *,
                    importance: float = 0.5,
                    memory_type: str = "conversation",
                    scope: str = "session",
                    source: str = "user",
                    tags: list[str] | None = None,
                    entry_id: str | None = None,
                    event_time: str | datetime | None = None) -> str:
        """存储一条分层记忆，自动生成 L1-L5 摘要。

        event_time 是调用方传入的逻辑对话时间；不传则用当前时间。
        created_at 仍由数据库记录真实写入时间，便于审计。
        """
        eid = entry_id or _generate_id()
        ts = _iso_time(event_time)

        levels = _summarize_levels(content)

        async with self._lock:
            if self._conn is None:
                raise RuntimeError("HierarchicalMemory not initialized")

            self._conn.execute("""INSERT OR REPLACE INTO hierarchical_entries
                (entry_id, l1_summary, l2_detail, l3_full, l4_extended, l5_raw,
                 importance, memory_type, scope, source, timestamp, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, levels["l1"], levels["l2"], levels["l3"],
                 levels["l4"], levels["l5"],
                 max(0.0, min(1.0, importance)),
                 memory_type, scope, source, ts,
                 json.dumps(tags or [], ensure_ascii=False)))
            try:
                self._conn.execute("DELETE FROM hierarchical_fts WHERE entry_id = ?", (eid,))
                self._conn.execute("""INSERT INTO hierarchical_fts
                    (entry_id, l1_summary, l2_detail, l3_full, l4_extended, l5_raw)
                    VALUES (?,?,?,?,?,?)""",
                    (eid, levels["l1"], levels["l2"], levels["l3"], levels["l4"], levels["l5"]))
            except sqlite3.OperationalError:
                pass
            self._write_seg(eid, levels["l1"], levels["l2"], levels["l3"])
            self._conn.commit()
        entry = HierarchicalEntry(
            entry_id=eid, l1_summary=levels["l1"], l2_detail=levels["l2"],
            l3_full=levels["l3"], l4_extended=levels["l4"], l5_raw=levels["l5"],
            importance=importance, memory_type=memory_type,
            scope=scope, source=source, timestamp=ts, tags=tags or [],
        )
        self._update_l1_cache(eid, entry)

        logger.debug("Hierarchical stored [%s] %s: %s", memory_type, eid[:8], levels["l1"])
        return eid

    # -- L1 摘要加载（始终注入 Roleplay 系统提示）---------------------------

    async def get_l1_summaries(self, max_items: int = 50,
                               scope_filter: str | None = None) -> list[dict[str, Any]]:
        """获取 L1 摘要列表（注入 Roleplay 系统提示）。

        排序: scope=global > cross_session > session,
              按 importance 降序,
              按 timestamp 降序。
        """
        async with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                """SELECT entry_id, l1_summary, importance, memory_type, scope, timestamp
                   FROM hierarchical_entries
                   WHERE (?1 IS NULL OR scope = ?1)
                   ORDER BY
                     CASE scope WHEN 'global' THEN 0 WHEN 'cross_session' THEN 1 ELSE 2 END,
                     importance DESC,
                     timestamp DESC
                   LIMIT ?2""",
                (scope_filter, max_items),
            ).fetchall()
            return [
                {"id": r[0], "l1": r[1], "importance": r[2],
                 "type": r[3], "scope": r[4], "timestamp": r[5]}
                for r in rows
            ]

    async def list_entries(self, limit: int = 20, offset: int = 0,
                           scope: str | None = None,
                           memory_type: str | None = None) -> list[dict[str, Any]]:
        """按时间倒序分页枚举条目（CRUD 浏览用）。"""
        async with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                """SELECT entry_id, l1_summary, importance, memory_type, scope, source, timestamp
                   FROM hierarchical_entries
                   WHERE (?1 IS NULL OR scope = ?1)
                     AND (?2 IS NULL OR memory_type = ?2)
                   ORDER BY timestamp DESC
                   LIMIT ?3 OFFSET ?4""",
                (scope, memory_type, limit, offset),
            ).fetchall()
            return [
                {"id": r[0], "l1": r[1], "importance": r[2], "type": r[3],
                 "scope": r[4], "source": r[5], "timestamp": r[6]}
                for r in rows
            ]

    # -- L2-L5 按需钻取（Agent 层通过 memory_search 调用）-----------------

    async def drill_down(self, entry_id: str, level: int = 2) -> str | None:
        """按钻取层级获取详情。
        level=2 → 段落, level=3 → 全文, level=4 → 扩展, level=5 → 原始
        """
        col = {2: "l2_detail", 3: "l3_full", 4: "l4_extended", 5: "l5_raw"}
        col_name = col.get(level, "l2_detail")
        async with self._lock:
            if self._conn is None:
                return None
            row = self._conn.execute(
                f"SELECT {col_name} FROM hierarchical_entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            return row[0] if row else None

    async def get(self, entry_id: str) -> dict[str, Any] | None:
        async with self._lock:
            if self._conn is None:
                return None
            row = self._conn.execute(
                """SELECT entry_id, l1_summary, l2_detail, l3_full, l4_extended,
                          l5_raw, importance, memory_type, scope, source, timestamp, tags
                   FROM hierarchical_entries WHERE entry_id = ?""",
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "l1": row[1], "l2": row[2], "l3": row[3],
            "l4": row[4], "l5": row[5], "importance": row[6],
            "type": row[7], "scope": row[8], "source": row[9],
            "timestamp": row[10], "tags": json.loads(row[11] or "[]"),
        }

    async def update(self, entry_id: str, content: str, **metadata: Any) -> bool:
        current = await self.get(entry_id)
        if current is None:
            return False
        levels = _summarize_levels(content)
        async with self._lock:
            if self._conn is None:
                return False
            self._conn.execute(
                """UPDATE hierarchical_entries SET l1_summary=?, l2_detail=?, l3_full=?,
                   l4_extended=?, l5_raw=?, importance=?, memory_type=?, scope=?, source=?, tags=?
                   WHERE entry_id=?""",
                (levels["l1"], levels["l2"], levels["l3"], levels["l4"], levels["l5"],
                 metadata.get("importance", current["importance"]),
                 metadata.get("memory_type", current["type"]), metadata.get("scope", current["scope"]),
                 metadata.get("source", current["source"]),
                 json.dumps(metadata.get("tags", current["tags"]), ensure_ascii=False), entry_id),
            )
            try:
                self._conn.execute("DELETE FROM hierarchical_fts WHERE entry_id = ?", (entry_id,))
                self._conn.execute("""INSERT INTO hierarchical_fts
                    (entry_id, l1_summary, l2_detail, l3_full, l4_extended, l5_raw)
                    VALUES (?,?,?,?,?,?)""",
                    (entry_id, levels["l1"], levels["l2"], levels["l3"], levels["l4"], levels["l5"]))
            except sqlite3.OperationalError:
                pass
            self._write_seg(entry_id, levels["l1"], levels["l2"], levels["l3"])
            self._conn.commit()
        return True

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            if self._conn is None:
                return False
            cur = self._conn.execute("DELETE FROM hierarchical_entries WHERE entry_id = ?", (entry_id,))
            try:
                self._conn.execute("DELETE FROM hierarchical_fts WHERE entry_id = ?", (entry_id,))
            except sqlite3.OperationalError:
                pass
            self._delete_seg(entry_id)
            self._conn.commit()
            self._l1_cache.pop(entry_id, None)
            return cur.rowcount > 0

    async def search(self, query: str, max_results: int = 5,
                     min_score: float = 0.1,
                     query_time: str | datetime | None = None) -> list[dict[str, Any]]:
        """分层记忆检索：预分词 FTS5/BM25 + 词项覆盖 + importance + recency。

        query_time 是题目/对话的逻辑“现在”；不传则用当前时间。时间衰减
        使用 event_time（写入时落在 timestamp 列），而不是真实回放时间。
        """
        if not query.strip() or self._conn is None:
            return []
        terms = segment(query, max_terms=16)
        if not terms:
            return []
        ref = _parse_time(query_time)
        async with self._lock:
            entity_terms = self._entity_terms(query)
            entity_rows = self._search_entities(entity_terms)
            entity_ids = {row["id"] for row in entity_rows}

            rows = self._search_seg(terms, max_results)
            if not rows:
                rows = self._search_legacy(terms, max_results)
            if entity_rows:
                already = {row["id"] for row in entity_rows}
                rows = entity_rows + [row for row in rows if row["id"] not in already]

            results: list[dict[str, Any]] = []
            for row in rows:
                content = row["l1"] or ""
                detail = row["l2"] or ""
                text = " ".join(part for part in (content, detail) if part).lower()
                coverage = sum(1 for term in terms if term in text) / len(terms)
                bm25_pos = row.get("bm25_pos") or 0.0
                bm25_rel = bm25_pos / (bm25_pos + self._bm25_k) if bm25_pos > 0 else 0.0
                relevance = max(bm25_rel, coverage)
                is_entity = row["id"] in entity_ids
                if is_entity:
                    # 实体（人名/称呼）命中视为强相关，避免被近因和重要度稀释
                    relevance = max(relevance, 0.98)
                recency = _recency(row["timestamp"], ref, self._recency_halflife_hours)
                score = (
                    self._weight_relevance * relevance
                    + self._weight_importance * max(0.0, min(1.0, row["importance"]))
                    + self._weight_recency * recency
                )
                if score < min_score:
                    continue
                results.append({
                    "id": row["id"],
                    "content": content,
                    "detail": detail,
                    "importance": row["importance"],
                    "type": row["type"],
                    "scope": row["scope"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "tier": "hierarchical",
                    "score": round(score, 4),
                    "raw_bm25": round(bm25_pos, 4),
                    "relevance": round(relevance, 4),
                    "recency": round(recency, 4),
                    "entity_hit": is_entity,
                })
            results.sort(key=lambda item: item["score"], reverse=True)
            if entity_ids:
                head = [item for item in results if item["entity_hit"]][:_ENTITY_RESERVED]
                rest = [item for item in results if not item["entity_hit"]]
                results = head + rest
            selected = results[:max_results]
            for item in selected:
                raw = self._conn.execute(
                    "SELECT l5_raw FROM hierarchical_entries WHERE entry_id = ?",
                    (item["id"],),
                ).fetchone()
                item["summary"] = item["content"]
                evidence = raw[0] if raw and raw[0] else item["detail"] or item["content"]
                item["content"] = evidence
                item["evidence_id"] = item["id"]
            return selected

    def _entity_terms(self, query: str) -> list[str]:
        """从查询中提取像具体称呼的实体，并用库内文档频率过滤泛词。"""
        if self._conn is None or not query.strip():
            return []
        candidates: list[str] = []

        def add(term: str) -> None:
            term = term.strip()
            if len(term) < 2 or term in candidates or term in _ENTITY_STOP:
                return
            candidates.append(term)

        for match in re.finditer(r"[阿小老]", query):
            for extra in (1, 2):
                add(query[match.start(): match.start() + 1 + extra])
        for match in re.finditer(r"(?:师傅|老师|医生|大夫|经理|老板|姐|哥|叔|姨)", query):
            for back in (1, 2, 3):
                start = match.start() - back
                if start >= 0:
                    add(query[start: match.end()])

        kept: list[str] = []
        for term in candidates:
            like = f"%{term}%"
            df = self._conn.execute(
                "SELECT COUNT(*) FROM hierarchical_entries "
                "WHERE l1_summary LIKE ? OR l2_detail LIKE ?",
                (like, like),
            ).fetchone()[0]
            if 0 < df <= _ENTITY_MAX_DF:
                kept.append(term)
            if len(kept) >= _ENTITY_MAX_TERMS:
                break
        return kept

    def _search_entities(self, terms: list[str]) -> list[dict[str, Any]]:
        """按实体名做精确子串检索，返回与普通候选同结构的数据。"""
        if self._conn is None or not terms:
            return []
        found: dict[str, dict[str, Any]] = {}
        for term in terms:
            like = f"%{term}%"
            for row in self._conn.execute(
                """SELECT entry_id, l1_summary, l2_detail, importance,
                          memory_type, scope, timestamp, source
                   FROM hierarchical_entries
                   WHERE l1_summary LIKE ? OR l2_detail LIKE ?""",
                (like, like),
            ):
                found.setdefault(row[0], {
                    "id": row[0], "l1": row[1], "l2": row[2], "importance": row[3],
                    "type": row[4], "scope": row[5], "timestamp": row[6],
                    "source": row[7], "bm25_pos": 0.0,
                })
        return list(found.values())

    def _search_seg(self, terms: list[str], max_results: int) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        match = " OR ".join('"' + term.replace('"', "") + '"' for term in terms)
        try:
            rows = self._conn.execute(
                """SELECT e.entry_id, e.l1_summary, e.l2_detail, e.importance,
                          e.memory_type, e.scope, e.timestamp, e.source,
                          -bm25(hierarchical_seg_fts) AS bm25_pos
                   FROM hierarchical_seg_fts f
                   JOIN hierarchical_entries e ON e.entry_id = f.entry_id
                   WHERE hierarchical_seg_fts MATCH ?
                   ORDER BY bm25_pos DESC
                   LIMIT ?""",
                (match, max(max_results * 4, 20)),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            return []
        return [{
            "id": r[0], "l1": r[1], "l2": r[2], "importance": r[3],
            "type": r[4], "scope": r[5], "timestamp": r[6], "source": r[7],
            "bm25_pos": r[8] or 0.0,
        } for r in rows]

    def _search_legacy(self, terms: list[str], max_results: int) -> list[dict[str, Any]]:
        """旧 FTS / LIKE 兜底；LIKE 只用最长 3 个词，不再整句匹配。"""
        if self._conn is None:
            return []
        rows: list[Any] = []
        try:
            rows = self._conn.execute(
                """SELECT e.entry_id, e.l1_summary, e.l2_detail, e.importance,
                          e.memory_type, e.scope, e.timestamp, e.source
                   FROM hierarchical_fts f
                   JOIN hierarchical_entries e ON e.entry_id = f.entry_id
                   WHERE hierarchical_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (_fts_query(" ".join(terms)), max_results),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            rows = []

        if not rows:
            like_terms = sorted(terms, key=len, reverse=True)[:3]
            conditions = " OR ".join(
                ["l1_summary LIKE ? OR l2_detail LIKE ? OR l3_full LIKE ?"] * len(like_terms)
            )
            params: list[str] = []
            for term in like_terms:
                params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
            try:
                rows = self._conn.execute(
                    f"""SELECT entry_id, l1_summary, l2_detail, importance,
                               memory_type, scope, timestamp, source
                        FROM hierarchical_entries
                        WHERE {conditions}
                        ORDER BY importance DESC, timestamp DESC
                        LIMIT ?""",
                    (*params, max_results * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

        return [{
            "id": r[0], "l1": r[1], "l2": r[2], "importance": r[3],
            "type": r[4], "scope": r[5], "timestamp": r[6], "source": r[7],
            "bm25_pos": 0.0,
        } for r in rows]

    # -- 内部 --------------------------------------------------------------

    def _update_l1_cache(self, eid: str, entry: HierarchicalEntry) -> None:
        self._l1_cache[eid] = entry
        if len(self._l1_cache) > self._cache_max:
            self._l1_cache.popitem(last=False)

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None




# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _summarize_levels(content: str) -> dict[str, str]:
    """将一段内容自动生成为 5 个层级。"""
    clean = content.strip()
    lines = clean.splitlines()
    first_line = lines[0] if lines else clean

    l1 = first_line.strip() if first_line.strip() else clean
    if len(l1) > _L1_MAX_CHARS:
        l1 = l1[:_L1_MAX_CHARS].rstrip() + "…"

    # L2: 第一段 (< _L2_MAX_CHARS)
    paragraph = ""
    for line in lines[:3]:
        paragraph += line.strip() + " "
    l2 = paragraph.strip() if paragraph.strip() else clean
    if len(l2) > _L2_MAX_CHARS:
        l2 = l2[:_L2_MAX_CHARS].rstrip() + "…"

    # L3: 全部内容 (< _L3_MAX_CHARS)
    l3 = clean if len(clean) <= _L3_MAX_CHARS else clean[:_L3_MAX_CHARS].rstrip() + "…\n[truncated]"

    # L4/L5: 完整内容
    l4 = clean
    l5 = clean

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "l5": l5}


def _iso_time(value: str | datetime | None) -> str:
    """把 event_time 规范成 ISO 字符串；None 使用当前时间。"""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.isoformat()
    return str(value)


def _parse_time(value: str | datetime | None) -> datetime:
    """解析逻辑时间；失败时退回当前时间。"""
    if value is None:
        return datetime.now(timezone.utc)
    try:
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _recency(timestamp: str, reference: datetime, halflife_hours: float) -> float:
    """指数时间衰减；无效时间戳返回 0.5（中性）。"""
    if not timestamp:
        return 0.5
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (reference - moment.astimezone(timezone.utc)).total_seconds() / 3600.0)
        return 0.5 ** (age_hours / max(1.0, halflife_hours))
    except (TypeError, ValueError):
        return 0.5


def _fts_query(query: str) -> str:
    """将查询转换为 FTS5 兼容格式。"""
    terms = re.sub(r"[^\w\s]", " ", query).split()
    return " OR ".join(terms) if terms else query


def _generate_id() -> str:
    import uuid
    return f"hm_{uuid.uuid4().hex[:16]}"
