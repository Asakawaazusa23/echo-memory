from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import (
    DEFAULT_TIER3_WEIGHT_COVERAGE,
    DEFAULT_TIER3_WEIGHT_EXACT,
)
from .zh_segment import normalize, segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 3: CoreMemory — 固化记忆层
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CoreFact:
    """一个不可变的固化事实。"""
    fact_id: str
    content: str
    category: str        # "user_preference", "project_rule", "identity", "knowledge"
    source: str          # "manual" (系统/用户手动添加), "inferred" (自动提取)
    created_at: str
    weight: float = 1.0 # 重要性权重，决定注入优先级
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at,
            "weight": self.weight,
        }


class CoreMemory:
    """Tier 3 — 固化记忆层

    用途:
    - 永远随系统提示注入的用户偏好 / 项目规则
    - 类似 Hermes Agent 的 MEMORY.md 概念
    - 类似 Mnemosyne import→1.0 的真实性记忆
    - 不会被压缩、覆盖或遗忘

    存储: SQLite (轻量, 数百条规模)
    注入: 始终随 Roleplay 系统提示注入最高权重的核心事实
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._facts: list[CoreFact] = []  # 内存缓存
        self._fact_terms: dict[str, set[str]] = {}
        self._fact_norm: dict[str, str] = {}
        self._weight_coverage = DEFAULT_TIER3_WEIGHT_COVERAGE
        self._weight_exact = DEFAULT_TIER3_WEIGHT_EXACT

    async def initialize(self) -> None:
        async with self._lock:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS core_facts (
                fact_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'knowledge',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )""")
            self._conn.commit()
            await self._reload_cache()
            logger.info("CoreMemory initialized @ %s", self._db_path)

    # -- 读取 ---------------------------------------------------------------

    async def get_active_facts(self, max_items: int = 30) -> list[dict[str, Any]]:
        """获取活跃的核心事实（按 weight 降序）。"""
        async with self._lock:
            sorted_facts = sorted(self._facts, key=lambda f: f.weight, reverse=True)
            return [f.to_dict() for f in sorted_facts[:max_items]]

    async def list_facts(self, limit: int = 20, offset: int = 0,
                         category: str | None = None) -> list[dict[str, Any]]:
        """分页枚举核心事实（weight 降序，再按创建时间倒序；CRUD 浏览用）。"""
        async with self._lock:
            facts = [f for f in self._facts
                     if category is None or f.category == category]
            facts.sort(key=lambda f: (f.weight, f.created_at), reverse=True)
            return [f.to_dict() for f in facts[offset:offset + limit]]

    async def search(self, query: str, max_results: int = 5,
                     min_score: float = 0.05) -> list[dict[str, Any]]:
        """词法检索核心事实：词项覆盖 + 原文精确匹配 + weight。"""
        if not query.strip():
            return []
        terms = set(segment(query, max_terms=20))
        if not terms:
            return []
        query_norm = normalize(query)
        scored = []
        for fact in self._facts:
            fact_terms = self._fact_terms.get(fact.fact_id, set())
            coverage = len(terms & fact_terms) / max(len(terms), 1)
            exact = 1.0 if query_norm and query_norm in self._fact_norm.get(fact.fact_id, "") else 0.0
            score = fact.weight * (self._weight_coverage * coverage + self._weight_exact * exact)
            if score >= min_score:
                scored.append((score, fact, coverage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**fact.to_dict(), "score": round(score, 4),
             "coverage": round(coverage, 4), "tier": "core"}
            for score, fact, coverage in scored[:max_results]
        ]

    # -- 写入 ---------------------------------------------------------------

    async def add_fact(self, content: str, *,
                       category: str = "knowledge",
                       source: str = "manual",
                       weight: float = 1.0) -> str:
        """添加一条固化事实。"""
        fid = f"cf_{_hash(content)}"
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("CoreMemory not initialized")
            self._conn.execute(
                "INSERT OR IGNORE INTO core_facts (fact_id, content, category, source, created_at, weight) VALUES (?,?,?,?,?,?)",
                (fid, content.strip(), category, source, now, max(0.0, min(1.0, weight))),
            )
            self._conn.commit()
        await self._reload_cache()
        logger.info("Core fact added [%s]: %s", category, content[:60])
        return fid

    async def set_auto_fact(self, slot: str, content: str, *,
                            category: str = "user_preference",
                            weight: float = 0.85,
                            conflicts: tuple[str, ...] = ()) -> str:
        """按稳定字段更新自动事实，保留手动录入的事实。"""
        import json
        fid = f"cf_auto_{_hash(slot)}"
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("CoreMemory not initialized")
            with self._conn:
                for conflict in conflicts:
                    self._conn.execute(
                        "DELETE FROM core_facts WHERE content=? AND source IN ('conversation', 'conversation_auto')",
                        (conflict,),
                    )
                if slot == "nickname":
                    self._conn.execute(
                        "DELETE FROM core_facts WHERE source='conversation' AND content LIKE '用户偏好称呼: %'"
                    )
                self._conn.execute(
                    "INSERT INTO core_facts (fact_id,content,category,source,created_at,weight,metadata) "
                    "VALUES (?,?,?,?,?,?,?) ON CONFLICT(fact_id) DO UPDATE SET "
                    "content=excluded.content, created_at=excluded.created_at, weight=excluded.weight",
                    (fid, content, category, "conversation_auto", now,
                     max(0.0, min(1.0, weight)), json.dumps({"slot": slot})),
                )
            await self._reload_cache()
        return fid

    async def remove_fact(self, fact_id: str) -> bool:
        async with self._lock:
            if self._conn is None:
                return False
            cur = self._conn.execute("DELETE FROM core_facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
        await self._reload_cache()
        return cur.rowcount > 0

    async def get_fact(self, fact_id: str) -> dict[str, Any] | None:
        async with self._lock:
            for fact in self._facts:
                if fact.fact_id == fact_id:
                    return fact.to_dict()
        return None

    async def update_fact(self, fact_id: str, content: str, **metadata: Any) -> bool:
        async with self._lock:
            if self._conn is None:
                return False
            current = self._conn.execute(
                "SELECT category, source, weight FROM core_facts WHERE fact_id=?", (fact_id,)
            ).fetchone()
            if current is None:
                return False
            cur = self._conn.execute(
                "UPDATE core_facts SET content=?, category=?, source=?, weight=? WHERE fact_id=?",
                (content, metadata.get("category", current[0]), metadata.get("source", current[1]),
                 max(0.0, min(1.0, float(metadata.get("weight", current[2])))), fact_id),
            )
            self._conn.commit()
        await self._reload_cache()
        return cur.rowcount > 0

    # -- 内部 ---------------------------------------------------------------

    async def _reload_cache(self) -> None:
        if self._conn is None:
            return
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(
            None,
            lambda: self._conn.execute(
                "SELECT fact_id, content, category, source, created_at, weight FROM core_facts ORDER BY weight DESC"
            ).fetchall(),
        )
        self._facts = [
            CoreFact(fact_id=r[0], content=r[1], category=r[2],
                     source=r[3], created_at=r[4], weight=r[5])
            for r in rows
        ]
        self._fact_terms = {
            fact.fact_id: set(segment(fact.content, max_terms=24))
            for fact in self._facts
        }
        self._fact_norm = {
            fact.fact_id: normalize(fact.content)
            for fact in self._facts
        }

    async def close(self) -> None:
        async with self._lock:
            self._facts.clear()
            self._fact_terms.clear()
            self._fact_norm.clear()
            if self._conn:
                self._conn.close()
                self._conn = None


def _hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()[:12]
