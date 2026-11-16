"""TokenTracker — LLM token 消耗追踪与统计。

记录每次 LLM 调用的 token 使用情况，支持：
- 实时累加统计
- 按操作类型（consolidation / evolution / agent）分组
- 持久化到 SQLite，便于后续查询和可视化
- 提供周期性摘要报告
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TokenUsageRecord:
    """单次 LLM 调用的 token 使用记录。"""
    operation: str         # "consolidation", "evolution", "agent", "roleplay" 等
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    duration_ms: float = 0.0
    detail: str = ""       # 可选描述

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


@dataclass(slots=True)
class TokenSummary:
    """某一维度的 token 汇总。"""
    operation: str
    call_count: int
    prompt_tokens_total: int
    completion_tokens_total: int
    total_tokens: int
    avg_tokens_per_call: float
    first_call: str
    last_call: str


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """LLM token 消耗追踪器。

    职责：
    - 记录每次 LLM API 调用的 token 使用量
    - 按操作类型分组统计
    - 持久化到 SQLite（token_usage.db）
    - 提供实时统计和历史查询接口
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

        # 内存中实时累计（避免频繁读 DB）
        self._session_stats: dict[str, _OperationStats] = {}
        self._session_start = datetime.now(timezone.utc).isoformat()

    # -- 生命周期 -----------------------------------------------------------

    async def initialize(self) -> None:
        """初始化 SQLite 存储。"""
        async with self._lock:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL,
                duration_ms REAL NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT ''
            )""")
            self._conn.execute("""CREATE INDEX IF NOT EXISTS idx_token_op
                ON token_usage(operation, timestamp)""")
            self._conn.commit()
        logger.info("TokenTracker initialized @ %s", self._db_path)

    # -- 记录 ----------------------------------------------------------------

    async def record(
        self,
        operation: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        duration_ms: float = 0.0,
        detail: str = "",
    ) -> None:
        """记录一次 LLM 调用的 token 消耗。"""
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        now = datetime.now(timezone.utc).isoformat()

        # 内存累计
        stats = self._session_stats.setdefault(operation, _OperationStats())
        stats.call_count += 1
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.total_tokens += total_tokens

        # 持久化到 SQLite
        async with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                """INSERT INTO token_usage
                   (operation, prompt_tokens, completion_tokens, total_tokens,
                    model, timestamp, duration_ms, detail)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (operation, prompt_tokens, completion_tokens, total_tokens,
                 model, now, duration_ms, detail),
            )
            self._conn.commit()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Token recorded [%s]: prompt=%d completion=%d total=%d",
                operation, prompt_tokens, completion_tokens, total_tokens,
            )

    async def record_from_response(
        self,
        operation: str,
        usage: Any,  # LLMUsage
        model: str = "",
        duration_ms: float = 0.0,
        detail: str = "",
    ) -> None:
        """从 LLMUsage 对象记录 token 消耗。"""
        await self.record(
            operation,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            model=model,
            duration_ms=duration_ms,
            detail=detail,
        )

    # -- 查询 ----------------------------------------------------------------

    async def get_session_summary(self) -> dict[str, Any]:
        """获取当前会话的 token 使用摘要。"""
        total_prompt = sum(s.prompt_tokens for s in self._session_stats.values())
        total_completion = sum(s.completion_tokens for s in self._session_stats.values())
        total = sum(s.total_tokens for s in self._session_stats.values())
        total_calls = sum(s.call_count for s in self._session_stats.values())

        by_operation = {}
        for op, stats in self._session_stats.items():
            by_operation[op] = {
                "call_count": stats.call_count,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.total_tokens,
                "avg_per_call": round(stats.total_tokens / max(stats.call_count, 1)),
            }

        return {
            "session_start": self._session_start,
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total,
            "by_operation": by_operation,
        }

    async def get_historical_summary(
        self,
        operation: str | None = None,
        since_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """查询历史 token 使用汇总。"""
        async with self._lock:
            if self._conn is None:
                return []

            since_dt = datetime.now(timezone.utc)
            from datetime import timedelta
            since_dt = since_dt - timedelta(hours=since_hours)
            since_str = since_dt.isoformat()

            if operation:
                rows = self._conn.execute(
                    """SELECT operation,
                              SUM(prompt_tokens) as p_sum,
                              SUM(completion_tokens) as c_sum,
                              SUM(total_tokens) as t_sum,
                              COUNT(*) as cnt,
                              MIN(timestamp) as first_ts,
                              MAX(timestamp) as last_ts
                       FROM token_usage
                       WHERE operation = ? AND timestamp >= ?
                       GROUP BY operation""",
                    (operation, since_str),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT operation,
                              SUM(prompt_tokens) as p_sum,
                              SUM(completion_tokens) as c_sum,
                              SUM(total_tokens) as t_sum,
                              COUNT(*) as cnt,
                              MIN(timestamp) as first_ts,
                              MAX(timestamp) as last_ts
                       FROM token_usage
                       WHERE timestamp >= ?
                       GROUP BY operation""",
                    (since_str,),
                ).fetchall()

            summaries = []
            for r in rows:
                summaries.append({
                    "operation": r[0],
                    "call_count": r[4],
                    "prompt_tokens": r[1],
                    "completion_tokens": r[2],
                    "total_tokens": r[3],
                    "avg_per_call": round(r[3] / max(r[4], 1)),
                    "first_call": r[5],
                    "last_call": r[6],
                })
            return summaries

    async def get_total_all_time(self) -> dict[str, Any]:
        """获取全历史 token 使用总量。"""
        async with self._lock:
            if self._conn is None:
                return {}
            row = self._conn.execute(
                """SELECT SUM(prompt_tokens), SUM(completion_tokens),
                          SUM(total_tokens), COUNT(*)
                   FROM token_usage"""
            ).fetchone()
            if not row:
                return {"prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "total_calls": 0}
            return {
                "prompt_tokens": row[0] or 0,
                "completion_tokens": row[1] or 0,
                "total_tokens": row[2] or 0,
                "total_calls": row[3] or 0,
            }

    # -- 清理 ----------------------------------------------------------------

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

@dataclass
class _OperationStats:
    """内存中的操作统计累加器。"""
    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
