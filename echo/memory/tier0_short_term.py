from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..models import LLMMessage, message_content_to_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ShortTermItem
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ShortTermItem:
    """单条短期记忆条目。"""
    content: str
    role: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.access_count += 1


# ---------------------------------------------------------------------------
# Tier 0: ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Tier 0 — 会话内短期记忆

    用途: Roleplay 层 + Decision 层的"刚刚发生了什么"即时上下文

    特征:
    - 纯内存 RingBuffer，O(1) 读写延迟
    - LRU 淘汰策略，默认 50 条
    - 存储完整消息原文（不压缩）
    - 按角色、时间、关键词检索
    """

    def __init__(self, max_size: int = 50) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size = max_size
        self._items: OrderedDict[str, ShortTermItem] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._items)

    # -- 写入 ---------------------------------------------------------------

    async def add(self, content: str, role: str = "user",
                  metadata: dict[str, Any] | None = None) -> str:
        """添加一条短期记忆，返回 key。"""
        async with self._lock:
            key = _make_key(content, role)
            now = datetime.now(timezone.utc)

            if key in self._items:
                item = self._items[key]
                item.timestamp = now
                item.access_count += 1
                if metadata:
                    item.metadata.update(metadata)
                self._items.move_to_end(key)
                return key

            while len(self._items) >= self.max_size:
                evicted_key, _ = self._items.popitem(last=False)
                logger.debug("ShortTerm evicted LRU: %s", evicted_key[:40])

            self._items[key] = ShortTermItem(
                content=content, role=role, timestamp=now,
                metadata=metadata or {},
            )
            return key

    async def add_messages(self, messages: list[LLMMessage]) -> list[str]:
        """批量从 LLMMessage 添加。"""
        keys = []
        for msg in messages:
            text = message_content_to_text(msg.content) if msg.content else ""
            if text.strip():
                k = await self.add(text.strip(), msg.role,
                                   metadata={"tool_calls": bool(msg.tool_calls)})
                keys.append(k)
        return keys

    # -- 读取 ---------------------------------------------------------------

    async def recent(self, count: int = 10) -> list[dict[str, Any]]:
        """最近 N 条消息（用于 Roleplay 上下文注入）。"""
        async with self._lock:
            items = list(self._items.items())
            return [
                {"key": k, "content": v.content[:300], "role": v.role,
                 "timestamp": v.timestamp.isoformat()}
                for k, v in reversed(items[-count:])
            ]

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """关键词检索（用于混合搜索）。"""
        if not query.strip():
            return []
        terms = set(query.lower().split())
        async with self._lock:
            scored = []
            for key, item in self._items.items():
                matches = sum(1 for t in terms if t in item.content.lower())
                if matches == 0:
                    continue
                age_h = (datetime.now(timezone.utc) - item.timestamp).total_seconds() / 3600
                score = (matches / max(len(terms), 1)) * 0.5 \
                        + min(item.access_count / 10, 1) * 0.2 \
                        + max(0, 1 - age_h / 24) * 0.3
                scored.append((score, key, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [{
                "key": k, "id": k, "content": v.content[:500], "role": v.role,
                "score": round(s, 4), "timestamp": v.timestamp.isoformat(),
                "tier": "short_term",
            } for s, k, v in scored[:max_results]]

    async def get(self, key: str) -> dict[str, Any] | None:
        """按键读取单条短期记忆（完整内容）。"""
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.touch()
            return {
                "key": key, "content": item.content, "role": item.role,
                "timestamp": item.timestamp.isoformat(),
                "access_count": item.access_count, "metadata": item.metadata,
            }

    async def delete(self, key: str) -> bool:
        """按键删除单条短期记忆。"""
        async with self._lock:
            return self._items.pop(key, None) is not None

    async def list_items(self, count: int = 20, offset: int = 0,
                         role: str | None = None) -> list[dict[str, Any]]:
        """按时间倒序分页枚举（CRUD 浏览用，比 recent 保留更多字段）。"""
        async with self._lock:
            items = [
                (k, v) for k, v in reversed(list(self._items.items()))
                if role is None or v.role == role
            ]
            return [
                {"key": k, "content": v.content[:500], "role": v.role,
                 "timestamp": v.timestamp.isoformat(),
                 "access_count": v.access_count}
                for k, v in items[offset:offset + count]
            ]

    async def as_messages(self, count: int = 10) -> list[LLMMessage]:
        """以 LLMMessage 列表返回最近消息（供 Roleplay 层直接注入）。"""
        recent = await self.recent(count)
        return [
            LLMMessage(role=r["role"], content=r["content"])
            for r in recent
        ]

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            roles = {}
            for item in self._items.values():
                roles[item.role] = roles.get(item.role, 0) + 1
            return {"size": len(self._items), "max_size": self.max_size,
                    "roles": roles}


def _make_key(content: str, role: str) -> str:
    prefix = content.strip()[:64].replace(" ", "_")
    return f"{role}:{prefix}:{hash(content) & 0xFFFFFFFF:08x}"
