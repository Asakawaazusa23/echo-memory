"""EvolutionAnalyzer — 用户模式进化分析引擎。

比 Swift EvolutionEngine 更智能的实现:
- 使用 LLM 进行深度用户模式分析
- 追踪用户偏好、习惯、知识水平的长期变化
- 发现新兴话题和兴趣转移
- 输出结构化的进化报告
- Token 消耗通过 TokenTracker 追踪
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
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import LLMProvider
    from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UserPattern:
    """识别出的用户模式。"""
    pattern_id: str
    pattern_type: str    # "preference", "habit", "knowledge", "vocabulary", "emotion", "topic", "schedule"
    description: str
    confidence: float    # 0.0 ~ 1.0
    evidence: list[str]  # 支持证据
    discovered_at: str
    updated_at: str
    strength: float      # 0.0 ~ 1.0, 模式的强度/持久度

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pattern_id,
            "type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "discovered_at": self.discovered_at,
            "updated_at": self.updated_at,
            "strength": self.strength,
        }


@dataclass
class EvolutionReport:
    """进化分析报告。"""
    patterns_found: int
    new_patterns: int
    updated_patterns: int
    topics_shifted: list[str]
    summary: str
    token_usage: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    patterns: list[UserPattern] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM 分析 Prompt
# ---------------------------------------------------------------------------

_EVOLUTION_SYSTEM_PROMPT = """你是一个用户行为分析专家。你的任务是从对话历史中分析用户的模式和特征。

请分析以下维度:
1. **偏好** (preference): 用户喜欢什么、不喜欢什么、偏好什么风格/方式
2. **习惯** (habit): 用户的常见行为模式、交流习惯
3. **知识** (knowledge): 用户的知识水平和专业领域
4. **词汇** (vocabulary): 用户常用的特殊词汇和表达方式
5. **情绪** (emotion): 用户的情绪特征和表达倾向
6. **话题** (topic): 用户关注的主要话题和兴趣领域
7. **时间** (schedule): 用户的活动时间规律

输出格式 (JSON 数组):
```json
[
  {
    "type": "preference|habit|knowledge|vocabulary|emotion|topic|schedule",
    "description": "简短描述",
    "confidence": 0.0-1.0,
    "evidence": ["证据1", "证据2"],
    "strength": 0.0-1.0
  }
]
```

规则:
- 每个模式需要有至少 1 条对话证据支持
- confidence 反映你的把握程度
- strength 反映模式的持久度和一致性
- 只输出有足够证据支持的模式，不要臆测
- 输出纯 JSON，不要包含 markdown 代码块标记"""


def _build_evolution_prompt(messages: list[dict[str, str]]) -> str:
    """构建进化分析的 user prompt。"""
    lines = ["请分析以下对话记录中的用户模式:\n"]
    for msg in messages[-50:]:  # 最近 50 条
        role = msg.get("role", "user")
        content = msg.get("content", "")[:300]
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvolutionAnalyzer
# ---------------------------------------------------------------------------

class EvolutionAnalyzer:
    """用户模式进化分析引擎。

    职责:
    - 从对话历史中提取用户模式 (偏好/习惯/知识等)
    - 持久化已发现的模式到 SQLite
    - 跟踪模式的变化 (新增、更新、衰退)
    - 生成进化报告
    - 在空闲时自动运行

    与 Swift EvolutionEngine 的区别:
    - 使用 LLM 深度分析，而非硬编码关键词
    - 支持模式的持久化和历史追踪
    - 支持置信度和强度的动态调整
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider | None = None,
        token_tracker: TokenTracker | None = None,
        db_path: Path | None = None,
        min_messages_for_analysis: int = 10,
    ) -> None:
        self._llm = llm_provider
        self._token_tracker = token_tracker
        self._db_path = db_path
        self._min_messages = min_messages_for_analysis
        self._patterns: list[UserPattern] = []
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._running = False

    # -- 生命周期 -----------------------------------------------------------

    async def initialize(self) -> None:
        """初始化 SQLite 存储。"""
        if self._db_path is None:
            return

        async with self._lock:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS evolution_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                evidence TEXT NOT NULL DEFAULT '[]',
                discovered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.5,
                is_active INTEGER NOT NULL DEFAULT 1
            )""")
            self._conn.commit()
            await self._reload_cache()
        logger.info("EvolutionAnalyzer initialized @ %s", self._db_path)

    # -- 分析 ----------------------------------------------------------------

    async def analyze_conversations(
        self, messages: list[dict[str, str]]
    ) -> EvolutionReport:
        """分析对话历史，提取用户模式。"""
        if self._running:
            return EvolutionReport(
                patterns_found=0, new_patterns=0, updated_patterns=0,
                topics_shifted=[], summary="Analysis already running",
            )

        if len(messages) < self._min_messages:
            return EvolutionReport(
                patterns_found=0, new_patterns=0, updated_patterns=0,
                topics_shifted=[],
                summary=f"Not enough messages ({len(messages)} < {self._min_messages})",
            )

        self._running = True
        start = time.monotonic()

        try:
            if self._llm:
                new_patterns = await self._analyze_with_llm(messages)
            else:
                new_patterns = self._analyze_with_rules(messages)

            # 合并与更新
            new_count = 0
            updated_count = 0
            topics_shifted: list[str] = []

            for pattern in new_patterns:
                existing = self._find_similar_pattern(pattern)
                if existing:
                    # 更新已有模式
                    existing.confidence = max(existing.confidence, pattern.confidence)
                    existing.strength = min(existing.strength + 0.1, 1.0)
                    existing.updated_at = pattern.updated_at
                    existing.description = pattern.description
                    existing.evidence = pattern.evidence
                    updated_count += 1
                else:
                    self._patterns.append(pattern)
                    new_count += 1

            # 衰减长期未更新的模式
            self._decay_stale_patterns()

            # 检测话题转移
            topics_shifted = self._detect_topic_shift()

            # 持久化
            await self._persist_patterns()

            duration_ms = (time.monotonic() - start) * 1000

            return EvolutionReport(
                patterns_found=len(self._patterns),
                new_patterns=new_count,
                updated_patterns=updated_count,
                topics_shifted=topics_shifted,
                summary=self._generate_summary(),
                duration_ms=duration_ms,
                patterns=list(self._patterns),
            )

        finally:
            self._running = False

    async def _analyze_with_llm(
        self, messages: list[dict[str, str]]
    ) -> list[UserPattern]:
        """使用 LLM 进行深度分析。"""
        from ..models import LLMMessage

        prompt = _build_evolution_prompt(messages)
        llm_messages = [
            LLMMessage(role="system", content=_EVOLUTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]

        start = time.monotonic()
        response = await self._llm.generate(
            llm_messages,
            temperature=0.3,
            max_tokens=2048,
        )
        duration_ms = (time.monotonic() - start) * 1000

        # 记录 token
        if self._token_tracker:
            await self._token_tracker.record_from_response(
                "evolution",
                response.usage,
                model=response.model,
                duration_ms=duration_ms,
                detail=f"messages_analyzed={len(messages)}",
            )

        # 解析 JSON
        content = response.message.content
        if isinstance(content, list):
            texts = [seg.get("text", "") for seg in content if isinstance(seg, dict)]
            content = " ".join(texts)

        return self._parse_llm_output(str(content))

    def _parse_llm_output(self, text: str) -> list[UserPattern]:
        """解析 LLM 输出的 JSON 模式列表。"""
        # 清理 markdown 代码块
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # 去掉第一行和最后一行的 ```
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].startswith("```"):
                lines = lines[1:]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取 JSON 数组
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM evolution output")
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        now = datetime.now(timezone.utc).isoformat()
        patterns = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ptype = item.get("type", "")
            if ptype not in {"preference", "habit", "knowledge", "vocabulary",
                             "emotion", "topic", "schedule"}:
                continue
            patterns.append(UserPattern(
                pattern_id=f"evo_{_short_hash(item.get('description', ''))}",
                pattern_type=ptype,
                description=item.get("description", ""),
                confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                evidence=item.get("evidence", [])[:5],
                discovered_at=now,
                updated_at=now,
                strength=min(max(float(item.get("strength", 0.5)), 0.0), 1.0),
            ))

        return patterns

    def _analyze_with_rules(
        self, messages: list[dict[str, str]]
    ) -> list[UserPattern]:
        """不使用 LLM 的规则分析 (降级方案)。"""
        from collections import Counter

        now = datetime.now(timezone.utc).isoformat()
        patterns: list[UserPattern] = []
        user_texts = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        # 话题检测
        topic_keywords = {
            "编程/开发": ["code", "代码", "编程", "bug", "api", "python",
                        "javascript", "typescript", "swift", "function", "debug"],
            "游戏": ["游戏", "玩", "steam", "switch", "ps5", "xbox", "game"],
            "动漫/二次元": ["动漫", "二次元", "番剧", "漫画", "anime", "manga"],
            "音乐": ["音乐", "歌", "听", "album", "spotify"],
            "学习": ["学习", "读书", "考试", "教程", "learn", "study"],
            "工作": ["工作", "项目", "deadline", "会议", "team"],
            "美食": ["吃", "美食", "做饭", "菜", "餐厅"],
            "运动": ["运动", "跑步", "健身", "瑜伽", "sport"],
        }
        lower_text = user_texts.lower()
        for topic, keywords in topic_keywords.items():
            count = sum(1 for kw in keywords if kw in lower_text)
            if count >= 3:
                patterns.append(UserPattern(
                    pattern_id=f"evo_topic_{_short_hash(topic)}",
                    pattern_type="topic",
                    description=f"用户关注话题: {topic} (匹配{count}次)",
                    confidence=min(count * 0.1, 0.8),
                    evidence=[f"关键词命中{count}次"],
                    discovered_at=now,
                    updated_at=now,
                    strength=min(count * 0.05, 0.7),
                ))

        # 情绪倾向
        positive = ["喜欢", "开心", "好", "棒", "谢谢", "赞", "great",
                    "good", "love", "perfect", "赞"]
        negative = ["烦", "讨厌", "不好", "差", "不行", "生气", "bad",
                    "hate", "angry", "frustrated"]
        pos_count = sum(1 for w in positive if w in lower_text)
        neg_count = sum(1 for w in negative if w in lower_text)
        if pos_count + neg_count >= 3:
            dominant = "积极" if pos_count >= neg_count else "消极"
            patterns.append(UserPattern(
                pattern_id=f"evo_emotion_{dominant}",
                pattern_type="emotion",
                description=f"用户情绪倾向: {dominant} (积极:{pos_count}, 消极:{neg_count})",
                confidence=min((pos_count + neg_count) * 0.08, 0.7),
                evidence=[f"积极词{pos_count}次, 消极词{neg_count}次"],
                discovered_at=now,
                updated_at=now,
                strength=0.5,
            ))

        return patterns

    # -- 模式管理 ------------------------------------------------------------

    def _find_similar_pattern(self, new: UserPattern) -> UserPattern | None:
        """查找是否已存在类似的模式。"""
        for p in self._patterns:
            if p.pattern_type != new.pattern_type:
                continue
            # 简单文本相似度: 共同词数 / 最大词数
            words_a = set(p.description.lower().split())
            words_b = set(new.description.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            if overlap > 0.4:
                return p
        return None

    def _decay_stale_patterns(self, decay_rate: float = 0.05) -> None:
        """衰减长期未更新的模式强度。"""
        now = datetime.now(timezone.utc)
        for p in self._patterns:
            try:
                updated = datetime.fromisoformat(p.updated_at)
                days_since = (now - updated).total_seconds() / 86400
                if days_since > 7:
                    p.strength = max(p.strength - decay_rate * days_since, 0.1)
            except (ValueError, TypeError):
                pass

        # 移除强度过低的模式
        self._patterns = [p for p in self._patterns if p.strength > 0.1]

    def _detect_topic_shift(self) -> list[str]:
        """检测最近新出现的话题。"""
        recent_topics = [
            p.description for p in self._patterns
            if p.pattern_type == "topic" and p.strength > 0.3
        ]
        return recent_topics[:5]

    def _generate_summary(self) -> str:
        """生成模式摘要。"""
        if not self._patterns:
            return "未发现显著的用户模式。"

        by_type: dict[str, list[UserPattern]] = {}
        for p in self._patterns:
            by_type.setdefault(p.pattern_type, []).append(p)

        parts = [f"共发现 {len(self._patterns)} 个用户模式:"]
        type_names = {
            "preference": "偏好", "habit": "习惯", "knowledge": "知识",
            "vocabulary": "词汇", "emotion": "情绪", "topic": "话题",
            "schedule": "时间",
        }
        for ptype, patterns in sorted(by_type.items()):
            name = type_names.get(ptype, ptype)
            best = max(patterns, key=lambda p: p.confidence * p.strength)
            parts.append(f"- {name}: {best.description} "
                        f"(置信度:{best.confidence:.1f}, 强度:{best.strength:.1f})")

        return "\n".join(parts)

    # -- 持久化 --------------------------------------------------------------

    async def _persist_patterns(self) -> None:
        """将模式持久化到 SQLite。"""
        async with self._lock:
            if self._conn is None:
                return
            for p in self._patterns:
                self._conn.execute(
                    """INSERT OR REPLACE INTO evolution_patterns
                       (pattern_id, pattern_type, description, confidence,
                        evidence, discovered_at, updated_at, strength, is_active)
                       VALUES (?,?,?,?,?,?,?,?,1)""",
                    (p.pattern_id, p.pattern_type, p.description,
                     p.confidence, json.dumps(p.evidence, ensure_ascii=False),
                     p.discovered_at, p.updated_at, p.strength),
                )
            self._conn.commit()

    async def _reload_cache(self) -> None:
        """从 SQLite 加载模式到内存。"""
        if self._conn is None:
            return
        rows = self._conn.execute(
            """SELECT pattern_id, pattern_type, description, confidence,
                      evidence, discovered_at, updated_at, strength
               FROM evolution_patterns
               WHERE is_active = 1
               ORDER BY strength DESC"""
        ).fetchall()
        self._patterns = [
            UserPattern(
                pattern_id=r[0], pattern_type=r[1], description=r[2],
                confidence=r[3],
                evidence=json.loads(r[4]) if r[4] else [],
                discovered_at=r[5], updated_at=r[6], strength=r[7],
            )
            for r in rows
        ]

    # -- 查询 ----------------------------------------------------------------

    async def get_patterns(
        self, pattern_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """获取用户模式列表。"""
        result = []
        for p in self._patterns:
            if pattern_type and p.pattern_type != pattern_type:
                continue
            if p.confidence < min_confidence:
                continue
            result.append(p.to_dict())
        return result

    async def get_active_topics(self) -> list[str]:
        """获取当前活跃话题。"""
        return [
            p.description for p in self._patterns
            if p.pattern_type == "topic" and p.strength > 0.3
        ]

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _short_hash(text: str) -> str:
    """生成文本的短哈希。"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()[:8]
