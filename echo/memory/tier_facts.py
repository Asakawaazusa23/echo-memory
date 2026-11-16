from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import struct
from collections import Counter
from datetime import date, datetime, timedelta
from math import log
from pathlib import Path
from typing import Any

from ..models import LLMMessage, message_content_to_text
from .zh_segment import CN_NUMERAL_CHARS, query_features, segment
from .zh_synonyms import synonyms

logger = logging.getLogger(__name__)

# BM25 参数：中文短摘要，长度归一不宜过强。
_BM25_K1 = 1.2
_BM25_B = 0.75
# 同义词/伪相关反馈扩展词相对原查询词的权重，压低词典噪声的影响。
_SYNONYM_WEIGHT = 0.35
_RM3_WEIGHT = 0.4
# 数字/日期与专名是事实型问题的强线索，单独加权。
_NUMERAL_WEIGHT = 1.6
_PROPER_WEIGHT = 1.8
# 归档段向量加分：只对明显高于背景相似度（长摘要普遍 0.55~0.62）的段生效。
_VEC_FLOOR = 0.60
_VEC_BONUS = 1.0
# 事件卡：规则词典归一动作，命中后按时间片取回原文窗口。无 LLM。
# 只从用户消息抽事件；助手的话可能带推测，不能当作用户的承诺或状态。
_EVENT_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shift_cover", ("替班", "顶班", "代班", "替个班", "替一天", "顶一天", "替小赵", "替罗姐",
                     "替他值", "替她值", "替她盯", "替他盯", "盯窗口", "盯一整天", "换班")),
    ("debt", ("还钱", "还我", "还你", "还他", "还她", "还清", "结清", "借了", "借我", "找我借",
              "转账", "垫付", "垫了", "欠着", "欠我")),
    ("treat", ("请客", "请饭", "请吃", "请一顿", "请我", "请他", "请她", "回礼", "买单")),
    ("birthday", ("生日", "过寿")),
    ("leave", ("补休", "调休", "请假", "休假", "休一整天", "休半天", "休一天", "歇一天")),
    ("deadline", ("月底前", "年前还", "之前还", "到期", "截止")),
    ("appointment", ("约好", "约了", "碰头", "见面", "挂号", "体检")),
)
_EVENT_GENERIC = frozenset({
    "今天", "现在", "时候", "什么", "怎么", "这些", "那些", "一个", "自己", "觉得",
    "可以", "应该", "还是", "就是", "这样", "那样", "一直", "已经", "因为", "所以",
    "如果", "然后", "最近", "有点", "一点", "一下", "不是", "没有", "知道", "咱们",
})
_EVENT_MIN_SCORE = 1.0
_EVENT_ACTION_WEIGHT = 2.0
_EVENT_OVERLAP_WEIGHT = 1.5
_EVENT_PROPER_WEIGHT = 0.8

# 到期日历：用户说过的、指向未来某天的约定，到期当天主动提起。
_WEEKDAY_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_DUE_PAST_MARKERS = ("昨天", "前天", "上周", "上礼拜", "上个月", "去年", "已经", "早就", "过了")
_DUE_WEEK_PREFIX = {"下下": 2, "下": 1, "这": 0, "本": 0, "": 0}
# 强约定动词：还钱、考试、请客、约会、取件、到货、复诊这类有对方的日程
_DUE_STRONG = ("还钱", "还我", "还他", "还她", "还清", "交钱", "交表", "交作业", "考试", "考科目",
               "请我", "请他", "请她", "请客", "约好", "约了", "见面", "碰头", "取件", "取镜",
               "取快递", "到驿站", "到货", "体检", "复诊", "截止", "到期", "生日", "面试", "开庭")
# 弱日程：只是自己的安排，提醒价值低，仅在没命中强约定时兜底
_DUE_WEAK = ("去", "来", "办", "带", "拿", "寄", "送", "跑一趟", "洗牙", "缴费")
_DUE_MAX_AGE_DAYS = 180
_WEEKDAY_CN = "一二三四五六日"
# 递归/长期约定不算“今天到期”，这类每天都会提到
_DUE_RECURRING = ("每天", "每周", "天天", "连着", "周一到周", "这一周")
# 疑问、假设、取消：不强制提醒
_DUE_HEDGE = ("要是", "如果", "要不要", "吗", "?", "？", "不确定", "再说", "取消", "算了")
# 隐式状态维度：问题问到某类状态时，只补同维度的实体证据
_ENTITY_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weight", ("胖", "瘦", "体重", "称", "斤", "公斤", "减肥")),
    ("feeding", ("吃", "喂", "粮", "罐头", "零食", "小鱼干", "猫条", "干粮", "喝")),
    ("money", ("借", "还", "欠", "块钱", "块钱", "转账", "红包", "小费")),
    ("health", ("病", "疼", "医院", "吃药", "体检", "复诊", "挂号", "检查")),
    ("work", ("窗口", "档案", "值班", "开会", "述职", "领导", "加班")),
    ("date", ("生日", "纪念日", "节日", "过年", "过节")),
)


def resolve_due_dates(text: str, base: date) -> set[date]:
    """把一句话里的时间表达解析成绝对日期；只返回不早于说话日的日期。

    支持：大后天/后天/明天、下下周X/下周X/这周X/周X、X月X日、X号、月底。
    带明确过去标记（昨天/上周/已经…）的句子整句跳过，避免把已完成的事当到期。
    """
    out: set[date] = set()
    if any(marker in text for marker in _DUE_PAST_MARKERS):
        return out
    if "大后天" in text:
        out.add(base + timedelta(days=3))
    elif "后天" in text:
        out.add(base + timedelta(days=2))
    if re.search(r"明天|明日", text):
        out.add(base + timedelta(days=1))
    for match in re.finditer(
            r"(下下|下|这|本)?(?:周|礼拜|星期)([一二三四五六日天])"
            r"(?!说|聊|提|问|讲|告诉)", text):
        week = _DUE_WEEK_PREFIX[match.group(1) or ""]
        target = _WEEKDAY_INDEX[match.group(2)]
        monday = base - timedelta(days=base.weekday())
        delta = week * 7 + target
        if week == 0 and target < base.weekday():
            delta += 7
        out.add(monday + timedelta(days=delta))
    for match in re.finditer(r"(\d{1,2})月(\d{1,2})[日号]", text):
        try:
            day = date(base.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            continue
        if day < base:
            day = date(base.year + 1, day.month, day.day)
        out.add(day)
    for match in re.finditer(r"(?<![月\d])(\d{1,2})[号日](?![\d月])", text):
        day_num = int(match.group(1))
        if not 1 <= day_num <= 31:
            continue
        try:
            day = date(base.year, base.month, day_num)
        except ValueError:
            continue
        if day < base:
            year, month = (base.year + 1, 1) if base.month == 12 else (base.year, base.month + 1)
            try:
                day = date(year, month, day_num)
            except ValueError:
                continue
        out.add(day)
    if re.search(r"月底|月末", text):
        nxt = date(base.year + (base.month == 12), base.month % 12 + 1, 1)
        out.add(nxt - timedelta(days=1))
    return out


def _event_actions(text: str) -> frozenset[str]:
    """把一句话归一到动作类型集合；只做词典匹配，不做推断。"""
    return frozenset(name for name, keys in _EVENT_ACTIONS
                     if any(key in text for key in keys))


def _event_terms(text: str) -> frozenset[str]:
    return frozenset(term for term in segment(text, max_terms=48)
                     if len(term) >= 2 and term not in _EVENT_GENERIC)


def _event_score(event_actions: frozenset[str], event_terms: frozenset[str],
                 event_text: str, q_terms: set[str], q_actions: set[str],
                 proper: tuple[str, ...]) -> float:
    """事件相关性：动作同类 > 专名/对象重合 > 词项覆盖；无任何信号则 0。"""
    signal = 0.0
    if q_actions & event_actions:
        signal += _EVENT_ACTION_WEIGHT
    overlap = q_terms & event_terms
    if overlap:
        signal += _EVENT_OVERLAP_WEIGHT * len(overlap) / max(1, len(q_terms))
    text = event_text.lower()
    hits = sum(1 for name in proper if name in text)
    if hits:
        signal += _EVENT_PROPER_WEIGHT * min(3, hits)
    if signal <= 0:
        return 0.0
    return signal + 0.5  # 带动作的事件比纯寒暄更可能是"事"


def _parse_ts(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _segment_prompt(max_lines: int) -> str:
    """按时间分段归档的整合提示词：一次调用、编号条目、无JSON、无逐条引文。"""
    return (
        "把下面一段对话整理成按时间归档的记录。只输出编号条目，不要解释、标题或JSON。\n"
        "格式：每行「序号. 主体+事件+关键值」。整段共用一个记录时间，条目内不重复写时间。\n"
        "规则：\n"
        "1. 只记用户亲身经历、约定、偏好、数字、地点、人物、计划；助手建议未经用户确认不记。\n"
        "2. 否定、条件、改口按原意保留，不同时间的值分行写，不要合并或覆盖。\n"
        "3. 未知信息不补全；代词换成本名；相对时间换成日期。\n"
        "4. 寒暄、重复确认、纯情绪表达不记。\n"
        "5. 关键名词照抄用户原话：物品名、书名、影音名、店名、地名、人名、金额、数量、体重等，"
        "不要换成同义词或更宽泛的说法；动作动词也按用户说法保留（借出/归还/退回/没取回/没收回）。\n"
        "6. 用户提出的待办、盘点、疑问要留下提问口径，"
        "例如用户问「哪些书借出去没收回」，就写「用户想知道哪些书借出后尚未收回」。\n"
        f"7. 最多{max_lines}条，每条不超过48字；没有可记内容就输出“无”。"
    )


class FactsMemory:
    """带原文证据的独立事实索引；不把事件提升为全局核心记忆。

    facts_segment_mode 开启时按时间分段归档原始对话（一次调用整理一段），
    关闭时沿用逐消息抽取原子事实。两者共用 sources 表，extracted 表示已处理。
    """

    def __init__(self, db_path: Path, batch_messages: int = 2, *,
                 segment_mode: bool = True, segment_gap_seconds: int = 900,
                 segment_max_messages: int = 40, segment_max_lines: int = 10,
                 segment_max_chars: int = 2400, segment_bm25: bool = True,
                 segment_synonyms: bool = True, segment_rm3: bool = False,
                 segment_vectors: bool = True,
                 embedding_model: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.path = db_path
        self.batch_messages = batch_messages
        self.segment_mode = segment_mode
        self.segment_gap_seconds = segment_gap_seconds
        self.segment_max_messages = segment_max_messages
        self.segment_max_lines = segment_max_lines
        self.segment_max_chars = segment_max_chars
        self.segment_bm25 = segment_bm25
        self.segment_synonyms = segment_synonyms
        self.segment_rm3 = segment_rm3
        # 归档段复用 Tier2 同款中文向量模型，补足“问法与归档转述错位”的词法盲区。
        self.segment_vectors = segment_vectors
        self.embedding_model = embedding_model
        self._conn: sqlite3.Connection | None = None
        self._extract_lock = asyncio.Lock()
        self._term_cache: dict[str, set[str]] = {}
        self._segment_terms: dict[str, Counter[str]] = {}
        self._embedder: Any = None
        self._embedder_failed = False
        self._vec_ids: list[str] = []
        self._vec_mat: Any = None
        self._vec_dirty = True
        self._event_index_ready = False
        self._due_cache: tuple[tuple[str, int], list[dict[str, Any]]] | None = None
        # sources 的 BM25 索引：(行数, docs, idf, avgdl)，按需构建
        self._source_docs: tuple[int, list, dict, float] | None = None
        self.stats = {"extract_calls": 0, "query_calls": 0, "verify_calls": 0,
                      "segment_calls": 0, "segments_written": 0,
                      "errors": 0, "rejected_facts": 0}

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                timestamp TEXT NOT NULL, extracted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
                summary TEXT NOT NULL, source_count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS segment_vectors (
                id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL, ts TEXT NOT NULL,
                action TEXT NOT NULL, text TEXT NOT NULL, terms TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        """)
        self._conn.commit()

    async def add_sources(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._conn.execute(
                "INSERT OR IGNORE INTO sources(id,role,content,timestamp) VALUES (?,?,?,?)",
                (message["id"], message["role"], message["content"], message["timestamp"]),
            )
            self._record_event(message)
        self._conn.commit()

    def _record_event(self, message: dict[str, Any]) -> None:
        """用户消息命中动作词典就写一张事件卡；重复消息按 id 幂等。"""
        if message.get("role") == "assistant":
            return
        text = str(message.get("content") or "")
        actions = _event_actions(text)
        if not actions:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO events(id,source_id,ts,action,text,terms) "
            "VALUES (?,?,?,?,?,?)",
            (str(message["id"]), str(message["id"]), str(message["timestamp"]),
             ",".join(sorted(actions)), text,
             " ".join(sorted(_event_terms(text)))),
        )

    def _group_segments(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """按插入顺序切分时间片：间隔超阈值、条数或字符上限即断开。"""
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        chars = 0
        last_ts: datetime | None = None
        for row in rows:
            ts = _parse_ts(row["timestamp"])
            gap = (ts - last_ts).total_seconds() if ts and last_ts else 0.0
            overflow = (len(current) >= self.segment_max_messages
                        or chars + len(row["content"]) > self.segment_max_chars)
            if current and (gap >= self.segment_gap_seconds or overflow):
                groups.append(current)
                current, chars = [], 0
            current.append(row)
            chars += len(row["content"])
            last_ts = ts
        if current:
            groups.append(current)
        return groups

    async def consolidate_pending(self, provider: Any, *, force: bool = False) -> None:
        """把未处理的原始对话按时间片整理成编号记录，一段一次调用。

        非 force 时保留最后一个时间片，等下一段对话开始或 flush 再整理，
        避免把仍在进行的对话过早归档。
        """
        if provider is None:
            return
        async with self._extract_lock:
            while True:
                rows = [dict(row) for row in self._conn.execute(
                    "SELECT id,role,content,timestamp FROM sources "
                    "WHERE extracted=0 ORDER BY rowid")]
                if not rows:
                    return
                groups = self._group_segments(rows)
                if not force:
                    groups = groups[:-1]
                if not groups:
                    return
                for group in groups:
                    await self._consolidate_group(provider, group)

    async def _consolidate_group(self, provider: Any, rows: list[dict[str, Any]]) -> None:
        start, end = rows[0]["timestamp"], rows[-1]["timestamp"]
        lines = []
        for row in rows:
            text = " ".join(row["content"].split())
            if len(text) > 240:
                text = text[:240] + "…"
            lines.append(("用户" if row["role"] != "assistant" else "助手") + "：" + text)
        body = "\n".join(lines)[:self.segment_max_chars]
        # 指令段到此结束；"记录时间／对话" 是运行时填入的数据区，不是示例。
        prompt = (f"{_segment_prompt(self.segment_max_lines)}\n"
                  f"记录时间：{start}~{end}\n对话：\n{body}")
        self.stats["segment_calls"] += 1
        try:
            response = await provider.generate(
                [LLMMessage(role="user", content=prompt)], temperature=0,
                max_tokens=min(640, 120 + self.segment_max_lines * 56),
            )
            text = message_content_to_text(response.message.content).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            summary = self._clean_summary(text)
            if summary:
                digest = hashlib.sha256(
                    (start + end + rows[-1]["id"]).encode()).hexdigest()
                self._conn.execute(
                    "INSERT OR REPLACE INTO segments VALUES (?,?,?,?,?)",
                    (digest, start, end, summary, len(rows)))
                self._segment_terms.pop("seg:" + digest, None)
                await self._store_segment_vector(digest, summary)
                self.stats["segments_written"] += 1
            self._conn.executemany("UPDATE sources SET extracted=1 WHERE id=?",
                                   [(row["id"],) for row in rows])
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            self.stats["errors"] += 1
            logger.exception("tier_facts segment consolidation failed; sources retained")

    def _clean_summary(self, text: str) -> str:
        """只保留编号条目，压缩空行与超长内容；无有效条目则返回空串。"""
        entries = [line.strip() for line in text.splitlines() if line.strip()]
        numbered = [line for line in entries if line[0].isdigit()]
        if numbered:
            entries = numbered
        entries = [line for line in entries if line not in {"无", "无记录", "（无）"}]
        return "\n".join(entries[:self.segment_max_lines])

    # -- 归档段向量 --------------------------------------------------------

    def _embedder_instance(self) -> Any:
        if self._embedder is not None or self._embedder_failed:
            return self._embedder
        try:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self.embedding_model)
        except Exception as exc:  # noqa: BLE001
            self._embedder_failed = True
            logger.warning("tier_facts vector search unavailable (%s)", type(exc).__name__)
        return self._embedder

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._embedder_instance()
        if model is None or not texts:
            return []
        return [[float(x) for x in vec] for vec in model.embed(texts)]

    async def _store_segment_vector(self, segment_id: str, summary: str) -> None:
        if not self.segment_vectors:
            return
        try:
            vectors = await asyncio.to_thread(self._embed_sync, [summary])
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            logger.warning("tier_facts segment embedding failed; lexical only")
            return
        if not vectors:
            return
        vec = vectors[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO segment_vectors VALUES (?,?,?)",
            (segment_id, len(vec), struct.pack(f"<{len(vec)}f", *vec)),
        )
        self._vec_dirty = True

    def _vector_matrix(self) -> Any:
        if not self._vec_dirty:
            return self._vec_mat
        try:
            import numpy as np
        except ImportError:
            self._vec_dirty = False
            self._vec_mat = None
            self._vec_ids = []
            return None
        ids, rows = [], []
        for segment_id, dim, blob in self._conn.execute(
                "SELECT id,dim,vec FROM segment_vectors"):
            if dim <= 0 or len(blob) != dim * 4:
                continue
            ids.append(segment_id)
            rows.append(np.frombuffer(blob, dtype=np.float32))
        self._vec_ids = ids
        if not rows:
            self._vec_mat = None
        else:
            mat = np.vstack(rows)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vec_mat = mat / norms
        self._vec_dirty = False
        return self._vec_mat

    def _vector_scores(self, query: str) -> dict[str, float]:
        if not self.segment_vectors:
            return {}
        try:
            import numpy as np

            mat = self._vector_matrix()
            if mat is None:
                return {}
            vectors = self._embed_sync([query])
            if not vectors:
                return {}
            q = np.asarray(vectors[0], dtype=np.float32)
            if q.shape[0] != mat.shape[1]:
                return {}
            norm = float(np.linalg.norm(q)) or 1.0
            sims = mat @ (q / norm)
            return dict(zip(self._vec_ids, (float(x) for x in sims)))
        except Exception:  # noqa: BLE001
            logger.warning("tier_facts vector scoring failed; lexical only")
            return {}

    async def search_segments(self, query: str, max_results: int = 5, *,
                              query_time: Any = None,
                              max_chars: int = 1200) -> list[dict[str, Any]]:
        """词法召回时间片，按时间正序返回，不调用LLM。

        默认 BM25（df 在 segments 内自算，数字/日期词加权）；可叠加同义词扩展
        与 RM3 伪相关反馈。三者都只做词项加权，不引入新的 LLM 调用。
        """
        if max_results <= 0 or not query.strip():
            return []
        query_terms = segment(query, max_terms=40)
        if not query_terms:
            return []

        docs: list[tuple[sqlite3.Row, Counter[str]]] = []
        for row in self._conn.execute(
                "SELECT id,start_ts,end_ts,summary FROM segments"):
            if query_time is not None:
                end = _parse_ts(row["end_ts"])
                now = _parse_ts(query_time)
                if end is None or now is None or end > now:
                    continue
            docs.append((row, self._segment_counts(row["id"], row["summary"])))
        if not docs:
            return []

        if not self.segment_bm25:
            weights = {term: 1.0 for term in query_terms}
            scored = self._lexical_overlap(docs, weights)
        else:
            frequencies = Counter()
            for _, counts in docs:
                frequencies.update(counts.keys())
            idf = self._idf(frequencies, len(docs))
            avgdl = sum(sum(counts.values()) for _, counts in docs) / len(docs)
            weights = self._query_weights(query, query_terms, idf, len(docs))
            scored = self._bm25(docs, weights, avgdl)
            if self.segment_rm3 and scored:
                weights = self._rm3_weights(scored, weights, idf,
                                            {row["id"]: counts for row, counts in docs})
                scored = self._bm25(docs, weights, avgdl)
        scored = self._fuse_vectors(query, scored)

        ranked = []
        for row, score in scored:
            if score <= 0:
                continue
            ranked.append({"id": "segments:" + row["id"], "evidence_id": row["id"],
                           "content": f"记录时间 {row['start_ts']}~{row['end_ts']}\n"
                                      f"{row['summary']}",
                           "source": "archive", "timestamp": row["start_ts"],
                           "tier": "segments", "score": round(score, 4),
                           "end_ts": row["end_ts"]})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = sorted(ranked[:max_results], key=lambda item: item["timestamp"])
        out, used = [], 0
        for row in selected:
            if used + len(row["content"]) > max_chars and out:
                break
            out.append(row)
            used += len(row["content"])
        return out

    async def recent_segments(self, query_time: Any, *, before: Any = None,
                              hours: float = 72.0,
                              max_results: int = 1) -> list[dict[str, Any]]:
        """取 cutoff 之前最近的归档段（按 end_ts 倒排，不看相关性）。

        用于情景式问题的中时段背景兜底：这类问题没有可检索锚点，命中不了任何段。
        before 给定时只取该时间点之前的段——实测「最近段」常常整段落在最近 6 轮
        原文覆盖的时间窗内，注入等于重复，所以调用方应传入最近对话窗口的起点。
        距 cutoff 超过 hours 时返回空，避免静默期后把很久以前的旧事当近期背景。
        """
        now = _parse_ts(query_time)
        if now is None or max_results <= 0:
            return []
        cutoff = _parse_ts(before) if before is not None else None
        ceiling = min(now, cutoff) if cutoff is not None else now
        lower = now.timestamp() - hours * 3600
        picked: list[sqlite3.Row] = []
        for row in self._conn.execute(
                "SELECT id,start_ts,end_ts,summary FROM segments "
                "ORDER BY end_ts DESC LIMIT 400"):
            end = _parse_ts(row["end_ts"])
            if end is None or end > ceiling:
                continue
            if end.timestamp() < lower:
                break
            picked.append(row)
            if len(picked) >= max_results:
                break
        return [{
            "id": "segments:" + row["id"], "evidence_id": row["id"],
            "content": f"记录时间 {row['start_ts']}~{row['end_ts']}\n{row['summary']}",
            "source": "archive", "timestamp": row["start_ts"], "tier": "segments",
            "score": 0.0, "end_ts": row["end_ts"], "recent": True,
        } for row in picked]

    async def segment_window(self, timestamp: Any, *, query_time: Any = None,
                             slack_hours: float = 6.0) -> dict[str, Any] | None:
        """返回包含该时间点的归档段时间范围，供事件链把种子轮映射回原文窗口。

        没有正好覆盖的段时取最近的一段（距离不超过 slack_hours），都没有则返回空。
        """
        target = _parse_ts(timestamp)
        if target is None:
            return None
        ceiling = _parse_ts(query_time) if query_time is not None else None
        best: tuple[float, dict[str, Any]] | None = None
        for row in self._conn.execute(
                "SELECT id,start_ts,end_ts,summary FROM segments"):
            start, end = _parse_ts(row["start_ts"]), _parse_ts(row["end_ts"])
            if start is None or end is None:
                continue
            if ceiling is not None and end > ceiling:
                continue
            if start <= target <= end:
                return dict(row)
            distance = min(abs((target - start).total_seconds()),
                           abs((target - end).total_seconds()))
            if best is None or distance < best[0]:
                best = (distance, dict(row))
        if best is not None and best[0] <= slack_hours * 3600:
            return best[1]
        return None

    def _event_window_text(self, window: dict[str, Any], seed_ts: str,
                           max_chars: int) -> str:
        """把时间片内的原始对话拼成事件行；离种子越近的轮越优先保留。"""
        rows = [dict(row) for row in self._conn.execute(
            "SELECT role,content,timestamp FROM sources "
            "WHERE timestamp >= ? AND timestamp <= ? ORDER BY rowid",
            (window["start_ts"], window["end_ts"]))]
        if not rows:
            return ""
        seed = _parse_ts(seed_ts)

        def distance(row: dict[str, Any]) -> float:
            ts = _parse_ts(row["timestamp"])
            return abs((ts - seed).total_seconds()) if ts is not None and seed is not None else 1e9

        order = sorted(range(len(rows)), key=lambda i: (distance(rows[i]), i))
        picked: list[tuple[int, str]] = []
        used = 0
        for i in order:
            row = rows[i]
            speaker = "小满" if row["role"] == "assistant" else "用户"
            body = " ".join(str(row["content"]).split())
            line = f"[{row['timestamp']}] {speaker}：{body}"[:240]
            if picked and used + len(line) > max_chars:
                continue
            picked.append((i, line))
            used += len(line)
        picked.sort()
        return "\n".join(line for _, line in picked)

    async def ensure_event_index(self) -> None:
        """老库/评测模板没有事件卡时，从 sources 一次性回填（幂等）。"""
        if self._event_index_ready:
            return
        self._event_index_ready = True
        if self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
            return
        for row in self._conn.execute(
                "SELECT id,role,content,timestamp FROM sources WHERE role!='assistant'"):
            self._record_event({"id": row["id"], "role": row["role"],
                                "content": row["content"], "timestamp": row["timestamp"]})
        self._conn.commit()

    async def search_events(self, query: str, max_results: int = 5, *,
                            query_time: Any = None, seeds: int = 6,
                            max_chars: int = 900) -> list[dict[str, Any]]:
        """事件检索通道：查询→事件卡→种子轮→所在时间片的原文。

        只读；同一时间片只出一行，避免一段内的重复短句吃光名额。
        """
        if max_results <= 0 or not query.strip():
            return []
        await self.ensure_event_index()
        proper, _ = query_features(query)
        q_terms = set(_event_terms(query))
        q_actions = set(_event_actions(query))
        now = _parse_ts(query_time)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._conn.execute("SELECT id,ts,action,text,terms FROM events"):
            ts = _parse_ts(row["ts"])
            if ts is None or (now is not None and ts > now):
                continue
            event = {"id": row["id"], "ts": row["ts"], "text": row["text"],
                     "actions": frozenset(row["action"].split(",")),
                     "terms": frozenset(row["terms"].split())}
            score = _event_score(event["actions"], event["terms"], event["text"],
                                 q_terms, q_actions, proper)
            if score >= _EVENT_MIN_SCORE:
                scored.append((score, event))
        scored.sort(key=lambda item: (item[0], item[1]["ts"]), reverse=True)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        scanned = 0
        for score, event in scored:
            if len(out) >= max_results or scanned >= seeds * 3:
                break
            scanned += 1
            window = await self.segment_window(event["ts"], query_time=query_time)
            key = str(window["id"]) if window else "turn:" + event["id"]
            if key in seen:
                continue
            seen.add(key)
            content = (self._event_window_text(window, event["ts"], max_chars)
                       if window else f"[{event['ts']}] 用户：{event['text']}"[:max_chars])
            if not content:
                continue
            out.append({
                "id": "events:" + event["id"], "content": content,
                "timestamp": event["ts"], "tier": "event", "score": round(score, 4),
                "action": sorted(event["actions"]), "seed": event["id"],
                "segment": window["id"] if window else None,
            })
        return out

    def _due_candidates(self, today: date) -> list[dict[str, Any]]:
        """扫用户原话，挑出“今天到期”的约定；强约定在前，同组按时间倒序。"""
        strong: list[dict[str, Any]] = []
        weak: list[dict[str, Any]] = []
        for row in self._conn.execute(
                "SELECT content,timestamp FROM sources WHERE role!='assistant'"):
            ts = _parse_ts(row["timestamp"])
            if ts is None or ts.date() > today:
                continue
            age = (today - ts.date()).days
            if age > _DUE_MAX_AGE_DAYS:
                continue
            text = str(row["content"])
            if any(key in text for key in _DUE_RECURRING):
                continue
            if any(key in text for key in _DUE_HEDGE):
                continue
            if today not in resolve_due_dates(text, ts.date()):
                continue
            item = {"ts": row["timestamp"], "text": text, "days": age}
            if any(key in text for key in _DUE_STRONG):
                strong.append(item)
            elif any(key in text for key in _DUE_WEAK):
                weak.append(item)
        strong.sort(key=lambda item: item["ts"], reverse=True)
        weak.sort(key=lambda item: item["ts"], reverse=True)
        return strong + weak

    async def search_due(self, query_time: Any, max_results: int = 2) -> list[dict[str, Any]]:
        """到期日历：今天该主动提起的约定（原话＋几天前说过），没有则返回空。

        只认“说话时还没到、今天正好到”的日期；当天临时计划、“昨天/上周/已经”
        这类已完成的话一律不算，避免把日常安排当成到期提醒。
        """
        now = _parse_ts(query_time)
        if now is None or max_results <= 0:
            return []
        today = now.date()
        count = self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        key = (today.isoformat(), count)
        if self._due_cache is not None and self._due_cache[0] == key:
            candidates = self._due_cache[1]
        else:
            candidates = self._due_candidates(today)
            self._due_cache = (key, candidates)
        out: list[dict[str, Any]] = []
        for item in candidates[:max_results]:
            when = "今天" if item["days"] == 0 else f"{item['days']}天前"
            body = " ".join(item["text"].split())
            out.append({
                "id": "due:" + item["ts"],
                "content": (f"【今天到期｜{today.isoformat()} "
                            f"周{_WEEKDAY_CN[today.weekday()]}】{when}（{item['ts'][:10]}）说过："
                            + (body[:180] + "…" if len(body) > 180 else body)),
                "timestamp": item["ts"], "tier": "due",
                "score": 1.0, "due_date": today.isoformat(),
                "text": body, "days": item["days"],
            })
        return out

    async def search_sources(self, query: str, max_results: int = 5, *,
                             query_time: Any = None, max_chars: int = 260) -> list[dict[str, Any]]:
        """原始轮召回：在 sources 里按 BM25 找最相关的用户原话。

        时间片是摘要，隐式题要的那句话常常没被摘要保留；这一路直接检索原文轮，
        补上“摘要没写、原文里在”的证据。词项计数按 persona 库缓存，只在新增轮次时重建。
        """
        if max_results <= 0 or not query.strip():
            return []
        query_terms = [term for term in segment(query, max_terms=40) if len(term) >= 2]
        if not query_terms:
            return []
        docs, idf, avgdl = self._source_index()
        if not docs:
            return []
        now = _parse_ts(query_time)
        visible = [(row, counts) for row, counts in docs
                   if now is None or (_parse_ts(row["timestamp"]) or now) <= now]
        if not visible:
            return []
        weights = self._query_weights(query, query_terms, idf, len(docs))
        scored = self._bm25(visible, weights, avgdl)
        scored = [(row, score) for row, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row, score in scored[:max_results]:
            body = " ".join(str(row["content"]).split())
            if not body or body in seen:
                continue
            seen.add(body)
            out.append({
                "id": "sources:" + str(row["id"]), "content": body[:max_chars],
                "timestamp": row["timestamp"], "tier": "raw", "score": round(score, 4),
            })
        return out

    def _source_index(self) -> tuple[list[tuple[sqlite3.Row, Counter[str]]], dict[str, float], float]:
        """sources 的 BM25 索引；按行数缓存，新增轮次后重建。"""
        count = self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        if self._source_docs is not None and self._source_docs[0] == count:
            return self._source_docs[1], self._source_docs[2], self._source_docs[3]
        rows = list(self._conn.execute(
            "SELECT id,content,timestamp FROM sources WHERE role!='assistant' ORDER BY timestamp"))
        docs = [(row, Counter(term for term in segment(str(row["content"]), max_terms=200)
                              if len(term) >= 2)) for row in rows]
        frequencies: Counter[str] = Counter()
        for _row, counts in docs:
            frequencies.update(counts.keys())
        idf = self._idf(frequencies, len(docs)) if docs else {}
        avgdl = (sum(sum(counts.values()) for _row, counts in docs) / len(docs)) if docs else 0.0
        self._source_docs = (count, docs, idf, avgdl)
        return docs, idf, avgdl

    async def search_entity_timeline(self, query: str, max_results: int = 2, *,
                                     query_time: Any = None, days: int = 120,
                                     max_chars: int = 600) -> list[dict[str, Any]]:
        """实体时间线：问题里点到的人/物，取最近提到它的时间片原文。

        针对“最近它怎么样/是不是太胖了”这类没带具体数字的隐式问题，
        用实体名直接锚定最近的状态描述，不依赖查询词与摘要措辞重合。
        """
        if max_results <= 0:
            return []
        proper, _ = query_features(query)
        names = [name for name in proper if len(name) >= 2][:2]
        if not names:
            return []
        now = _parse_ts(query_time)
        candidates: list[tuple[datetime, dict[str, Any]]] = []
        for row in self._conn.execute("SELECT id,start_ts,end_ts,summary FROM segments"):
            end = _parse_ts(row["end_ts"])
            if end is None or (now is not None and end > now):
                continue
            if now is not None and (now - end).total_seconds() > days * 86400:
                continue
            if not any(name in row["summary"] for name in names):
                continue
            candidates.append((end, dict(row)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[dict[str, Any]] = []
        for end, row in candidates:
            if len(out) >= max_results:
                break
            text = self._event_window_text(row, row["end_ts"], max_chars)
            if not text:
                continue
            out.append({
                "id": "entity:" + str(row["id"]), "content": text,
                "timestamp": row["end_ts"], "tier": "entity", "score": 1.0,
                "entity": ",".join(names),
            })
        return out

    async def search_entity_state(self, query: str, query_time: Any, *,
                                  days: int = 14, max_chars: int = 250) -> list[dict[str, Any]]:
        """隐式题补位：问题只点到实体＋状态维度时，取最近一条同维度原文（1 条）。

        与时间线通道不同：先要求实体唯一，再要求候选原文命中同一状态维度
        （体重/喂养/钱/健康/工作/日期），避免“提到汤团”的闲聊压过真正的体重证据。
        """
        proper, _ = query_features(query)
        names = [name for name in proper if len(name) >= 2][:1]
        if not names:
            return []
        keys = [key for _name, group in _ENTITY_DIMENSIONS if any(k in query for k in group)
                for key in group]
        if not keys:
            return []
        now = _parse_ts(query_time)
        hits: list[tuple[datetime, dict[str, Any], str]] = []
        for row in self._conn.execute("SELECT id,start_ts,end_ts,summary FROM segments"):
            end = _parse_ts(row["end_ts"])
            if end is None or (now is not None and end > now):
                continue
            if now is not None and (now - end).total_seconds() > days * 86400:
                continue
            if names[0] not in row["summary"]:
                continue
            text = self._event_window_text(dict(row), row["end_ts"], 900)
            if not text or not any(key in text for key in keys):
                continue
            hits.append((end, dict(row), text))
        if not hits:
            return []
        hits.sort(key=lambda item: item[0], reverse=True)
        end, row, text = hits[0]
        return [{
            "id": "state:" + str(row["id"]), "content": text[:max_chars],
            "timestamp": row["end_ts"], "tier": "state", "score": 1.0,
            "entity": names[0],
        }]

    def _fuse_vectors(self, query: str,
                      scored: list[tuple[sqlite3.Row, float]]) -> list[tuple[sqlite3.Row, float]]:
        """词法为主、向量只做加分：bge 对长摘要的余弦区分度低，直接融合会把词法强命中挤下去。

        只有明显高于背景相似度（约 0.6）的片段才拿加分，用于救“问法与归档转述错位”的段。
        """
        if not self.segment_vectors or not scored:
            return scored
        sims = self._vector_scores(query)
        if not sims:
            return scored
        peak = max(score for _, score in scored) or 1.0
        fused = []
        for row, score in scored:
            cosine = sims.get(row["id"], 0.0)
            bonus = max(0.0, cosine - _VEC_FLOOR) * _VEC_BONUS
            if score <= 0 and bonus <= 0:
                continue
            fused.append((row, score / peak + bonus))
        return fused

    def _segment_counts(self, segment_id: str, summary: str) -> Counter[str]:
        key = "seg:" + segment_id
        counts = self._segment_terms.get(key)
        if counts is None:
            counts = Counter(segment(summary, max_terms=400))
            self._segment_terms[key] = counts
        return counts

    @staticmethod
    def _idf(frequencies: Counter[str], total: int) -> dict[str, float]:
        return {term: log(1 + (total - freq + 0.5) / (freq + 0.5))
                for term, freq in frequencies.items()}

    def _query_weights(self, query: str, query_terms: list[str], idf: dict[str, float],
                       total: int) -> dict[str, float]:
        """原查询词按 idf 加权；数字/日期与专名再乘系数；可选同义扩展降权加入。"""
        unseen = log(1 + (total + 0.5) / 0.5)
        proper, numerals = query_features(query)
        proper_terms = set(proper)
        numeral_terms = set(numerals)
        weights: dict[str, float] = {}
        for term in query_terms:
            weight = idf.get(term, unseen)
            if (term in numeral_terms or any(char.isdigit() for char in term)
                    or any(char in CN_NUMERAL_CHARS for char in term)):
                weight *= _NUMERAL_WEIGHT
            if term in proper_terms:
                weight *= _PROPER_WEIGHT
            weights[term] = max(weights.get(term, 0.0), weight)
        if not self.segment_synonyms:
            return weights
        base = len(weights)
        for term in query_terms:
            if len(weights) >= base + 20:
                break
            if term in proper_terms or term in numeral_terms:
                continue
            for word in synonyms(term):
                if word not in weights:
                    weights[word] = _SYNONYM_WEIGHT * idf.get(term, unseen)
        return weights

    @staticmethod
    def _bm25(docs: list[tuple[sqlite3.Row, Counter[str]]],
              weights: dict[str, float], avgdl: float) -> list[tuple[sqlite3.Row, float]]:
        scored = []
        for row, counts in docs:
            length = sum(counts.values()) or 1
            score = 0.0
            for term, weight in weights.items():
                tf = counts.get(term)
                if not tf:
                    continue
                score += (weight * tf * (_BM25_K1 + 1)
                          / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / (avgdl or 1))))
            scored.append((row, score))
        return scored

    @staticmethod
    def _lexical_overlap(docs: list[tuple[sqlite3.Row, Counter[str]]],
                         weights: dict[str, float]) -> list[tuple[sqlite3.Row, float]]:
        """旧版重合比例打分，仅用于关闭 BM25 时做对照。"""
        total = max(1, len(weights))
        return [(row, len(weights.keys() & counts.keys()) / total) for row, counts in docs]

    def _rm3_weights(self, scored: list[tuple[sqlite3.Row, float]],
                     weights: dict[str, float], idf: dict[str, float],
                     counts_by_id: dict[str, Counter[str]]) -> dict[str, float]:
        """用首轮前 5 个片段的词分布扩写查询；扩展词数固定，避免噪声挤占。"""
        top = sorted(((row, score) for row, score in scored if score > 0),
                     key=lambda item: item[1], reverse=True)[:5]
        if not top:
            return weights
        total = sum(score for _, score in top) or 1.0
        feedback: Counter[str] = Counter()
        for row, score in top:
            counts = counts_by_id[row["id"]]
            length = sum(counts.values()) or 1
            for term, tf in counts.items():
                if term in weights or (len(term) < 2 and not term.isascii()):
                    continue
                feedback[term] += score / total * tf / length
        if not feedback:
            return weights
        expanded = dict(weights)
        peak = max(feedback.values())
        for term, probability in feedback.most_common(10):
            relative = probability / peak
            if relative < 0.2:
                break
            expanded[term] = _RM3_WEIGHT * idf.get(term, 0.0) * relative
        return expanded

    async def _json(self, provider: Any, prompt: str, phase: str) -> dict:
        self.stats[phase + "_calls"] += 1
        response = await provider.generate(
            [LLMMessage(role="user", content=prompt)], temperature=0, max_tokens=1024,
        )
        text = message_content_to_text(response.message.content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result

    async def extract_pending(self, provider: Any, *, force: bool = False) -> None:
        if provider is None:
            return
        async with self._extract_lock:
            async def extract_batch(rows: list[dict[str, Any]]) -> dict:
                context = [dict(row) for row in self._conn.execute(
                    "SELECT id,role,content,timestamp FROM sources WHERE rowid < "
                    "(SELECT rowid FROM sources WHERE id=?) ORDER BY rowid DESC LIMIT 2",
                    (rows[0]["id"],),
                )][::-1]
                return await self._json(provider,
                        '从按时间排列的对话提取可检索的原子事实，只输出JSON {"facts":[{'
                        '"source_id":"原消息id","quote":"该消息中逐字原文片段",'
                        '"subject":"主体","predicate":"属性或事件","value":"值",'
                        '"time":"事实发生时间，未知留空","condition":"条件，未知留空",'
                        '"status":"asserted|negated|planned|hypothetical|question|quoted"}]}。'
                        '允许facts为空。禁止补全未知值。结合上下文解析代词，区分用户和助手，'
                        '助手建议不得当作用户经历。保留否定、计划、改口与条件。每条必须能由'
                        'source_id对应原文quote支持；不要把问题或假设当成已发生事实。'
                        '最多提取4条事实，输出最多1024 token，优先保留具体实体、数值、时间、关系。\n'
                        'context仅用于理解指代，只提取messages中的事实，不重复提取context。\n'
                        + json.dumps({"context": context, "messages": rows}, ensure_ascii=False), "extract")

            while True:
                pending = [dict(row) for row in self._conn.execute(
                    "SELECT * FROM sources WHERE extracted=0 ORDER BY rowid LIMIT ?",
                    (self.batch_messages * 8,),
                )]
                if not force:
                    pending = pending[:len(pending) // self.batch_messages * self.batch_messages]
                if not pending:
                    return
                batches = [pending[i:i + self.batch_messages]
                           for i in range(0, len(pending), self.batch_messages)]
                results = await asyncio.gather(
                    *(extract_batch(rows) for rows in batches), return_exceptions=True,
                )
                failed = False
                for rows, data in zip(batches, results):
                    try:
                        if isinstance(data, BaseException):
                            raise data
                        if not isinstance(data.get("facts"), list):
                            raise ValueError("Missing facts list")
                        sources = {row["id"]: row for row in rows}
                        for fact in data["facts"]:
                            if not isinstance(fact, dict):
                                continue
                            source_id = fact.get("source_id")
                            source = sources.get(source_id) if isinstance(source_id, str) else None
                            quote = fact.get("quote")
                            if (not source or not isinstance(quote, str) or not quote.strip()
                                    or quote not in source["content"]
                                    or not isinstance(fact.get("status"), str)
                                    or fact["status"] not in {
                                        "asserted", "negated", "planned", "hypothetical", "question", "quoted"}
                                    or not all(isinstance(fact.get(k), str) for k in
                                               ("subject", "predicate", "value", "time", "condition"))):
                                self.stats["rejected_facts"] += 1
                                continue
                            encoded = json.dumps(fact, ensure_ascii=False, sort_keys=True)
                            fid = hashlib.sha256(encoded.encode()).hexdigest()
                            self._conn.execute("INSERT OR IGNORE INTO facts VALUES (?,?,?)",
                                               (fid, source["id"], encoded))
                        self._conn.executemany("UPDATE sources SET extracted=1 WHERE id=?",
                                               [(row["id"],) for row in rows])
                        self._conn.commit()
                    except Exception:
                        self._conn.rollback()
                        self.stats["errors"] += 1
                        logger.exception("tier_facts extraction failed; original sources retained")
                        failed = True
                if failed:
                    return

    async def search(self, query: str, max_results: int = 5, provider: Any = None,
                     query_time: Any = None) -> list[dict[str, Any]]:
        if max_results <= 0 or not query.strip():
            return []
        needs: dict[str, Any] = {}
        if provider is not None:
            try:
                needs = await self._json(provider,
                    '分析查询所需的证据，只输出JSON {"entities":[],"attributes":[],"constraints":[],"queries":[]}。'
                    '保留人名、数字、时间、否定和条件；不得猜测答案或代词的未知指代。queries最多3个短检索词。\n'
                    f'查询时间：{query_time}\n问题：{query}', "query")
            except Exception:
                self.stats["errors"] += 1
                logger.exception("tier_facts query planning failed")
        search_terms = [query]
        for key in ("entities", "attributes", "constraints", "queries"):
            values = needs.get(key, [])
            if isinstance(values, list):
                search_terms.extend(value for value in values if isinstance(value, str))
        terms = set(segment(" ".join(search_terms), max_terms=60))
        ranked = []
        for row in self._conn.execute(
                "SELECT f.id,f.data,s.id AS source_id,s.role,s.content,s.timestamp "
                "FROM facts f JOIN sources s ON f.source_id=s.id"):
            if query_time is not None:
                try:
                    if datetime.fromisoformat(row["timestamp"]) > datetime.fromisoformat(str(query_time)):
                        continue
                except (ValueError, TypeError):
                    continue
            fact = json.loads(row["data"])
            text = " ".join(fact[k] for k in ("subject", "predicate", "value", "condition", "time"))
            if row["id"] not in self._term_cache:
                self._term_cache[row["id"]] = set(segment(text + " " + row["content"], max_terms=100))
            overlap = terms & self._term_cache[row["id"]]
            if not overlap:
                continue
            score = len(overlap) / max(1, len(terms))
            ranked.append({"id": "facts:" + row["id"], "evidence_id": row["source_id"],
                           "content": row["content"], "source": row["role"],
                           "timestamp": row["timestamp"], "fact": fact,
                           "tier": "facts", "score": score})
        ranked.sort(key=lambda row: (row["score"], row["timestamp"]), reverse=True)
        candidates = ranked[:24]
        if provider is not None and candidates:
            try:
                verified = await self._json(provider,
                    '按问题需要选择原文证据，只输出JSON {"selected":[候选序号整数]}。'
                    '最多选择' + str(max_results) + '条。检查主体、属性、时间、条件、否定和改口；'
                    '不要把助手建议、提问、假设当作已发生事实。同属性冲突保留时间与条件差异，'
                    '不能仅凭摘要相似判断真实。只能选择有原文支持的候选，缺失则空列表。\n'
                    + json.dumps({"question": query, "now": str(query_time), "needs": needs,
                                  "candidates": candidates}, ensure_ascii=False), "verify")
                selected = verified.get("selected")
                if not isinstance(selected, list):
                    raise ValueError("Missing evidence selection")
                candidates = [candidates[i] for i in dict.fromkeys(
                    i for i in selected if type(i) is int and 0 <= i < len(candidates))]
            except Exception:
                self.stats["errors"] += 1
                logger.exception("tier_facts evidence selection failed")
                return []
        out, seen = [], set()
        for row in candidates:
            if row["evidence_id"] in seen:
                continue
            seen.add(row["evidence_id"])
            row["score"] = max(0.5, row["score"])
            out.append(row)
            if len(out) >= max_results:
                break
        return out

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.stats,
            "sources": self._conn.execute("SELECT count(*) FROM sources").fetchone()[0],
            "facts": self._conn.execute("SELECT count(*) FROM facts").fetchone()[0],
            "segments": self._conn.execute("SELECT count(*) FROM segments").fetchone()[0],
            "pending_sources": self._conn.execute(
                "SELECT count(*) FROM sources WHERE extracted=0").fetchone()[0],
        }

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._term_cache.clear()
        self._segment_terms.clear()
