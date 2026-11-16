"""检索改造的最小回归测试（unittest，不依赖网络/LLM）。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from echo.memory.service import MemoryService
from echo.memory.settings import MemorySettings
from echo.memory.tier1_hierarchical import HierarchicalMemory
from echo.memory.tier2_beam import BeamMemory
from echo.memory.tier3_core import CoreMemory
from echo.memory.tier_facts import FactsMemory
from echo.models import LLMMessage


class Tier1Test(unittest.IsolatedAsyncioTestCase):
    async def test_long_chinese_query_has_score_and_recency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(
                MemorySettings(working_dir=Path(tmp), idle_consolidation_enabled=False),
                lightweight=True,
            )
            await service.ensure_started()
            await service.remember_turn(
                [
                    LLMMessage(role="user", content="我欠你三件夜班见闻，第一件是保安老陈帮我留门。"),
                    LLMMessage(role="assistant", content="记下了，三件夜班见闻。"),
                ],
                event_time="2026-06-10T22:50:00+08:00",
            )
            await service.remember_turn(
                [
                    LLMMessage(role="user", content="今天吃了牛肉面。"),
                    LLMMessage(role="assistant", content="好吃吗？"),
                ],
                event_time="2026-07-01T12:00:00+08:00",
            )
            rows = await service.hierarchical.search(
                "我上次跟你说我欠你三件夜班见闻是什么",
                10,
                0.1,
                query_time="2026-07-05T12:00:00+08:00",
            )
            self.assertTrue(rows)
            self.assertEqual(rows[0]["timestamp"], "2026-06-10T22:50:00+08:00")
            self.assertIn("score", rows[0])
            self.assertGreater(rows[0]["score"], 0)
            await service.close()

    async def test_backfill_seg_for_legacy_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "hierarchical.db"
            conn = sqlite3.connect(db)
            conn.execute(
                """CREATE TABLE hierarchical_entries (
                    entry_id TEXT PRIMARY KEY, l1_summary TEXT NOT NULL,
                    l2_detail TEXT, l3_full TEXT, l4_extended TEXT, l5_raw TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    memory_type TEXT NOT NULL DEFAULT 'conversation',
                    scope TEXT NOT NULL DEFAULT 'session',
                    source TEXT NOT NULL DEFAULT 'user',
                    timestamp TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            conn.execute(
                "INSERT INTO hierarchical_entries(entry_id,l1_summary,l2_detail,l3_full,"
                "importance,timestamp) VALUES (?,?,?,?,?,?)",
                ("legacy", "夜班见闻", "夜班见闻详情", "夜班见闻全文", 0.5, "2026-06-01T00:00:00+08:00"),
            )
            conn.commit()
            conn.close()
            memory = HierarchicalMemory(db)
            await memory.initialize()
            check = sqlite3.connect(db)
            self.assertEqual(check.execute("SELECT count(*) FROM hierarchical_seg").fetchone()[0], 1)
            check.close()
            await memory.close()


class Tier3Test(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_coverage_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = CoreMemory(Path(tmp) / "core.db")
            await core.initialize()
            await core.add_fact("用户偏好称呼: 罗姐就行", category="user_preference", weight=0.9)
            await core.add_fact("用户喜欢晚上十点后聊夜班见闻", category="user_preference", weight=0.7)
            rows = await core.search("我该怎么称呼你比较合适")
            self.assertTrue(rows)
            self.assertEqual(rows[0]["content"], "用户偏好称呼: 罗姐就行")
            self.assertGreater(rows[0]["score"], 0.05)
            await core.close()


class CoreExtractionTest(unittest.IsolatedAsyncioTestCase):
    async def test_stable_facts_updates_and_manual_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(MemorySettings(working_dir=Path(tmp)), lightweight=True)
            service.core = CoreMemory(Path(tmp) / "core.db")
            await service.core.initialize()
            try:
                manual = await service.core.add_fact("用户生日: 4月17日", category="identity")
                await service._extract_user_preferences([
                    LLMMessage(role="assistant", content="我的职业是医生"),
                    LLMMessage(role="user", content="叫我小王。我的生日是4月17日。我常住在杭州。我的职业是教师。我喜欢喝茶。请记住本项目必须使用venv。"),
                    LLMMessage(role="user", content="不要叫我小王。叫我宋老师。我的职业是护士。我不喜欢喝茶。"),
                ])
                facts = await service.core.list_facts(limit=100)
                contents = {row["content"] for row in facts}
                self.assertIn("用户偏好称呼: 宋老师", contents)
                self.assertIn("用户不希望被称呼为: 小王", contents)
                self.assertNotIn("用户偏好称呼: 小王", contents)
                self.assertIn("用户职业: 护士", contents)
                self.assertNotIn("用户职业: 教师", contents)
                self.assertNotIn("用户职业: 医生", contents)
                self.assertIn("用户不喜欢: 喝茶", contents)
                self.assertNotIn("用户喜欢: 喝茶", contents)
                self.assertIn("项目约定: 本项目必须使用venv", contents)
                self.assertIsNotNone(await service.core.get_fact(manual))
                count = len(facts)
                await service._extract_user_preferences([LLMMessage(role="user", content="我的职业是护士。")])
                self.assertEqual(len(await service.core.list_facts(limit=100)), count)
                await service.core.close()
                await service.core.initialize()
                self.assertIn("用户职业: 护士", {f["content"] for f in await service.core.list_facts(limit=100)})
            finally:
                await service.core.close()

    def test_questions_quotes_hypotheticals_and_temporary_are_not_core(self) -> None:
        for text in ["如果叫我小王就好了", "你以后叫我小王好吗？", "我叫你把门关上",
                     "我的职业是教师？", "我喜欢喝茶吗?", "她说我喜欢喝茶",
                     "我今天喜欢喝茶", "我暂时常住在杭州", "比如我喜欢喝茶",
                     "我以前喜欢喝茶", "我不再喜欢喝茶", "‘我喜欢喝茶’", "如果下雨，我喜欢在家看书"]:
            with self.subTest(text=text):
                self.assertEqual(MemoryService._iter_core_preferences(text), [])


class FusionTest(unittest.IsolatedAsyncioTestCase):
    async def test_normalized_weighted_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(
                MemorySettings(working_dir=Path(tmp), idle_consolidation_enabled=False),
                lightweight=True,
            )
            tier1 = [{"id": "a", "content": "a", "score": 0.8, "timestamp": "2026-06-01T00:00:00+08:00"}]
            tier2 = [{"id": "b", "content": "b", "score": 0.4, "timestamp": "2026-06-02T00:00:00+08:00"}]
            merged = service._merge_tiers([], tier1, tier2, [], max_results=2, min_score=0.1)
            self.assertEqual(merged[0]["id"], "a")
            self.assertAlmostEqual(merged[0]["score"], 0.8)
            self.assertAlmostEqual(merged[1]["score"], 0.6667, places=3)
            self.assertEqual(merged[0]["raw_score"], 0.8)


class BeamEventTimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_store_and_search_use_event_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            beam = BeamMemory(Path(tmp) / "beam", backend="mnemosyne")
            await beam.initialize()
            await beam.store(
                "我老公周二周四上夜班，白天补觉怕吵。",
                source="user",
                importance=0.6,
                event_time="2026-05-01T12:00:00+08:00",
            )
            rows = await beam.search(
                "夜班",
                top_k=5,
                min_score=0.1,
                query_time="2026-05-10T12:00:00+08:00",
            )
            self.assertTrue(rows)
            self.assertEqual(rows[0]["timestamp"], "2026-05-01T12:00:00+08:00")
            await asyncio.sleep(0)
            await beam.close()


class EvidenceFusionTest(unittest.IsolatedAsyncioTestCase):
    async def test_original_condition_survives_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = HierarchicalMemory(Path(tmp) / "h.db")
            await memory.initialize()
            content = "项目会议安排。" + "背景说明。" * 60 + "如果审批通过才在周五开会，目前尚未确定。"
            try:
                await memory.store(content)
                rows = await memory.search("项目会议安排")
                self.assertEqual(rows[0]["content"], content)
                self.assertNotIn("尚未确定", rows[0]["summary"])
                self.assertEqual(rows[0]["evidence_id"], rows[0]["id"])
            finally:
                await memory.close()

    async def test_fusion_deduplicates_ids_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(MemorySettings(working_dir=Path(tmp)), lightweight=True)
            a = {"id": "a", "content": "审批通过才开会", "source": "user", "score": 0.7}
            b = {**a, "id": "b", "score": 0.3}
            other = {"id": "c", "content": "会议取消", "source": "user", "score": 0.9}
            rows = service._merge_tiers([], [a, a], [other, b], [], 10)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["content"], other["content"])
            self.assertEqual(sum(row["id"] == "a" for row in rows), 1)
            self.assertEqual(a["score"], 0.7)
            self.assertEqual(service._merge_tiers([], [a], [], [], 0), [])
            self.assertEqual(service._merge_tiers([], [{**a, "score": 0.01}], [], [], 10), [])

    async def test_namespace_refill_remains_filtered(self):
        from unittest.mock import AsyncMock
        from types import SimpleNamespace
        beam = BeamMemory(Path(tempfile.gettempdir()) / "unused_mock_beam", backend="mnemosyne")
        beam._backend = "mnemosyne"
        beam._vector_ready = True
        beam._ensure_ready = AsyncMock()
        budgets = []
        def recall(**kwargs):
            budgets.append(kwargs["top_k"])
            return [{"id": str(i), "score": 0.8, "content": str(i), "source": "agent"} for i in range(min(kwargs["top_k"], 5))]
        beam._engine = SimpleNamespace(recall=recall)
        def lookup(ids):
            return {i: "memory" if i == "4" else "log" for i in ids}
        beam._lookup_namespaces = lookup
        async def run(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        beam._run = run
        found = await beam.search("审批", top_k=1)
        self.assertEqual(budgets, [2, 4, 8])
        self.assertEqual([r["id"] for r in found], ["4"])
        self.assertTrue(all(r["namespace"] == "memory" for r in found))


class FactsMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = FactsMemory(Path(self.tmp.name) / "facts.db", batch_messages=2)
        await self.memory.initialize()
        self.addAsyncCleanup(self.memory.close)

    @staticmethod
    def provider(*responses):
        replies = [
            SimpleNamespace(message=LLMMessage(role="assistant", content=json.dumps(reply, ensure_ascii=False)))
            if isinstance(reply, dict) else reply
            for reply in responses
        ]
        return SimpleNamespace(generate=AsyncMock(side_effect=replies))

    @staticmethod
    def source(source_id="source-1", content="如果审批通过，周五在杭州开会。",
               timestamp="2026-06-01T12:00:00+08:00"):
        return {"id": source_id, "role": "user", "content": content, "timestamp": timestamp}

    @staticmethod
    def fact(source_id="source-1", **overrides):
        return {"source_id": source_id, "quote": "如果审批通过，周五在杭州开会。",
                "subject": "项目", "predicate": "会议地点", "value": "杭州",
                "time": "周五", "condition": "审批通过", "status": "hypothetical", **overrides}

    async def test_rejects_fabricated_source_and_nonverbatim_quote(self) -> None:
        await self.memory.add_sources([self.source()])
        valid = self.fact()
        provider = self.provider({"facts": [
            self.fact(source_id="unknown"),
            self.fact(quote="审批已通过，周五在杭州开会。"),
            self.fact(quote=""),
            valid,
        ]})
        await self.memory.extract_pending(provider, force=True)
        rows = await self.memory.search("杭州会议")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fact"], valid)
        self.assertEqual(rows[0]["evidence_id"], "source-1")
        self.assertEqual(rows[0]["content"], self.source()["content"])
        self.assertEqual(self.memory.stats["rejected_facts"], 3)
        self.assertEqual(rows[0]["fact"]["status"], "hypothetical")
        self.assertEqual(rows[0]["fact"]["condition"], "审批通过")

    async def test_failed_extraction_retains_sources_for_retry_after_reopen(self) -> None:
        original = self.source()
        await self.memory.add_sources([original])
        failed = self.provider(RuntimeError("fake provider unavailable"))
        with self.assertLogs("echo.memory.tier_facts", level="ERROR"):
            await self.memory.extract_pending(failed, force=True)
        self.assertEqual(await self.memory.search("杭州"), [])
        await self.memory.close()
        await self.memory.initialize()
        retry = self.provider({"facts": [self.fact()]})
        await self.memory.extract_pending(retry, force=True)
        prompt = retry.generate.call_args.args[0][0].content
        self.assertIn(original["content"], prompt)
        self.assertIn(original["id"], prompt)
        rows = await self.memory.search("杭州")
        self.assertEqual([row["evidence_id"] for row in rows], [original["id"]])
        await self.memory.extract_pending(retry, force=True)
        self.assertEqual(retry.generate.await_count, 1)
        await self.memory.close()
        await self.memory.initialize()
        self.assertEqual((await self.memory.search("杭州"))[0]["content"], original["content"])

    async def test_query_time_filters_future_sources_including_timezone_boundary(self) -> None:
        sources = [
            self.source("past", timestamp="2026-06-01T11:59:59+08:00"),
            self.source("at-cutoff", timestamp="2026-06-01T04:00:00+00:00"),
            self.source("future", timestamp="2026-06-01T12:00:01+08:00"),
        ]
        await self.memory.add_sources(sources)
        await self.memory.extract_pending(self.provider(
            {"facts": [self.fact("past"), self.fact("at-cutoff")]},
            {"facts": [self.fact("future")]},
        ), force=True)
        rows = await self.memory.search("杭州", query_time="2026-06-01T12:00:00+08:00")
        self.assertEqual({row["evidence_id"] for row in rows}, {"past", "at-cutoff"})
        self.assertEqual(len(await self.memory.search("杭州")), 3)

    async def test_selected_evidence_deduplicates_source_and_rejects_invalid_indices(self) -> None:
        await self.memory.add_sources([self.source(), self.source("source-2")])
        await self.memory.extract_pending(self.provider({"facts": [
            self.fact(), self.fact(predicate="会议条件", value="审批通过"), self.fact("source-2"),
        ]}))
        provider = self.provider(
            {"entities": ["杭州"], "queries": ["会议"]},
            {"selected": [2, 2, 0, 1, True, -1, 99, "0"]},
        )
        rows = await self.memory.search("杭州会议", provider=provider)
        self.assertEqual([row["evidence_id"] for row in rows], ["source-2", "source-1"])
        self.assertTrue(all(row["content"] == self.source()["content"] for row in rows))
        self.assertEqual(provider.generate.await_count, 2)
        empty = self.provider({}, {"selected": []})
        self.assertEqual(await self.memory.search("杭州会议", provider=empty), [])

    async def test_malformed_verifier_selection_returns_no_unverified_evidence(self) -> None:
        await self.memory.add_sources([self.source()])
        await self.memory.extract_pending(self.provider({"facts": [self.fact()]}), force=True)
        malformed_json = SimpleNamespace(message=LLMMessage(role="assistant", content="not JSON"))
        for selection in ({}, {"selected": "0"}, {"selected": None}, malformed_json):
            with self.subTest(selection=selection):
                provider = self.provider({}, selection)
                with self.assertLogs("echo.memory.tier_facts", level="ERROR"):
                    rows = await self.memory.search("杭州会议", provider=provider)
                self.assertEqual(rows, [])
                self.assertEqual(provider.generate.await_count, 2)


class FactsServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_lightweight_has_no_facts_database_or_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(working_dir=Path(tmp), facts_enabled=True,
                                      idle_consolidation_enabled=False)
            service = MemoryService(settings, lightweight=True)
            provider = FactsMemoryTest.provider()
            service.facts_provider = provider
            try:
                await service.remember_turn([LLMMessage(role="user", content="如果审批通过，周五开会。")])
                await service.flush_background()
                self.assertIsNone(service.facts)
                self.assertFalse((Path(tmp) / settings.facts_db_name).exists())
                for limit in (0, -1):
                    self.assertEqual(await service.agent_search("审批", max_results=limit), [])
                provider.generate.assert_not_awaited()
            finally:
                await service.close()

    async def test_sources_reuse_tier1_ids_roles_and_event_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(MemorySettings(working_dir=Path(tmp)), lightweight=True)
            # 使用真实 Tier1 与事实数据库，仅跳过无关的 BEAM/Core 后台写入。
            service.facts = FactsMemory(Path(tmp) / "facts.db")
            await service.facts.initialize()
            stamp = "2026-06-01T12:00:00+08:00"
            messages = [LLMMessage(role="user", content="我周五去杭州。"),
                        LLMMessage(role="assistant", content="建议先确认审批。"),
                        LLMMessage(role="user", content="   ")]
            try:
                await service.remember_turn(messages, event_time=stamp)
                await service.flush_background()
                with closing(sqlite3.connect(service.facts.path)) as conn:
                    sources = conn.execute("SELECT id,role,content,timestamp FROM sources ORDER BY rowid").fetchall()
                with closing(sqlite3.connect(Path(tmp) / "hierarchical.db")) as conn:
                    evidence = conn.execute("SELECT entry_id,source,l5_raw,timestamp FROM hierarchical_entries ORDER BY rowid").fetchall()
                self.assertEqual(len(sources), 2)
                self.assertEqual(sources, evidence)
                self.assertEqual([(row[1], row[2], row[3]) for row in sources],
                                 [(message.role, message.content, stamp) for message in messages[:2]])
                self.assertEqual(len({row[0] for row in sources}), 2)
            finally:
                await service.close()


class FactsSegmentTest(unittest.IsolatedAsyncioTestCase):
    """时间片归档：切分、一次调用整理、按时间正序召回与兜底门控。"""

    @staticmethod
    def _provider(reply: str) -> SimpleNamespace:
        generate = AsyncMock(return_value=SimpleNamespace(
            message=SimpleNamespace(content=reply)))
        return SimpleNamespace(generate=generate)

    def _memory(self, tmp: str, **kwargs) -> FactsMemory:
        return FactsMemory(Path(tmp) / "facts.db", **kwargs)

    async def test_group_segments_by_gap_and_limits(self) -> None:
        memory = self._memory(tempfile.gettempdir(), segment_gap_seconds=900,
                              segment_max_messages=3, segment_max_chars=5000)
        rows = [{"content": "a" * 10, "timestamp": "2026-10-03T08:15:00"},
                {"content": "b" * 10, "timestamp": "2026-10-03T08:28:03"},
                {"content": "c" * 10, "timestamp": "2026-10-03T08:56:23"},
                {"content": "d" * 10, "timestamp": "2026-10-03T08:57:00"},
                {"content": "e" * 10, "timestamp": "2026-10-03T08:58:00"},
                {"content": "f" * 10, "timestamp": "2026-10-03T08:59:00"}]
        groups = memory._group_segments(rows)
        # 28 分钟间隔断开；后 4 条按条数上限 3 再断开。
        self.assertEqual([[row["content"][0] for row in group] for group in groups],
                         [["a", "b"], ["c", "d", "e"], ["f"]])

    async def test_consolidate_defers_last_group_until_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            await memory.initialize()
            provider = self._provider("记录时间 无\n1. 主人在超市买了西瓜\n2. 当天天气是晴天")
            try:
                await memory.add_sources([
                    {"id": "s1", "role": "user", "content": "今天在超市买了个西瓜",
                     "timestamp": "2026-10-03T08:15:00"},
                    {"id": "s2", "role": "assistant", "content": "挺好，天晴适合出门",
                     "timestamp": "2026-10-03T08:16:30"},
                    {"id": "s3", "role": "user", "content": "晚上再去一趟",
                     "timestamp": "2026-10-03T08:56:23"},
                ])
                await memory.consolidate_pending(provider)
                self.assertEqual(provider.generate.await_count, 1)
                self.assertIn("记录时间：2026-10-03T08:15:00~2026-10-03T08:16:30",
                              provider.generate.await_args.args[0][0].content)
                await memory.consolidate_pending(provider, force=True)
                self.assertEqual(provider.generate.await_count, 2)
                with closing(sqlite3.connect(memory.path)) as conn:
                    summaries = conn.execute(
                        "SELECT start_ts,summary FROM segments ORDER BY start_ts").fetchall()
                    pending = conn.execute(
                        "SELECT count(*) FROM sources WHERE extracted=0").fetchone()[0]
                self.assertEqual([row[0] for row in summaries],
                                 ["2026-10-03T08:15:00", "2026-10-03T08:56:23"])
                # 只保留编号条目，丢弃模型回显的时间行。
                self.assertNotIn("记录时间 无", summaries[0][1])
                self.assertIn("西瓜", summaries[0][1])
                self.assertEqual(pending, 0)
            finally:
                await memory.close()

    async def test_search_segments_orders_by_time_and_filters_future(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            await memory.initialize()
            try:
                memory._conn.executemany(
                    "INSERT INTO segments VALUES (?,?,?,?,?)",
                    [("a", "2026-10-03T08:56:23", "2026-10-03T09:30:00",
                      "1. 主人晚上又去超市买了西瓜", 6),
                     ("b", "2026-10-03T08:15:00", "2026-10-03T08:28:03",
                      "1. 主人在超市买了西瓜", 4),
                     ("c", "2026-11-01T10:00:00", "2026-11-01T10:20:00",
                      "1. 主人在超市买了苹果", 3)])
                memory._conn.commit()
                rows = await memory.search_segments(
                    "我什么时候购买过西瓜", 5, query_time="2026-10-05T00:00:00")
                self.assertEqual([row["id"] for row in rows], ["segments:b", "segments:a"])
                self.assertTrue(all(row["tier"] == "segments" for row in rows))
                self.assertTrue(all(row["timestamp"] <= "2026-10-05" for row in rows))
                self.assertIn("记录时间 2026-10-03T08:15:00", rows[0]["content"])
            finally:
                await memory.close()

    async def test_bm25_prefers_rare_term_over_common_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            await memory.initialize()
            try:
                memory._conn.executemany(
                    "INSERT INTO segments VALUES (?,?,?,?,?)",
                    [("c1", "2026-10-01T08:00:00", "2026-10-01T08:05:00",
                      "1. 今天天气不错", 2),
                     ("c2", "2026-10-01T09:00:00", "2026-10-01T09:05:00",
                      "1. 今天去了公园", 2),
                     ("c3", "2026-10-01T10:00:00", "2026-10-01T10:05:00",
                      "1. 今天在家看书", 2),
                     ("rare", "2026-10-02T08:00:00", "2026-10-02T08:05:00",
                      "1. 换了个小轴承", 2)])
                memory._conn.commit()
                # “今天”出现在 3 个片段里，“轴承”只出现 1 次；两者各命中 1 个
                # 查询词时，旧版重合比例会打平，BM25 应按 idf 选中稀有词片段。
                top = await memory.search_segments("今天的轴承", 1)
                self.assertEqual([row["id"] for row in top], ["segments:rare"])
            finally:
                await memory.close()

    async def test_synonym_expansion_recalls_paraphrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            await memory.initialize()
            try:
                memory._conn.execute(
                    "INSERT INTO segments VALUES (?,?,?,?,?)",
                    ("s1", "2026-10-03T08:00:00", "2026-10-03T08:10:00",
                     "1. 主人上周采购了两个苹果", 2))
                memory._conn.commit()
                memory.segment_synonyms = False
                self.assertEqual(
                    await memory.search_segments("去购买", 5), [])
                memory.segment_synonyms = True
                rows = await memory.search_segments("去购买", 5)
                self.assertEqual([row["id"] for row in rows], ["segments:s1"])
            finally:
                await memory.close()

    async def test_agent_search_injects_best_segment_even_with_mixed_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = MemoryService(MemorySettings(working_dir=Path(tmp)), lightweight=True)
            service.facts = FactsMemory(Path(tmp) / "facts.db")
            await service.facts.initialize()
            try:
                service.facts._conn.execute(
                    "INSERT INTO segments VALUES (?,?,?,?,?)",
                    ("s1", "2026-09-30T08:00:00", "2026-09-30T08:10:00",
                     "1. 主人上周采购了两箱苹果", 2))
                service.facts._conn.commit()

                async def mixed(query, limit, min_score, *, query_time=None):
                    return [[{"id": "m1", "content": "苹果很好吃", "source": "hierarchical",
                              "tier": "facts", "timestamp": "2026-10-01T10:00:00+08:00",
                              "score": 0.6}], [], [], []]

                service._collect_tiers = mixed
                rows = await service.agent_search("我什么时候购买过苹果", max_results=5)
                self.assertTrue(any(row["tier"] == "segments" for row in rows))
            finally:
                await service.close()


if __name__ == "__main__":
    unittest.main()
