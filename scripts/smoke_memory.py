#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回声记忆系统冒烟：写入两轮对话，再按中文问题检索一次。"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo.memory.service import MemoryService  # noqa: E402
from echo.memory.settings import MemorySettings  # noqa: E402
from echo.models import LLMMessage  # noqa: E402


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="echo-smoke-") as tmp:
        service = MemoryService(MemorySettings(
            working_dir=Path(tmp),
            beam_backend="mnemosyne",
            vector_enabled=True,
            consolidation_use_llm=False,
            idle_consolidation_enabled=False,
            facts_enabled=True,
            facts_batch_messages=2,
            facts_defer_extraction=True,
        ))
        await service.ensure_started()
        turns = [
            ("我前天把田恬的软木瑜伽砖放猫爬架旁边，被汤团抓了好几道印子，还没敢跟她说。", "啊这，那你得找机会跟田恬说清楚，别一直搁着。"),
            ("下周三是罗姐批我的补休，我打算去取眼镜。", "行，补休那天把取眼镜的事一起办了。"),
        ]
        for user_text, bot_text in turns:
            await service.remember_turn([
                LLMMessage(role="user", content=user_text),
                LLMMessage(role="assistant", content=bot_text),
            ])
        await service.flush_background()
        rows = await service.agent_search("我是不是还欠着田恬什么东西没交代？", max_results=5)
        print(f"检索到 {len(rows)} 条：")
        for row in rows:
            print(" -", str(row.get("content", ""))[:120].replace("\n", " "))
        assert rows, "agent_search 没有返回任何记忆"
        stats = await service.get_memory_stats()
        print("stats:", {k: stats.get(k) for k in ("facts", "segments", "vector_search")})
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
