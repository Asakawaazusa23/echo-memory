"""IdleConsolidationScheduler — 空闲时段自动记忆巩固调度器。

在对话间隙/长待机状态中自动触发:
1. 记忆整合 (MemoryConsolidator)
2. 进化分析 (EvolutionAnalyzer)
3. Token 消耗统计报告

设计:
- 基于"最后对话时间"检测空闲状态
- 支持可配置的空闲阈值
- 后台 asyncio 任务运行，不阻塞主路径
- 防止重复运行
- 可手动触发
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .consolidator import MemoryConsolidator
    from .evolution import EvolutionAnalyzer
    from .token_tracker import TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SchedulerConfig:
    """调度器配置。"""
    idle_threshold_seconds: int = 300       # 空闲 5 分钟后触发
    consolidation_interval_seconds: int = 1800  # 整合最小间隔 30 分钟
    evolution_interval_seconds: int = 3600     # 进化分析最小间隔 1 小时
    token_report_interval_seconds: int = 7200  # Token 报告间隔 2 小时
    enabled: bool = True
    use_llm: bool = True                       # 是否使用 LLM 做深度归纳


@dataclass
class SchedulerState:
    """调度器运行状态。"""
    last_user_activity: float = 0.0       # 最后用户活动时间戳
    last_consolidation: float = 0.0       # 上次整合时间
    last_evolution: float = 0.0           # 上次进化分析时间
    last_token_report: float = 0.0        # 上次 Token 报告时间
    total_consolidations: int = 0
    total_evolutions: int = 0
    running_task: str | None = None       # 当前正在执行的任务名


# ---------------------------------------------------------------------------
# IdleConsolidationScheduler
# ---------------------------------------------------------------------------

class IdleConsolidationScheduler:
    """空闲时自动巩固调度器。

    工作原理:
    1. 启动后台循环，每 N 秒检查一次空闲状态
    2. 当空闲时间超过阈值，按优先级依次执行:
       - 记忆整合 (最高优先级，最频繁)
       - 进化分析 (中等频率)
       - Token 统计报告 (最低频率)
    3. 每次用户活动时重置空闲计时器
    """

    def __init__(
        self,
        *,
        consolidator: MemoryConsolidator | None = None,
        evolution: EvolutionAnalyzer | None = None,
        token_tracker: TokenTracker | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._consolidator = consolidator
        self._evolution = evolution
        self._token_tracker = token_tracker
        self._config = config or SchedulerConfig()

        self._state = SchedulerState(
            last_user_activity=time.monotonic(),
        )
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._callbacks: list[Any] = []
        self._consolidation_callback: Any | None = None

    def set_consolidation_callback(self, callback: Any) -> None:
        """设置由 MemoryService 统一提供的整合入口。"""
        self._consolidation_callback = callback

    # -- 生命周期 ------------------------------------------------------------

    async def start(self) -> None:
        """启动后台调度循环。"""
        if self._running:
            return
        if not self._config.enabled:
            logger.info("IdleConsolidationScheduler disabled by config")
            return

        self._running = True
        self._state.last_user_activity = time.monotonic()
        self._task = asyncio.create_task(
            self._scheduler_loop(),
            name="novaecho_idle_consolidation",
        )
        logger.info(
            "IdleConsolidationScheduler started "
            "(idle_threshold=%ds, consolidation_interval=%ds)",
            self._config.idle_threshold_seconds,
            self._config.consolidation_interval_seconds,
        )

    async def stop(self) -> None:
        """停止调度循环。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("IdleConsolidationScheduler stopped")

    # -- 活动通知 ------------------------------------------------------------

    def notify_user_activity(self) -> None:
        """通知调度器有用户活动（重置空闲计时器）。"""
        self._state.last_user_activity = time.monotonic()

    def notify_conversation_end(self) -> None:
        """通知一轮对话结束，可考虑立即触发整合。"""
        # 不立即触发，但可以缩短空闲检测间隔
        pass

    # -- 后台循环 ------------------------------------------------------------

    async def _scheduler_loop(self) -> None:
        """主调度循环。"""
        check_interval = 30  # 每 30 秒检查一次

        while self._running:
            try:
                await asyncio.sleep(check_interval)

                if not self._is_idle():
                    continue

                now = time.monotonic()
                idle_time = now - self._state.last_user_activity

                if idle_time < self._config.idle_threshold_seconds:
                    continue

                # 空闲时间足够，按优先级执行任务
                await self._run_scheduled_tasks(now)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("IdleConsolidationScheduler error: %s", exc,
                            exc_info=True)
                await asyncio.sleep(60)  # 出错后等待更久

    async def _run_scheduled_tasks(self, now: float) -> None:
        """按优先级执行调度任务。"""

        # 1. 记忆整合 (优先级最高)
        if (self._consolidator and not self._consolidator.is_running
                and now - self._state.last_consolidation
                    >= self._config.consolidation_interval_seconds):
            await self._run_consolidation(now)

        # 2. 进化分析
        if (self._evolution
                and now - self._state.last_evolution
                    >= self._config.evolution_interval_seconds):
            await self._run_evolution_analysis(now)

        # 3. Token 报告
        if (self._token_tracker
                and now - self._state.last_token_report
                    >= self._config.token_report_interval_seconds):
            await self._run_token_report(now)

    async def _run_consolidation(self, now: float) -> None:
        """执行记忆整合。"""
        self._state.running_task = "consolidation"
        logger.info("Starting idle memory consolidation...")
        try:
            if self._consolidation_callback:
                result = await self._consolidation_callback(use_llm=self._config.use_llm)
            else:
                result = await self._consolidator.run_full_consolidation(
                    use_llm=self._config.use_llm,
                )
            self._state.last_consolidation = now
            self._state.total_consolidations += 1
            logger.info(
                "Idle consolidation done: %d clusters, %d merged, %.1fs",
                result.clusters_found, result.memories_merged,
                result.duration_ms / 1000,
            )
        except Exception as exc:
            logger.error("Consolidation failed: %s", exc, exc_info=True)
        finally:
            self._state.running_task = None

    async def _run_evolution_analysis(self, now: float) -> None:
        """执行进化分析。"""
        self._state.running_task = "evolution"
        logger.info("Starting idle evolution analysis...")
        try:
            # 收集对话历史用于分析
            # 从回调中获取最近的对话
            messages = await self._collect_recent_messages()
            if messages:
                report = await self._evolution.analyze_conversations(messages)
                self._state.last_evolution = now
                self._state.total_evolutions += 1
                logger.info(
                    "Idle evolution done: %d patterns (%d new, %d updated)",
                    report.patterns_found, report.new_patterns,
                    report.updated_patterns,
                )
        except Exception as exc:
            logger.error("Evolution analysis failed: %s", exc, exc_info=True)
        finally:
            self._state.running_task = None

    async def _run_token_report(self, now: float) -> None:
        """生成 Token 消耗报告。"""
        self._state.running_task = "token_report"
        try:
            summary = await self._token_tracker.get_session_summary()
            total = summary.get("total_tokens", 0)
            calls = summary.get("total_calls", 0)
            logger.info(
                "Token usage report: %d calls, %d total tokens "
                "(prompt: %d, completion: %d)",
                calls, total,
                summary.get("total_prompt_tokens", 0),
                summary.get("total_completion_tokens", 0),
            )
            self._state.last_token_report = now
        except Exception as exc:
            logger.error("Token report failed: %s", exc)
        finally:
            self._state.running_task = None

    # -- 对话历史收集 (通过回调) --------------------------------------------

    async def _collect_recent_messages(self) -> list[dict[str, str]]:
        """收集最近的对话历史，用于进化分析。

        通过注册的回调函数获取。
        """
        for callback in self._callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, list):
                    return result
            except Exception as exc:
                logger.debug("Message collection callback failed: %s", exc)
        return []

    def register_message_source(self, callback: Any) -> None:
        """注册一个获取最近对话的回调函数。"""
        self._callbacks.append(callback)

    # -- 状态查询 ------------------------------------------------------------

    def _is_idle(self) -> bool:
        """检查是否处于空闲状态。"""
        if not self._config.enabled:
            return False
        return time.monotonic() - self._state.last_user_activity \
            >= self._config.idle_threshold_seconds

    def get_state(self) -> dict[str, Any]:
        """获取调度器状态。"""
        now = time.monotonic()
        idle_seconds = now - self._state.last_user_activity
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "is_idle": self._is_idle(),
            "idle_seconds": round(idle_seconds),
            "idle_threshold_seconds": self._config.idle_threshold_seconds,
            "running_task": self._state.running_task,
            "total_consolidations": self._state.total_consolidations,
            "total_evolutions": self._state.total_evolutions,
            "last_consolidation_ago": round(now - self._state.last_consolidation) if self._state.last_consolidation else None,
            "last_evolution_ago": round(now - self._state.last_evolution) if self._state.last_evolution else None,
        }

    # -- 手动触发 ------------------------------------------------------------

    async def trigger_consolidation_now(self) -> Any:
        """手动触发记忆整合（立即执行，不受空闲限制）。"""
        if self._consolidator is None:
            return None
        logger.info("Manual consolidation triggered")
        if self._consolidation_callback:
            result = await self._consolidation_callback(use_llm=self._config.use_llm)
        else:
            result = await self._consolidator.run_full_consolidation(
                use_llm=self._config.use_llm,
            )
        self._state.last_consolidation = time.monotonic()
        self._state.total_consolidations += 1
        return result

    async def trigger_evolution_now(self) -> Any:
        """手动触发进化分析。"""
        if self._evolution is None:
            return None
        logger.info("Manual evolution analysis triggered")
        messages = await self._collect_recent_messages()
        if messages:
            report = await self._evolution.analyze_conversations(messages)
            self._state.last_evolution = time.monotonic()
            self._state.total_evolutions += 1
            return report
        return None
