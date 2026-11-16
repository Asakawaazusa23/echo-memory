from __future__ import annotations

import asyncio
import functools
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 命名空间常量
# ---------------------------------------------------------------------------

NAMESPACE_MEMORY = "memory"   # 对话/偏好/角色记忆 → 供 roleplay/agent_search 使用
NAMESPACE_LOG = "log"          # Agent 工具调用日志 → 供 agent 自省/进化使用


# ---------------------------------------------------------------------------
# 写入长度上限
# ---------------------------------------------------------------------------

# Mnemosyne 的 remember() 内部会无条件执行 MEMORIA 正则事实抽取
# （mnemosyne/core/beam.py 的 extract_and_store_facts），其中版本号正则
#   r'([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)'
# 的 (?:\s*[A-Z][a-zA-Z]+)* 允许零空格切分，在长串 CamelCase 文本上会
# 灾难性回溯（ReDoS）。实测：一条 47280 字符的进程列表
# （Chrome 的 --enable-features=... 类长选项名）令该正则永久自旋，
# 独占事件循环后整个 web 服务失去响应（连静态文件都超时）。
#
# 因此在唯一写入入口 store() 统一截断。2000 与 agent.py 写工具日志时的
# content[:2000] 保持一致，也与 Mnemosyne 自身 l3_full 的 2013 字符截断相当。
MAX_CONTENT_CHARS = 2000


# ---------------------------------------------------------------------------
# Mnemosyne 可选导入
# ---------------------------------------------------------------------------

def _load_mnemosyne(data_dir: Path):
    """延迟导入 Mnemosyne，并把数据目录锚定到 NovaEcho 工作区。

    必须在 import 之前设置 MNEMOSYNE_DATA_DIR / HERMES_HOME：
    mnemosyne.core.memory 在模块导入期就会执行 init_db()，
    若不预先锚定，会在 ~/.hermes 建一个与本实例无关的全局库，
    导致写入与检索落在不同数据库（历史上的"库分叉"问题）。

    返回 (Mnemosyne 类, embeddings 模块 | None)。失败时抛出原始异常。
    """
    import os

    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("HERMES_HOME", str(data_dir))

    # 向量模型必须在导入 mnemosyne 之前定下来：
    # mnemosyne.core.embeddings 在模块导入期读取 MNEMOSYNE_EMBEDDING_MODEL，
    # beam.py 同期的 EMBEDDING_DIM 决定 sqlite-vec 表的向量维度；
    # 中途换模型会造成"库 512 维 / 进程 384 维"的维度冲突而反复降级 FTS5。
    #
    # Mnemosyne 自带默认是英文的 BAAI/bge-small-en-v1.5（384 维），中文检索
    # 基本无效。项目默认改为中文小模型 bge-small-zh-v1.5：512 维、ONNX 纯 CPU
    # 推理（fastembed 内置支持，约 100MB，无需 GPU，可长期常驻内存）。
    # .env 中显式设置的 MNEMOSYNE_EMBEDDING_MODEL 优先级更高。
    os.environ.setdefault("MNEMOSYNE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    # 模型首次使用需从 HuggingFace 下载，国内直连经常超时/失败（失败会触发
    # 运行期降级）。与项目其它模型下载（TTS/vits）一致默认国内镜像站，
    # 用户显式配置的 HF_ENDPOINT 优先。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # Mnemosyne 的 working_memory 默认 TTL 是 168 小时（7 天）：
    # 每次 remember() 都会删掉超期且未被 sleep() 巩固的行。
    # NovaEcho 用自己的整合引擎（memory/consolidator.py），不调 mnemosyne 的
    # sleep()（它会联网下载 GGUF 摘要模型并把原文压成摘要），
    # 所以这里必须把 TTL 拉长，否则 Tier2 记忆 7 天后被静默清空。
    # 容量仍受 WORKING_MEMORY_MAX_ITEMS（默认 10000 条）约束，不会无限膨胀。
    os.environ.setdefault("MNEMOSYNE_WM_TTL_HOURS", "876000")  # ~100 年

    # NovaEcho 是长期陪伴记忆，不是 7 天会话缓存。Mnemosyne 原版
    # RECENCY_HALFLIFE 默认 168h，45～60 天前的事实会被衰减到接近 0，
    # 导致长间隔题只能靠 Tier1 摘要。这里默认放宽到 90 天，可用
    # MNEMOSYNE_RECENCY_HALFLIFE 显式覆盖。
    os.environ.setdefault("MNEMOSYNE_RECENCY_HALFLIFE", "2160")

    from mnemosyne.core.memory import Mnemosyne

    try:
        from mnemosyne.core import embeddings as embeddings_module
    except Exception:  # noqa: BLE001 - embedding 可选，缺失不应阻断
        embeddings_module = None
    return Mnemosyne, embeddings_module


# ---------------------------------------------------------------------------
# Tier 2: 拾忆（BeamMemory）— Mnemosyne 后端深度检索 (双命名空间)
# ---------------------------------------------------------------------------

class BeamMemory:
    """Tier 2 — 拾忆：基于 Mnemosyne 的深度语义检索

    双命名空间设计:
      namespace="memory" — 对话记忆、用户偏好、角色事实
                            供 roleplay 层 & agent_search 使用
      namespace="log"    — Agent 工具调用日志、搜索结果、代码搜索输出
                            供 agent 自省、技能总结、进化分析使用

    特征:
    - 工作记忆 (Working): 当前会话活跃上下文, 约束 50 条, TTL 72h
    - 情景记忆 (Episodic): 自动压缩的长期记忆
    - 暂存记忆 (Scratchpad): Agent 推理临时记录
    - 混合检索: 向量语义 + FTS5 关键词 + 重要性 + 时间衰减
    - 自动睡眠周期: 工作记忆 → 情景记忆 自动压缩
    """

    NAMESPACE_MEMORY = NAMESPACE_MEMORY
    NAMESPACE_LOG = NAMESPACE_LOG

    def __init__(self, data_dir: Path, backend: str = "auto") -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._backend_config = backend
        self._backend = "uninitialized"
        self._engine = None          # Mnemosyne 实例（显式绑定 db_path）
        self._vector_ready = False   # 向量语义检索是否真正可用
        self._fallback: _FallbackStore | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def initialize(self) -> None:
        """初始化 Mnemosyne BEAM 引擎。"""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return

            if self._backend_config == "fallback":
                logger.info("BeamMemory backend forced to fallback by configuration")
                self._backend = "fallback"
            else:
                try:
                    loop = asyncio.get_running_loop()
                    # Mnemosyne 内部持有单个 sqlite3 连接。多个 asyncio.to_thread
                    # 调用会并发落到不同工作线程，触发 sqlite3.InterfaceError:
                    # "bad parameter or other API misuse" 并让 Tier2 静默降级。
                    # 这里用单线程 executor：引擎构造与所有读写都在同一线程内
                    # 串行执行，连接始终只被一个线程使用。
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="novaecho-beam",
                    )
                    engine, vector_ready = await loop.run_in_executor(
                        self._executor, self._init_engine_sync,
                    )
                    self._engine = engine
                    self._backend = "mnemosyne"
                    self._vector_ready = vector_ready
                    logger.info(
                        "BeamMemory (Mnemosyne BEAM) initialized @ %s (vector=%s)",
                        self._data_dir / "beam.db",
                        "on" if self._vector_ready else "off",
                    )
                    if not self._vector_ready:
                        logger.warning(
                            "Tier2 vector search unavailable (fastembed missing or "
                            "MNEMOSYNE_NO_EMBEDDINGS set); recall degrades to keyword + "
                            "importance only. Install with: pip install fastembed"
                        )
                except Exception as exc:
                    self._reset_engine()
                    if self._backend_config == "mnemosyne":
                        # 显式要求主后端时不得静默降级：否则调用方看到的是
                        # "写入成功"但数据其实落在 FTS5 兜底库，问题被掩盖。
                        raise RuntimeError(
                            f"Mnemosyne 初始化失败（{type(exc).__name__}: {exc}）；"
                            "MEMORY_BEAM_BACKEND=mnemosyne 禁止回退到 FTS5"
                        ) from exc
                    logger.warning(
                        "Mnemosyne init failed (%s: %s); using FTS5 fallback",
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    self._engine = None
                    self._backend = "fallback"

            if self._backend == "fallback":
                self._fallback = _FallbackStore(self._data_dir)
                await self._fallback.initialize()

            self._initialized = True

    def _init_engine_sync(self) -> tuple[Any, bool]:
        """在专属线程内构造 Mnemosyne 实例（连接归属该线程）。"""
        engine_cls, embeddings_module = _load_mnemosyne(self._data_dir)
        # 显式绑定 db_path，避免退回 ~/.hermes 的全局默认库
        engine = engine_cls(db_path=self._data_dir / "beam.db")
        vector_ready = bool(
            embeddings_module is not None and embeddings_module.available()
        )
        return engine, vector_ready

    def _reset_engine(self) -> None:
        self._engine = None
        self._vector_ready = False
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """在 Mnemosyne 专属线程内串行执行同步调用。"""
        if self._executor is None:
            raise RuntimeError("Mnemosyne executor is not running")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(func, *args, **kwargs),
        )

    async def _degrade_to_fallback(self, reason: str, exc: Exception) -> None:
        """运行时热降级到 FTS5 存储（仅在 auto 模式允许）。"""
        if self._backend_config == "mnemosyne":
            # 显式主后端模式下，任何运行期异常都必须暴露给调用方，
            # 不能悄悄切到 FTS5 兜底库造成"写入成功"的假象。
            raise RuntimeError(
                f"Mnemosyne {reason} 失败（{type(exc).__name__}: {exc}）；"
                "MEMORY_BEAM_BACKEND=mnemosyne 禁止回退到 FTS5"
            ) from exc
        logger.warning("Mnemosyne %s failed (%s: %s); degrading to FTS5 fallback",
                       reason, type(exc).__name__, exc, exc_info=True)
        self._backend = "fallback"
        self._reset_engine()
        if self._fallback is None:
            self._fallback = _FallbackStore(self._data_dir)
            await self._fallback.initialize()

    # -- 写入 (记忆层) ----------------------------------------------------

    async def store(self, content: str, *,
                    source: str = "agent",
                    importance: float = 0.5,
                    scope: str = "session",
                    namespace: str = NAMESPACE_MEMORY,
                    extract_entities: bool = False,
                    metadata: dict[str, Any] | None = None,
                    event_time: Any | None = None) -> str:
        """存储一条记忆到指定命名空间。

        Parameters
        ----------
        content: 记忆内容
        source: 来源 ("agent", "roleplay", "user", "tool")
        importance: 重要性 0.0~1.0
        scope: "session" / "cross_session" / "global"
        namespace: "memory" 或 "log" — 决定搜索时是否可见
        extract_entities: 是否提取实体关系图
        """
        await self._ensure_ready()

        # 唯一写入入口统一截断，详见 MAX_CONTENT_CHARS 注释。
        # Tier1 (hierarchical) 仍保存全文（L5 raw），这里截断只影响 Tier2 的
        # 语义索引，不会丢失数据。
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS]

        meta = dict(metadata or {})
        meta["namespace"] = namespace

        if self._backend == "mnemosyne" and self._engine is not None:
            try:
                # remember() 是同步 CPU 密集调用（内含正则抽取、向量写入、
                # 图/事实抽取），必须下放线程池：一旦它卡住（如 ReDoS），
                # 在事件循环线程内直呼会冻结整个 asyncio 循环，
                # 表现为 HTTP 全部超时、job 永远拿不到终态。
                # 所有调用都走同一个单线程 executor，避免多线程并发使用
                # 同一个 sqlite3 连接。
                mem_id = await self._run(
                    self._engine.remember,
                    content=content,
                    source=source,
                    importance=importance,
                    scope=scope,
                    extract_entities=extract_entities,
                    metadata=meta,
                    timestamp=event_time,
                )
                return mem_id or ""
            except Exception as exc:
                await self._degrade_to_fallback("remember", exc)

        return await self._fallback.store(
            content, source=source, importance=importance,
            scope=scope, namespace=namespace,
        )

    async def store_messages_batch(self, items: list[dict[str, Any]], *,
                                   timestamp: str | None = None,
                                   chunk_size: int = 16) -> None:
        """高吞吐批量写入：每块只调用一次 embedding API。

        items 为 [{content, role, time}]；同一块内的条目共用一个时间戳
        （取块内最后一条），避免逐条网络调用把回放拖到数小时。
        仅用于评测/导入这类批量回放场景，不改变单条写入路径。
        """
        await self._ensure_ready()
        payloads: list[tuple[str, dict[str, Any]]] = []
        for item in items:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            payloads.append((
                str(item.get("time") or timestamp or ""),
                {
                    "content": content[:MAX_CONTENT_CHARS],
                    "source": str(item.get("role") or "user"),
                    "importance": 0.3,
                    "metadata": {"namespace": NAMESPACE_MEMORY, "scope": "session"},
                },
            ))
        if not payloads:
            return
        if self._backend == "mnemosyne" and self._engine is not None:
            for start in range(0, len(payloads), chunk_size):
                chunk = payloads[start:start + chunk_size]
                for attempt in range(3):
                    try:
                        await self._run(
                            self._engine.beam.remember_batch,
                            [payload for _, payload in chunk],
                            timestamp=chunk[-1][0] or timestamp,
                        )
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc).lower() or attempt == 2:
                            raise
                        await asyncio.sleep(1.0 + attempt)
            return
        for _, payload in payloads:
            await self._fallback.store(
                payload["content"], source=payload["source"],
                importance=payload["importance"], scope="session",
                namespace=NAMESPACE_MEMORY,
            )

    async def store_messages(self, messages: list, *,
                             scope: str = "session",
                             event_time: Any | None = None) -> None:
        """批量存储对话消息到 memory 命名空间。"""
        for msg in messages:
            content = self._extract_text(msg)
            if content.strip():
                await self.store(
                    content,
                    source=getattr(msg, "role", "user"),
                    importance=0.3,
                    scope=scope,
                    namespace=NAMESPACE_MEMORY,
                    event_time=event_time,
                )

    # -- 写入 (日志层) ----------------------------------------------------

    async def store_log(self, content: str, *,
                        source: str = "tool",
                        importance: float = 0.3,
                        scope: str = "global",
                        metadata: dict[str, Any] | None = None) -> str:
        """存储 Agent 工具执行日志到 log 命名空间。

        log 命名空间的内容不会被角色扮演层看到，
        仅供 Agent 自省和进化分析使用。
        """
        # 确保 source 以 "tool:" 开头，便于 Mnemosyne 路径通过 source 过滤
        if not source.startswith("tool:"):
            source = f"tool:{source}"
        return await self.store(
            content,
            source=source,
            importance=importance,
            scope=scope,
            namespace=NAMESPACE_LOG,
            metadata=metadata,
        )

    async def store_log_messages(self, messages: list, *,
                                 scope: str = "global") -> None:
        """批量存储工具日志到 log 命名空间。"""
        for msg in messages:
            content = self._extract_text(msg)
            if content.strip():
                await self.store_log(
                    content,
                    source=getattr(msg, "role", "tool"),
                    importance=0.3,
                    scope=scope,
                )

    # -- 读取 (记忆层检索) ------------------------------------------------

    async def search(self, query: str, top_k: int = 5, *,
                     scope: str | None = None,
                     min_score: float = 0.1,
                     namespaces: tuple[str, ...] | None = None,
                     query_time: Any | None = None,
                     weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
        """检索记忆层 (默认 namespace="memory")。

        Parameters
        ----------
        namespaces: 要搜索的命名空间列表
                    默认 (None) 只搜索 memory 命名空间
        weights: 覆盖 vec/fts/importance 权重（内部会归一化到 1.0）。
                 传纯词法或纯向量权重即可拿到单通道候选，用于独立配额召回。
        """
        await self._ensure_ready()
        ns = namespaces or (NAMESPACE_MEMORY,)

        if self._backend == "mnemosyne" and self._engine is not None:
            try:
                # 向量不可用时把 vec 权重让给关键词，避免 50% 权重空转
                if weights is not None:
                    active = dict(weights)
                elif self._vector_ready:
                    active = {"vec_weight": 0.5, "fts_weight": 0.3, "importance_weight": 0.2}
                else:
                    active = {"vec_weight": 0.0, "fts_weight": 0.75, "importance_weight": 0.25}
                for multiplier in (2, 4, 8):
                    budget = top_k * multiplier
                    results = await self._run(
                        self._engine.recall,
                        query=query,
                        top_k=budget,
                        query_time=query_time,
                        **active,
                    )
                    recalled_count = len(results or [])
                    results = [r for r in (results or [])
                               if r.get("score", 0) >= min_score]

                    # 回查真实 namespace：recall() 不带 metadata，
                    # 只靠 source 前缀会把 store(namespace="log", source="agent")
                    # 误判成 memory。查不到的（历史条目）才退回 source 推断。
                    ns_map = await self._run(
                        self._lookup_namespaces, [r.get("id", "") for r in results]
                    )

                    out = []
                    for r in results:
                        actual = ns_map.get(r.get("id", ""))
                        if actual is not None:
                            if actual not in ns:
                                continue
                        elif not self._match_namespace_by_source(
                                r.get("source", "") or "", ns):
                            continue
                        out.append({
                            "id": r.get("id", ""),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.0),
                            "source": r.get("source", ""),
                            "namespace": actual or (
                                NAMESPACE_LOG
                                if (r.get("source", "") or "").startswith("tool:")
                                else NAMESPACE_MEMORY
                            ),
                            "importance": r.get("importance", 0.5),
                            "timestamp": r.get("timestamp", ""),
                            "tier": "beam",
                        })
                    if len(out) >= top_k or recalled_count < budget:
                        break
                return out[:top_k]
            except Exception as exc:
                await self._degrade_to_fallback("recall", exc)

        return await self._fallback.search(query, top_k=top_k, namespaces=ns)

    # -- 读取 (日志层检索) ------------------------------------------------

    async def search_log(self, query: str, top_k: int = 5, *,
                         min_score: float = 0.1) -> list[dict[str, Any]]:
        """检索 Agent 工具日志层 (namespace="log")。"""
        return await self.search(
            query, top_k=top_k, min_score=min_score,
            namespaces=(NAMESPACE_LOG,),
        )

    async def get_log_summary(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近的工具日志摘要 (供 Agent 自省)。"""
        await self._ensure_ready()
        if self._fallback is not None:
            return await self._fallback.get_recent(limit=limit, namespace=NAMESPACE_LOG)
        # mnemosyne 模式下无独立日志表，改用 log 命名空间检索近期条目
        return await self.search_log("", top_k=limit, min_score=0.0)

    # -- 通用读取 ---------------------------------------------------------

    async def search_all(self, query: str, top_k: int = 5, *,
                         min_score: float = 0.1) -> list[dict[str, Any]]:
        """检索所有命名空间 (兼顾模式)。"""
        return await self.search(
            query, top_k=top_k, min_score=min_score,
            namespaces=(NAMESPACE_MEMORY, NAMESPACE_LOG),
        )

    async def get_working_memory(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取工作记忆（当前会话活跃上下文）。"""
        await self._ensure_ready()

        if self._backend == "mnemosyne" and self._engine is not None:
            try:
                results = await self._run(
                    self._engine.recall,
                    query="", top_k=limit,
                    temporal_weight=1.0,
                )
                return [{
                    "id": r.get("id", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "timestamp": r.get("timestamp", ""),
                } for r in (results or [])]
            except Exception as exc:
                logger.warning("Mnemosyne working-memory read failed: %s", exc)
        if self._fallback is not None:
            return await self._fallback.get_recent(limit=limit, namespace=NAMESPACE_MEMORY)
        return []

    # -- 整合支持（后端无关）------------------------------------------------
    # 记忆整合流程需要枚举/计数/删除条目。mnemosyne 用 working_memory +
    # episodic_memory 两张表且 namespace 存在 metadata_json 里，
    # fallback 用单表 memories，因此这里统一抽象，避免调用方直接摸
    # 内部连接（历史上整合逻辑写死 _fallback._conn，切到 mnemosyne 后
    # 会静默跳过整个 Tier2）。

    _MNEMOSYNE_TABLES = ("working_memory", "episodic_memory")

    async def count_entries(self, *, namespace: str | None = None) -> int:
        """统计条目数（用于整合前后对比）。"""
        await self._ensure_ready()

        if self._backend == "mnemosyne" and self._engine is not None:
            def _count() -> int:
                total = 0
                for table in self._MNEMOSYNE_TABLES:
                    sql = f"SELECT COUNT(*) FROM {table}"
                    params: tuple = ()
                    if namespace is not None:
                        sql += " WHERE json_extract(metadata_json, '$.namespace') = ?"
                        params = (namespace,)
                    try:
                        row = self._engine.conn.execute(sql, params).fetchone()
                        total += int(row[0]) if row else 0
                    except sqlite3.Error as exc:
                        logger.debug("count_entries on %s failed: %s", table, exc)
                return total

            return await self._run(_count)

        if self._fallback is not None:
            return await self._fallback.count_entries(namespace=namespace)
        return 0

    async def list_entries(
        self, *, limit: int = 2000, exclude_namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """枚举条目供整合使用（按时间倒序）。"""
        await self._ensure_ready()

        if self._backend == "mnemosyne" and self._engine is not None:
            def _list() -> list[dict[str, Any]]:
                rows: list[dict[str, Any]] = []
                for table in self._MNEMOSYNE_TABLES:
                    sql = (
                        f"SELECT id, content, importance, source, scope, timestamp, "
                        f"COALESCE(json_extract(metadata_json, '$.namespace'), 'memory') "
                        f"FROM {table}"
                    )
                    params: tuple = ()
                    if exclude_namespace is not None:
                        sql += (
                            " WHERE COALESCE(json_extract(metadata_json, '$.namespace'), "
                            "'memory') != ?"
                        )
                        params = (exclude_namespace,)
                    sql += " ORDER BY timestamp DESC LIMIT ?"
                    try:
                        for r in self._engine.conn.execute(sql, (*params, limit)):
                            rows.append({
                                "id": r[0],
                                "content": r[1] or "",
                                "importance": r[2] if r[2] is not None else 0.5,
                                "source": r[3] or "",
                                "scope": r[4] or "session",
                                "timestamp": r[5] or "",
                                "namespace": r[6] or NAMESPACE_MEMORY,
                            })
                    except sqlite3.Error as exc:
                        logger.debug("list_entries on %s failed: %s", table, exc)
                rows.sort(key=lambda item: item["timestamp"], reverse=True)
                return rows[:limit]

            return await self._run(_list)

        if self._fallback is not None:
            return await self._fallback.list_entries(
                limit=limit, exclude_namespace=exclude_namespace
            )
        return []

    async def delete_entries(self, entry_ids: list[str]) -> int:
        """按 id 批量删除条目，返回实际删除数。"""
        await self._ensure_ready()
        if not entry_ids:
            return 0

        if self._backend == "mnemosyne" and self._engine is not None:
            def _delete() -> int:
                # 必须走 engine.forget()：一条 remember 会同时落 legacy memories、
                # working_memory、memory_embeddings、vec_working、annotations。
                # 直接 DELETE 单表会留下孤儿行（计数虚高、向量索引残留）。
                removed = 0
                for eid in entry_ids:
                    hit = False
                    try:
                        hit = bool(self._engine.forget(eid))
                    except Exception as exc:
                        logger.debug("forget(%s) failed: %s", eid, exc)
                    if not hit:
                        # forget() 只覆盖 working_memory；sleep() 巩固后的条目
                        # 在 episodic_memory，需要补删。
                        try:
                            cur = self._engine.conn.execute(
                                "DELETE FROM episodic_memory WHERE id = ?", (eid,)
                            )
                            hit = bool(cur.rowcount and cur.rowcount > 0)
                        except sqlite3.Error as exc:
                            logger.debug("delete episodic %s failed: %s", eid, exc)
                    # gists 是 remember() 顺带写的摘要行，forget() 不清理，
                    # 不删会随整合次数持续堆积。
                    try:
                        self._engine.conn.execute(
                            "DELETE FROM gists WHERE memory_id = ?", (eid,)
                        )
                    except sqlite3.Error as exc:
                        logger.debug("delete gist for %s failed: %s", eid, exc)
                    if hit:
                        removed += 1
                self._engine.conn.commit()
                return removed

            return await self._run(_delete)

        if self._fallback is not None:
            return await self._fallback.delete_entries(entry_ids)
        return 0

    async def get_stats(self) -> dict[str, Any]:
        """记忆统计。"""
        await self._ensure_ready()
        if self._backend == "mnemosyne":
            def _stats() -> dict[str, Any]:
                counts = {}
                for table in self._MNEMOSYNE_TABLES:
                    try:
                        counts[table] = int(
                            self._engine.conn.execute(
                                f"SELECT COUNT(*) FROM {table}"
                            ).fetchone()[0]
                        )
                    except sqlite3.Error as exc:
                        logger.debug("stats count on %s failed: %s", table, exc)
                        counts[table] = 0
                return {
                    "backend": "mnemosyne",
                    "configured_backend": self._backend_config,
                    "data_dir": str(self._data_dir),
                    "db_path": str(self._data_dir / "beam.db"),
                    "vector_search": "on" if self._vector_ready else "off",
                    "count": sum(counts.values()),
                    "working_count": counts.get("working_memory", 0),
                    "episodic_count": counts.get("episodic_memory", 0),
                }

            return await self._run(_stats)
        return await self._fallback.get_stats()

    async def get(self, entry_id: str) -> dict[str, Any] | None:
        """按 id 读取单条记忆（主后端与兜底后端均支持）。"""
        await self._ensure_ready()
        if self._backend != "mnemosyne":
            return await self._fallback.get(entry_id)

        def _get() -> dict[str, Any] | None:
            for table in self._MNEMOSYNE_TABLES:
                try:
                    row = self._engine.conn.execute(
                        "SELECT id, content, source, importance, scope, timestamp, "
                        "COALESCE(json_extract(metadata_json, '$.namespace'), 'memory') "
                        f"FROM {table} WHERE id = ?",
                        (entry_id,),
                    ).fetchone()
                except sqlite3.Error as exc:
                    logger.debug("get on %s failed: %s", table, exc)
                    continue
                if row:
                    return {
                        "id": row[0],
                        "content": row[1] or "",
                        "source": row[2] or "",
                        "importance": row[3] if row[3] is not None else 0.5,
                        "scope": row[4] or "session",
                        "timestamp": row[5] or "",
                        "namespace": row[6] or NAMESPACE_MEMORY,
                        "tier": "beam",
                    }
            return None

        return await self._run(_get)

    async def update(self, entry_id: str, content: str | None,
                     **metadata: Any) -> bool:
        """按 id 更新记忆。

        content/importance 走 Mnemosyne 的 update()（同时刷新 working_memory
        的 FTS5 触发器和向量）；source/scope/namespace 直接改对应列/metadata，
        因为上游 update() 不覆盖这些字段。
        """
        await self._ensure_ready()
        if self._backend != "mnemosyne":
            return await self._fallback.update(entry_id, content, **metadata)

        new_importance = metadata.get("importance")
        new_source = metadata.get("source")
        new_scope = metadata.get("scope")
        new_namespace = metadata.get("namespace")
        text = None if content is None else str(content)
        if text is not None and len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS]

        def _update() -> bool:
            rows: list[tuple[str, Any]] = []
            for table in self._MNEMOSYNE_TABLES:
                try:
                    row = self._engine.conn.execute(
                        f"SELECT rowid, content, metadata_json FROM {table} WHERE id = ?",
                        (entry_id,),
                    ).fetchone()
                except sqlite3.Error as exc:
                    logger.debug("update lookup on %s failed: %s", table, exc)
                    continue
                if row:
                    rows.append((table, row))
            if not rows:
                return False

            changed = False
            engine_updated = False
            if text is not None or new_importance is not None:
                # 先把 working_memory + 旧 memories 表交给 Mnemosyne 同步，
                # 这样向量与 FTS 索引会一起更新。
                for table, _row in rows:
                    if table != "working_memory":
                        continue
                    try:
                        engine_updated = bool(self._engine.update(
                            entry_id,
                            content=text,
                            importance=new_importance,
                        ))
                        changed = changed or engine_updated
                    except Exception as exc:
                        logger.warning("Mnemosyne update(%s) failed: %s", entry_id, exc)
                        raise

            for table, row in rows:
                assignments: list[str] = []
                params: list[Any] = []
                handled_by_engine = table == "working_memory" and engine_updated
                if text is not None and not handled_by_engine:
                    assignments.append("content = ?")
                    params.append(text)
                if new_importance is not None and not handled_by_engine:
                    assignments.append("importance = ?")
                    params.append(max(0.0, min(1.0, float(new_importance))))
                if new_source is not None:
                    assignments.append("source = ?")
                    params.append(new_source)
                if new_scope is not None:
                    assignments.append("scope = ?")
                    params.append(new_scope)
                if assignments:
                    try:
                        self._engine.conn.execute(
                            f"UPDATE {table} SET {', '.join(assignments)} WHERE id = ?",
                            (*params, entry_id),
                        )
                        changed = True
                    except sqlite3.Error as exc:
                        logger.warning("update %s failed on %s: %s", entry_id, table, exc)
                if new_namespace is not None:
                    try:
                        self._engine.conn.execute(
                            f"UPDATE {table} SET metadata_json = "
                            "json_set(COALESCE(metadata_json, '{}'), '$.namespace', ?) "
                            "WHERE id = ?",
                            (new_namespace, entry_id),
                        )
                        changed = True
                    except sqlite3.Error as exc:
                        logger.warning(
                            "namespace update %s failed on %s: %s", entry_id, table, exc,
                        )
                if table == "episodic_memory" and text is not None:
                    # 摘要条目改正文后必须重算向量，否则命中内容与向量不一致。
                    try:
                        self._engine.beam._refresh_episodic_embedding(
                            entry_id, int(row[0]), text,
                        )
                    except Exception as exc:
                        logger.warning(
                            "episodic embedding refresh failed for %s: %s", entry_id, exc,
                        )

            # 旧 memories 表同步（working 路径已由 engine.update 处理）。
            if not engine_updated and any(
                table == "working_memory" for table, _row in rows
            ):
                try:
                    legacy_assignments: list[str] = []
                    legacy_params: list[Any] = []
                    if text is not None:
                        legacy_assignments.append("content = ?")
                        legacy_params.append(text)
                    if new_importance is not None:
                        legacy_assignments.append("importance = ?")
                        legacy_params.append(max(0.0, min(1.0, float(new_importance))))
                    if new_source is not None:
                        legacy_assignments.append("source = ?")
                        legacy_params.append(new_source)
                    if legacy_assignments:
                        self._engine.conn.execute(
                            f"UPDATE memories SET {', '.join(legacy_assignments)} "
                            "WHERE id = ?",
                            (*legacy_params, entry_id),
                        )
                        changed = True
                except sqlite3.Error as exc:
                    logger.debug("legacy memories update %s failed: %s", entry_id, exc)

            self._engine.conn.commit()
            return changed

        return await self._run(_update)

    async def delete(self, entry_id: str) -> bool:
        """按 id 删除单条记忆（含向量、gists 与 episodic 补删）。"""
        await self._ensure_ready()
        if self._backend != "mnemosyne":
            return await self._fallback.delete(entry_id)
        removed = await self.delete_entries([entry_id])
        return removed > 0

    # -- 内部 ---------------------------------------------------------------

    async def _ensure_ready(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _lookup_namespaces(self, entry_ids: list[str]) -> dict[str, str]:
        """批量读取条目实际写入的 namespace（存于 metadata_json）。

        Mnemosyne 的 recall() 不返回 metadata，因此命名空间无法从检索结果直接拿到。
        必须回查一次，否则只能靠 source 前缀猜——
        store(namespace="log", source="agent") 这类调用会被误判成 memory，
        导致工具日志泄漏进角色扮演层。
        """
        if not entry_ids or self._engine is None:
            return {}
        placeholders = ",".join("?" * len(entry_ids))
        found: dict[str, str] = {}
        for table in self._MNEMOSYNE_TABLES:
            try:
                rows = self._engine.conn.execute(
                    f"SELECT id, json_extract(metadata_json, '$.namespace') "
                    f"FROM {table} WHERE id IN ({placeholders})",
                    entry_ids,
                ).fetchall()
            except sqlite3.Error as exc:
                logger.debug("namespace lookup on %s failed: %s", table, exc)
                continue
            for entry_id, namespace in rows:
                if namespace:
                    found[entry_id] = namespace
        return found

    @staticmethod
    def _match_namespace_by_source(source: str, namespaces: tuple[str, ...]) -> bool:
        """兜底：无 metadata 的历史条目按 source 前缀推断命名空间。"""
        is_tool = source.startswith("tool:")
        want_memory = NAMESPACE_MEMORY in namespaces
        want_log = NAMESPACE_LOG in namespaces
        if is_tool and want_log:
            return True
        if not is_tool and want_memory:
            return True
        # 同时要两个命名空间时都返回 True
        return want_memory and want_log

    @staticmethod
    def _extract_text(msg: Any) -> str:
        if hasattr(msg, "content"):
            c = msg.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                texts = [seg.get("text", "") for seg in c if isinstance(seg, dict)]
                return " ".join(texts)
        return str(msg) if msg else ""

    async def close(self) -> None:
        self._initialized = False
        self._backend = "uninitialized"
        if self._fallback is not None:
            await self._fallback.close()

        executor = self._executor
        engine = self._engine
        self._executor = None
        self._engine = None
        self._vector_ready = False
        if executor is None:
            return

        # 引擎连接是本线程 thread-local 的，必须在同一线程关闭，
        # 否则会留下未关闭的 sqlite 句柄（Windows 下会锁住 db 文件）。
        loop = asyncio.get_running_loop()

        def _close_engine() -> None:
            candidates = [
                getattr(engine, "conn", None),
                getattr(getattr(engine, "beam", None), "conn", None),
            ]
            for conn in candidates:
                if conn is None:
                    continue
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响其余清理
                    pass

        try:
            await loop.run_in_executor(executor, _close_engine)
        except Exception as exc:  # noqa: BLE001 - 关闭阶段只记录
            logger.debug("Mnemosyne engine close failed: %s", exc)
        executor.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------
# Fallback — 纯 SQLite FTS5 存储 (带命名空间)
# ---------------------------------------------------------------------------

def _fts_escape(query: str) -> str:
    """把查询包成 FTS5 字符串字面量，避免用户输入被当作 MATCH 语法。

    FTS5 里 " * ( ) : ^ - 等字符有语法含义，中文标点也可能触发解析错误。
    包成双引号短语并转义内部双引号是最稳妥的做法。
    """
    return '"' + query.replace('"', '""') + '"'


class _FallbackStore:
    """Mnemosyne 不可用时的 FTS5 降级存储 (支持 memory/log 双命名空间)。"""

    def __init__(self, data_dir: Path) -> None:
        self._db_path = data_dir / "beam_fallback.db"
        self._conn = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        self._conn = await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # 建表 — 带 namespace 列
        conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'agent',
            importance REAL NOT NULL DEFAULT 0.5,
            scope TEXT NOT NULL DEFAULT 'session',
            namespace TEXT NOT NULL DEFAULT 'memory',
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        )""")
        # 兼容旧表: 如果 namespace 列不存在则添加
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'memory'")
        except sqlite3.OperationalError:
            pass  # 列已存在

        # trigram 分词器原生支持中文（unicode61 会把整句 CJK 当成一个 token）。
        # SQLite >= 3.34 才有 trigram，失败则回退到默认分词器。
        self._fts_tokenizer = None
        for tokenizer in ("trigram", None):
            ddl_suffix = f", tokenize='{tokenizer}'" if tokenizer else ""
            try:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                    f"USING fts5(content, source{ddl_suffix})"
                )
                self._fts_tokenizer = tokenizer or "unicode61"
                break
            except sqlite3.OperationalError as exc:
                logger.debug("FTS5 tokenizer %s unavailable: %s", tokenizer, exc)

        if self._fts_tokenizer is None:
            logger.warning(
                "FTS5 virtual table unavailable; fallback search degrades to LIKE scan"
            )

        # 旧库可能是 unicode61 建的，检测并就地重建为 trigram
        if self._fts_tokenizer == "trigram":
            try:
                existing = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_fts'"
                ).fetchone()
                if existing and "trigram" not in (existing[0] or ""):
                    logger.info("Rebuilding memories_fts with trigram tokenizer for CJK support")
                    conn.execute("DROP TABLE memories_fts")
                    conn.execute(
                        "CREATE VIRTUAL TABLE memories_fts "
                        "USING fts5(content, source, tokenize='trigram')"
                    )
                    conn.execute(
                        "INSERT INTO memories_fts (rowid, content, source) "
                        "SELECT id, content, source FROM memories"
                    )
            except sqlite3.OperationalError as exc:
                logger.warning("memories_fts rebuild failed: %s", exc)

        conn.commit()
        logger.info(
            "BeamMemory fallback (FTS5/%s) initialized @ %s",
            self._fts_tokenizer or "LIKE-only",
            self._db_path,
        )
        return conn

    async def store(self, content: str, *, source: str = "agent",
                    importance: float = 0.5,
                    scope: str = "session",
                    namespace: str = NAMESPACE_MEMORY) -> str:
        if self._conn is None:
            return ""
        loop = asyncio.get_running_loop()

        def _do() -> str:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                "INSERT INTO memories (content, source, importance, scope, namespace, timestamp) VALUES (?,?,?,?,?,?)",
                (content, source, importance, scope, namespace, ts),
            )
            eid = f"fb_{cur.lastrowid}"
            try:
                self._conn.execute(
                    "INSERT INTO memories_fts (rowid, content, source) VALUES (?,?,?)",
                    (cur.lastrowid, content, source),
                )
            except Exception:
                pass
            self._conn.commit()
            return eid

        return await loop.run_in_executor(None, _do)

    async def search(self, query: str, top_k: int = 5, *,
                     namespaces: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        if self._conn is None or not query.strip():
            return []
        ns = namespaces or (NAMESPACE_MEMORY,)
        loop = asyncio.get_running_loop()

        def _do() -> list[dict[str, Any]]:
            placeholders = ",".join("?" for _ in ns)
            rows = []
            # trigram 分词器对中英文都有效，用 bm25() 做相关性排序；
            # 仅在 FTS5 不可用或返回 0 条时才降级到 LIKE 扫描。
            if self._fts_tokenizer:
                try:
                    rows = self._conn.execute(
                        f"""SELECT m.id, m.content, m.source, m.importance,
                                   m.scope, m.timestamp, m.namespace
                           FROM memories m
                           JOIN memories_fts f ON m.id = f.rowid
                           WHERE memories_fts MATCH ?
                             AND m.namespace IN ({placeholders})
                           ORDER BY bm25(memories_fts) ASC,
                                    m.importance DESC, m.timestamp DESC
                           LIMIT ?""",
                        (_fts_escape(query), *ns, top_k),
                    ).fetchall()
                except (sqlite3.OperationalError, sqlite3.ProgrammingError) as exc:
                    logger.debug("FTS5 MATCH failed (%s); falling back to LIKE", exc)

            # FTS5 返回 0 条或中文查询 → LIKE 降级
            if not rows:
                rows = self._conn.execute(
                    f"""SELECT id, content, source, importance,
                               scope, timestamp, namespace
                       FROM memories
                       WHERE content LIKE ?
                         AND namespace IN ({placeholders})
                       ORDER BY importance DESC, timestamp DESC
                       LIMIT ?""",
                    (f"%{query}%", *ns, top_k),
                ).fetchall()
            return [{
                "id": f"fb_{r[0]}", "content": r[1], "score": r[3],
                "source": r[2], "importance": r[3], "timestamp": r[5],
                "namespace": r[6], "tier": "beam_fallback",
            } for r in rows]

        return await loop.run_in_executor(None, _do)

    async def count_entries(self, *, namespace: str | None = None) -> int:
        if self._conn is None:
            return 0
        loop = asyncio.get_running_loop()

        def _do() -> int:
            if namespace is None:
                row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ?", (namespace,)
                ).fetchone()
            return int(row[0]) if row else 0

        return await loop.run_in_executor(None, _do)

    async def list_entries(
        self, *, limit: int = 2000, exclude_namespace: str | None = None
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        loop = asyncio.get_running_loop()

        def _do() -> list[dict[str, Any]]:
            sql = (
                "SELECT id, content, importance, source, scope, timestamp, namespace "
                "FROM memories"
            )
            params: tuple = ()
            if exclude_namespace is not None:
                sql += " WHERE namespace IS NULL OR namespace != ?"
                params = (exclude_namespace,)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            return [{
                "id": f"fb_{r[0]}",
                "content": r[1] or "",
                "importance": r[2] if r[2] is not None else 0.5,
                "source": r[3] or "",
                "scope": r[4] or "session",
                "timestamp": r[5] or "",
                "namespace": r[6] or NAMESPACE_MEMORY,
            } for r in self._conn.execute(sql, (*params, limit))]

        return await loop.run_in_executor(None, _do)

    async def delete_entries(self, entry_ids: list[str]) -> int:
        if self._conn is None or not entry_ids:
            return 0
        loop = asyncio.get_running_loop()

        def _do() -> int:
            raw_ids = []
            for eid in entry_ids:
                text = str(eid)
                if text.startswith("fb_"):
                    text = text[3:]
                try:
                    raw_ids.append(int(text))
                except ValueError:
                    continue
            if not raw_ids:
                return 0
            cur = self._conn.executemany(
                "DELETE FROM memories WHERE id = ?", [(i,) for i in raw_ids]
            )
            removed = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            try:
                self._conn.executemany(
                    "DELETE FROM memories_fts WHERE rowid = ?", [(i,) for i in raw_ids]
                )
            except sqlite3.Error:
                pass
            self._conn.commit()
            return removed

        return await loop.run_in_executor(None, _do)

    async def get_recent(self, limit: int = 20,
                         namespace: str = NAMESPACE_MEMORY) -> list[dict[str, Any]]:
        """获取最近条目（按时间倒序）。"""
        if self._conn is None:
            return []
        async with self._lock:
            rows = self._conn.execute(
                """SELECT id, content, source, importance, timestamp
                   FROM memories
                   WHERE namespace = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (namespace, limit),
            ).fetchall()
        return [{
            "id": f"fb_{r[0]}", "content": r[1][:500] if r[1] else "",
            "source": r[2], "importance": r[3], "timestamp": r[4],
            "namespace": namespace,
        } for r in rows]

    async def get(self, entry_id: str) -> dict[str, Any] | None:
        if self._conn is None or not entry_id.startswith("fb_"):
            return None
        async with self._lock:
            row = self._conn.execute(
                "SELECT id, content, source, importance, scope, timestamp, namespace FROM memories WHERE id = ?",
                (entry_id[3:],),
            ).fetchone()
        if row is None:
            return None
        return {"id": entry_id, "content": row[1], "source": row[2], "importance": row[3],
                "scope": row[4], "timestamp": row[5], "namespace": row[6], "tier": "beam_fallback"}

    async def update(self, entry_id: str, content: str, **metadata: Any) -> bool:
        if self._conn is None or not entry_id.startswith("fb_"):
            return False
        async with self._lock:
            current = self._conn.execute(
                "SELECT source FROM memories WHERE id = ?", (entry_id[3:],)
            ).fetchone()
            if current is None:
                return False
            source = metadata.get("source", current[0])
            values = [content]
            assignments = ["content = ?"]
            for key in ("source", "importance", "scope", "namespace"):
                if key in metadata:
                    assignments.append(f"{key} = ?")
                    values.append(metadata[key])
            values.append(entry_id[3:])
            cur = self._conn.execute(f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?", values)
            self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (entry_id[3:],))
            try:
                self._conn.execute("INSERT INTO memories_fts (rowid, content, source) VALUES (?,?,?)",
                                   (entry_id[3:], content, source))
            except sqlite3.OperationalError:
                pass
            self._conn.commit()
        return cur.rowcount > 0

    async def delete(self, entry_id: str) -> bool:
        if self._conn is None or not entry_id.startswith("fb_"):
            return False
        async with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id[3:],))
            self._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (entry_id[3:],))
            self._conn.commit()
        return cur.rowcount > 0

    async def get_stats(self) -> dict[str, Any]:
        if self._conn is None:
            return {"backend": "fallback", "count": 0}
        async with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            mem = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace=?", (NAMESPACE_MEMORY,)
            ).fetchone()[0]
            log = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace=?", (NAMESPACE_LOG,)
            ).fetchone()[0]
        return {
            "backend": "fallback (FTS5)",
            "count": total,
            "memory_count": mem,
            "log_count": log,
        }

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
