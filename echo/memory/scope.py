"""渠道人格会话的记忆作用域。

渠道人格（如 qq-mimic 的个人 QQ）代表的是「以真实用户身份接入的外部对话」，
它们的人设与记忆都不该和主项目记忆互通，否则会出现角色与记忆互相污染。

作用域用网关的渠道路由键表示（``<channel>__<chat_id>__<hash>``），
因此同一个好友/群聊的多次会话共享同一份隔离记忆，
而不同的好友/群聊之间完全隔离。
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

# 需要隔离记忆的渠道。渠道名同时也是路由键前缀（见 channels.types._slug）。
ISOLATED_CHANNEL_SLUGS: tuple[str, ...] = ("qq-mimic",)

# 网关路由会话名的分隔符，与 gateway.route_sessions 保持一致。
_ROUTE_SESSION_DELIMITER = "__session__"

_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-Z_-]+")

# 当前协程/任务所处的隔离作用域；None 表示主项目记忆。
active_memory_scope: ContextVar[str | None] = ContextVar(
    "active_memory_scope", default=None,
)


def memory_scope_for_session(session_name: str) -> str | None:
    """返回会话所属的隔离作用域；主项目会话返回 ``None``。"""
    name = (session_name or "").strip()
    if not name:
        return None
    route_key = name.split(_ROUTE_SESSION_DELIMITER, 1)[0].strip()
    if not route_key:
        return None
    if any(
        route_key == slug or route_key.startswith(f"{slug}__")
        for slug in ISOLATED_CHANNEL_SLUGS
    ):
        return route_key
    return None


def scope_directory_name(scope: str) -> str:
    """把作用域键转换成文件系统安全的目录名。"""
    cleaned = _SAFE_NAME_RE.sub("_", scope).strip("_")
    return cleaned or "scope"


@contextmanager
def memory_scope(session_name: str) -> Iterator[str | None]:
    """在会话处理期间激活隔离作用域（主项目会话下是空操作）。"""
    scope = memory_scope_for_session(session_name)
    if not scope:
        yield None
        return
    token: Token = active_memory_scope.set(scope)
    try:
        yield scope
    finally:
        active_memory_scope.reset(token)
