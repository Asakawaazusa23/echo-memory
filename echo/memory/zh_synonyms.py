"""中文同义词扩展（仅用于时间片归档的词法召回）。

优先查 cnsyn（中文同义词词林，内存倒排、零运行时依赖）；不可用时退回内置的
少量领域词表。扩展词必须以低权重参与打分，避免“窗口→海口”这类词典噪声
改变排序。查表结果按词缓存，重复查询不重复加载。
"""
from __future__ import annotations

import logging
import warnings
from functools import lru_cache

logger = logging.getLogger(__name__)

# 领域/口语词表：cnsyn 收录不足或口语化太强的词，按项目场景补充。
# 不含单字词（如“买”）：分词器会丢弃单字，单字同义词无法与索引词项对上。
_CUSTOM_GROUPS = (
    ("购买", "采购", "选购", "入手", "购置", "采买"),
    ("体重", "称重", "重量", "分量"),
    ("顶班", "替班", "代班", "顶窗口", "看窗口"),
    ("写信", "寄信", "去信", "通信", "致函", "修函"),
    ("超市", "超级市场", "商超", "卖场"),
    ("西瓜", "寒瓜"),
    ("晾衣架", "衣架", "晾衣杆"),
)

_MAX_TERMS = 2

_CUSTOM: dict[str, tuple[str, ...]] = {}
for _group in _CUSTOM_GROUPS:
    for _word in _group:
        _CUSTOM[_word] = tuple(word for word in _group if word != _word)

_cnsyn_ready: bool | None = None


def _load_cnsyn() -> bool:
    global _cnsyn_ready
    if _cnsyn_ready is not None:
        return _cnsyn_ready
    try:
        import cnsyn

        for group in _CUSTOM_GROUPS:
            try:
                cnsyn.add_synonyms(list(group))
            except ValueError:
                continue
        _cnsyn_ready = True
    except Exception as exc:  # noqa: BLE001 - 可选依赖，缺失必须降级
        logger.info("cnsyn 不可用（%s），同义扩展仅用内置词表", type(exc).__name__)
        _cnsyn_ready = False
    return _cnsyn_ready


def _acceptable(term: str, word: str) -> bool:
    """过滤掉词形重叠、单字和过长短语：这些词靠 bigram 已能覆盖。"""
    if not word or word == term or len(word) < 2 or len(word) > 4:
        return False
    return term not in word and word not in term


@lru_cache(maxsize=4096)
def synonyms(term: str, limit: int = _MAX_TERMS) -> tuple[str, ...]:
    """返回少量近义词；只接受 2–4 字且与查询词无包含关系的词。"""
    term = (term or "").strip().lower()
    if len(term) < 2 or term.isascii():
        return ()
    found: list[str] = []

    for word in _CUSTOM.get(term, ()):
        if _acceptable(term, word) and word not in found:
            found.append(word)

    if len(found) < limit and _load_cnsyn():
        import cnsyn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for word in cnsyn.search(term, topK=limit + 2):
                if _acceptable(term, word) and word not in found:
                    found.append(word)
                    if len(found) >= limit:
                        break
    return tuple(found[:limit])
