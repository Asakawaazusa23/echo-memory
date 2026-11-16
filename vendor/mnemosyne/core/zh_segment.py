"""CJK 检索词项工具（本地魔改版 Mnemosyne 专用）。

Mnemosyne 的 unicode61 FTS 对连续中文几乎不建索引（实测单字/双字 MATCH 均为 0），
因此中文检索必须走 bigram 词项 + LIKE 候选。本模块只提供纯 Python 的轻量实现，
不引入额外依赖，避免 site-packages 内出现项目依赖。
"""
from __future__ import annotations

import re
from typing import Iterable

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9_.:/+-]*")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def cjk_chars(text: str) -> list[str]:
    return _CJK_RE.findall(text or "")


def bigrams(text: str) -> list[str]:
    """按连续 CJK 片段生成 2 字 bigram；单字段保留单字。"""
    grams: list[str] = []
    for run in cjk_chars(text):
        if len(run) == 1:
            grams.append(run)
            continue
        grams.extend(run[index:index + 2] for index in range(len(run) - 1))
    return grams


def terms(text: str) -> set[str]:
    """查询/内容共用的词项集合：CJK bigram + ASCII 词。"""
    lowered = (text or "").lower()
    output = set(bigrams(lowered))
    output.update(match.group(0) for match in _ASCII_RE.finditer(lowered))
    return output


def coverage(query: str, content: str) -> float:
    """查询词项被内容覆盖的比例，∈[0,1]。"""
    query_terms = terms(query)
    if not query_terms:
        return 0.0
    return len(query_terms & terms(content)) / len(query_terms)


def dice(query: str, content: str) -> float:
    """bigram/词项 Dice 系数，∈[0,1]，比单字覆盖率更抗常见字噪声。"""
    query_terms = terms(query)
    content_terms = terms(content)
    if not query_terms or not content_terms:
        return 0.0
    return 2 * len(query_terms & content_terms) / (len(query_terms) + len(content_terms))


def fts_terms(query: str, limit: int = 16) -> list[str]:
    """把查询转成 FTS5 安全词项（短词/词组的引号由调用方加）。"""
    lowered = (query or "").lower()
    output: list[str] = []
    seen: set[str] = set()
    for term in list(bigrams(lowered)) + [m.group(0) for m in _ASCII_RE.finditer(lowered)]:
        if term and term not in seen:
            seen.add(term)
            output.append(term)
        if len(output) >= limit:
            break
    return output


def first_term(terms_list: Iterable[str]) -> str:
    return next(iter(terms_list), "")
