"""中文分词与检索词项工具（Tier1/Tier3 共用）。

设计目标：
- 低开销：分词器进程级单例，结果走 lru_cache；
- 可降级：THULAC → jieba → 纯 bigram，任何一步失败都不影响记忆写入；
- 检索友好：除词级 token 外，始终补中文 bigram，避免未登录词/切错词导致 0 命中。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/+-]*")
_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")

_STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "在", "有", "和", "就", "不", "也", "都", "这", "那", "这个", "那个",
    "什么", "怎么", "为什么", "哪个", "哪些", "多少", "几", "吗", "呢", "吧",
    "啊", "呀", "啦", "的话", "一个", "一下", "可以", "没有", "不是", "还",
    "又", "再", "很", "太", "更", "最", "会", "能", "要", "把", "被", "给",
}

_lock = threading.Lock()
_segmenter = None
_provider = "uninitialized"
_failed: set[str] = set()


def normalize(text: str) -> str:
    """去掉空白/标点，用于“整句精确匹配”。"""
    return _NORMALIZE_RE.sub("", (text or "").lower())


def _cjk_runs(text: str) -> list[str]:
    return _CJK_RE.findall(text or "")


def _bigrams(text: str) -> list[str]:
    grams: list[str] = []
    for run in _cjk_runs(text):
        if len(run) == 1:
            grams.append(run)
            continue
        for index in range(len(run) - 1):
            grams.append(run[index:index + 2])
    return grams


def _try_thulac():
    global _provider
    try:
        import thulac  # type: ignore

        instance = thulac.thulac(seg_only=True)
        _provider = "thulac"
        logger.info("中文分词器：THULAC")
        return instance
    except Exception as exc:  # noqa: BLE001 - 可选依赖，缺失必须降级
        _failed.add("thulac")
        logger.warning("THULAC 不可用（%s: %s），尝试 jieba", type(exc).__name__, exc)
        return None


def _try_jieba():
    global _provider
    try:
        import jieba  # type: ignore

        jieba.setLogLevel(logging.WARNING)
        _provider = "jieba"
        logger.info("中文分词器：jieba")
        return jieba
    except Exception as exc:  # noqa: BLE001
        _failed.add("jieba")
        logger.warning("jieba 不可用（%s: %s），降级为 bigram", type(exc).__name__, exc)
        return None


def _select_segmenter():
    """按配置选择分词器；auto 依次尝试 THULAC、jieba。"""
    global _segmenter, _provider
    desired = (os.environ.get("NOVAEECHO_SEGMENTER") or "auto").strip().lower()
    with _lock:
        if _segmenter is not None or _provider == "bigram":
            return _segmenter
        if desired in ("bigram", "none"):
            _provider = "bigram"
            logger.info("中文分词器：bigram（显式配置）")
            return None
        if desired in ("thulac", "auto") and "thulac" not in _failed:
            _segmenter = _try_thulac()
        if _segmenter is None and desired in ("jieba", "auto") and "jieba" not in _failed:
            _segmenter = _try_jieba()
        if _segmenter is None:
            _provider = "bigram"
            logger.info("中文分词器：bigram（降级）")
        return _segmenter


def provider_name() -> str:
    _select_segmenter()
    return _provider


# 中文数字字符：用于识别“二十七号/六块/九斤二/半天”这类数量词项。
CN_NUMERAL_CHARS = frozenset("零〇一二三四五六七八九十百千万亿两半")
_PROPER_TAGS = frozenset({"np", "ns", "ni", "nz", "nr", "nt"})
_NUMERAL_TAGS = frozenset({"m", "mq"})

_tagger = None
_tagger_kind = ""
_tagger_failed = False
_tagger_lock = threading.Lock()


def _select_tagger():
    """选择带词性的标注器；只用于查询词加权，不影响检索分词。"""
    global _tagger, _tagger_kind, _tagger_failed
    desired = (os.environ.get("NOVAEECHO_SEGMENTER") or "auto").strip().lower()
    with _tagger_lock:
        if _tagger is not None or _tagger_failed:
            return _tagger
        if desired in ("bigram", "none"):
            _tagger_failed = True
            return None
        if desired in ("thulac", "auto"):
            try:
                import thulac  # type: ignore

                _tagger = thulac.thulac(seg_only=False)
                _tagger_kind = "thulac"
                return _tagger
            except Exception as exc:  # noqa: BLE001
                logger.warning("词性标注器 THULAC 不可用（%s: %s）",
                               type(exc).__name__, exc)
                if desired == "thulac":
                    _tagger_failed = True
                    return None
        try:
            import jieba  # type: ignore
            import jieba.posseg as posseg  # type: ignore

            jieba.setLogLevel(logging.WARNING)
            _tagger = posseg
            _tagger_kind = "jieba"
            return _tagger
        except Exception as exc:  # noqa: BLE001
            logger.warning("词性标注器 jieba.posseg 不可用（%s: %s）",
                           type(exc).__name__, exc)
            _tagger_failed = True
            return None


def _tag_pairs(text: str) -> list[tuple[str, str]]:
    tagger = _select_tagger()
    if tagger is None or not text:
        return []
    try:
        if _tagger_kind == "thulac":
            return [(str(word), str(tag)) for word, tag in tagger.cut(text, text=False)]
        return [(str(pair.word), str(pair.flag)) for pair in tagger.cut(text)]
    except Exception as exc:  # noqa: BLE001 - 标注失败只影响加权
        logger.warning("词性标注失败（%s），本次不做专名/数字加权", type(exc).__name__)
        return []


@lru_cache(maxsize=8192)
def query_features(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回查询中的(专有名词, 数量词)。仅用于提高这些词项的权重。"""
    proper: list[str] = []
    numeral: list[str] = []
    for word, tag in _tag_pairs((text or "").lower()):
        word = word.strip().lower()
        if not word:
            continue
        if tag in _PROPER_TAGS and len(word) >= 2 and word not in proper:
            proper.append(word)
        if tag in _NUMERAL_TAGS and word not in numeral:
            numeral.append(word)
    return tuple(proper), tuple(numeral)


def _word_tokens(text: str) -> list[str]:
    segmenter = _select_segmenter()
    if segmenter is None:
        return []
    try:
        if _provider == "thulac":
            # thulac.cut(text, text=True) 返回空格分隔的词串
            raw = segmenter.cut(text, text=True)
            return [part for part in str(raw).split() if part]
        return list(segmenter.lcut(text))
    except Exception as exc:  # noqa: BLE001 - 分词失败不能阻断写入
        logger.warning("分词失败（%s），本条降级为 bigram", type(exc).__name__)
        return []


def _clean(tokens: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip().lower()
        if not token or token in seen:
            continue
        has_cjk = bool(_CJK_RE.search(token))
        has_ascii = bool(re.search(r"[a-z0-9]", token))
        if not (has_cjk or has_ascii):
            continue
        if token in _STOPWORDS:
            continue
        if has_cjk and not has_ascii and len(token) < 2:
            # 单字只在 bigram/单字实体里保留，避免“的/了”这类噪声
            continue
        seen.add(token)
        output.append(token)
    return output


@lru_cache(maxsize=8192)
def _segment_cached(text: str, max_terms: int) -> tuple[str, ...]:
    lowered = (text or "").lower()
    tokens: list[str] = []
    tokens.extend(_word_tokens(lowered))
    tokens.extend(_bigrams(lowered))
    tokens.extend(match.group(0) for match in _ASCII_RE.finditer(lowered))
    return tuple(_clean(tokens)[:max_terms])


def segment(text: str, *, max_terms: int = 24) -> list[str]:
    """返回检索用词项：词 + 中文 bigram + ASCII token。"""
    return list(_segment_cached(text, max_terms))


def index_text(text: str, *, max_terms: int = 160) -> str:
    """把一段文本转成空格分隔的索引串，供 FTS5 预分词索引使用。"""
    return " ".join(_segment_cached(text, max_terms))
