"""回声（Echo）——中文长期陪伴记忆系统。

分层：
    Tier 0  短期      最近几轮
    Tier 1  层级      会话摘要 + 逐层下钻
    Tier 2  拾忆      向量检索（Mnemosyne 后端）
    Tier 3  核心      偏好 / 边界
    Facts   事实      时间片摘要 + 原文回取 + 事件卡 + 原文轮 BM25

随仓库 vendored 了改过底层的 ``mnemosyne``：导入本包时会把 ``vendor/``
放到 ``sys.path`` 最前面，因此无需单独安装 mnemosyne-memory。
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

__version__ = "0.1.0"
