#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 models/*.lock 下载模型。

用法：
    python scripts/fetch_models.py                 # 下载全部
    python scripts/fetch_models.py --only bge-small-zh-v1.5
    python scripts/fetch_models.py --source modelscope   # 国内网络
    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_models.py

说明：
- 向量模型走 fastembed（ONNX），下载到 fastembed 自己的缓存目录；
- 重排模型走 transformers，下载到 models/<name>/，加载时 CUDA 自动用 fp16。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def locks() -> list[dict]:
    out = []
    for path in sorted(MODELS.glob("*.lock")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def fetch_embedding(spec: dict, source: str) -> None:
    if source == "modelscope":
        try:
            from modelscope import snapshot_download  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("未安装 modelscope：pip install modelscope") from exc
        path = snapshot_download(spec["model_id"])
        print(f"[embedding] modelscope -> {path}")
        return
    from fastembed import TextEmbedding

    # 构造一次 TextEmbedding 就会把 ONNX 权重拉到本地缓存
    TextEmbedding(model_name=spec["model_id"])
    print(f"[embedding] fastembed 已就绪：{spec['model_id']}")


def fetch_reranker(spec: dict, source: str) -> None:
    target = MODELS / spec["name"]
    if (target / "config.json").exists():
        print(f"[reranker] 已存在：{target}")
        return
    repo = spec["model_id"]
    if source == "modelscope":
        try:
            from modelscope import snapshot_download  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("未安装 modelscope：pip install modelscope") from exc
        path = snapshot_download(repo, local_dir=str(target))
        print(f"[reranker] modelscope -> {path}")
        return
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo, local_dir=str(target),
                             ignore_patterns=["*.onnx", "*.msgpack", "*.h5"])
    print(f"[reranker] huggingface -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    for spec in locks():
        if args.only and spec["name"] != args.only:
            continue
        print(f"== {spec['name']} ({spec['kind']}) ==")
        if spec["kind"] == "embedding":
            fetch_embedding(spec, args.source)
        elif spec["kind"] == "reranker":
            fetch_reranker(spec, args.source)
        else:
            print(f"未知类型：{spec['kind']}", file=sys.stderr)


if __name__ == "__main__":
    main()
