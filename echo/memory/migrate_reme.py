"""ReMe → 4 层记忆系统迁移脚本。

将 .novaecho/reme/memory/*.md 中的旧 ReMe markdown 记忆
迁移至新的 4 层记忆系统:
- 用户偏好/关键事实 → Tier 3 CoreMemory (永久固化)
- 每日记忆/经验反思 → Tier 1 HierarchicalMemory (5级分层)
- MEMORY.md 核心记忆 → Tier 3 CoreMemory

用法:
    python -m novaecho.memory.migrate_reme
    python -m novaecho.memory.migrate_reme --dry-run  # 只预览不写入
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from .settings import MemorySettings
from .service import MemoryService


# ---------------------------------------------------------------------------
# 记忆内容解析
# ---------------------------------------------------------------------------

def parse_daily_memory(file_path: Path) -> dict:
    """解析一个 daily memory markdown 文件为结构化数据。

    返回: {
        "date": "2026-05-29",
        "facts": [str, ...],          # 客观事实
        "preferences": [str, ...],    # 用户偏好
        "reflections": [str, ...],    # 经验反思
        "raw": str,                   # 原始全文
    }
    """
    text = file_path.read_text(encoding="utf-8")
    stem = file_path.stem  # e.g. "2026-05-29"

    # 提取日期
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    date = date_match.group(1) if date_match else stem

    facts: list[str] = []
    preferences: list[str] = []
    reflections: list[str] = []

    # 按 section 分割
    current_section = "facts"
    current_block: list[str] = []

    for line in text.splitlines():
        # 检测 section 标题
        lower = line.lower()
        if "持久化记忆" in line or "事实记忆" in line or "📌" in line:
            if current_block:
                _classify_block(current_section, current_block,
                                facts, preferences, reflections)
            current_section = "facts"
            current_block = []
        elif "经验反思" in line or "反思" in line or "💡" in line:
            if current_block:
                _classify_block(current_section, current_block,
                                facts, preferences, reflections)
            current_section = "reflections"
            current_block = []
        elif "偏好" in line or "审美" in line or "兴趣" in line:
            if current_block:
                _classify_block(current_section, current_block,
                                facts, preferences, reflections)
            current_section = "preferences"
            current_block = []

        current_block.append(line)

    # 最后一块
    if current_block:
        _classify_block(current_section, current_block,
                        facts, preferences, reflections)

    return {
        "date": date,
        "facts": facts,
        "preferences": preferences,
        "reflections": reflections,
        "raw": text,
    }


def _classify_block(section: str, lines: list[str],
                    facts: list[str], preferences: list[str],
                    reflections: list[str]) -> None:
    """将一块内容归类到对应列表。"""
    text = "\n".join(lines).strip()
    if not text or len(text) < 20:
        return

    # 提取要点行（以 - 或 * 开头的行）
    points = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            points.append(stripped[2:])
        elif stripped.startswith("####") or stripped.startswith("###"):
            points.append(stripped.lstrip("#").strip())

    block_text = "\n".join(points) if points else text[:500]

    if section == "preferences":
        preferences.append(block_text)
    elif section == "reflections":
        reflections.append(block_text)
    else:
        facts.append(block_text)


def extract_user_preferences(texts: list[str]) -> list[str]:
    """从多个文本块中提取高价值用户偏好。

    使用两种策略:
    1. 强信号: "我喜欢/偏好/prefer..." 等明确偏好声明
    2. 事实信号: 具体个人信息（学校、职业、家庭等）
    """
    seen = set()
    preferences: list[str] = []

    for text in texts:
        for line in text.splitlines():
            clean = line.strip().lstrip("- *#").strip()
            if not clean or len(clean) < 8:
                continue

            is_preference = False

            # 策略1: 明确偏好声明 ("我喜欢...", "我喜欢白毛红瞳", etc.)
            if re.search(r"用户(?:说|表示|认为|偏好|喜欢|想要|希望)", clean):
                is_preference = True
            elif re.search(r"(?:我喜欢|我偏好|我喜欢|I prefer|I love)", clean):
                is_preference = True
            # 强关键词直接命中
            elif any(kw in clean for kw in [
                "白毛红瞳", "老祖宗严选",
                "广州大学附属中学", "富源校区",
                "C172R", "C172S",
            ]):
                is_preference = True

            # 策略2: 具体个人信息事实
            elif re.search(r"(?:孩子|儿子|初中|年级|飞行员|学员)", clean):
                # 排除纯讨论性的提及
                if not re.search(r"(?:可以|建议|可能|应该|需要)", clean):
                    is_preference = True

            if is_preference:
                # 去重
                fingerprint = clean[:40]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    preferences.append(clean)

    return preferences


# ---------------------------------------------------------------------------
# 迁移主流程
# ---------------------------------------------------------------------------

async def migrate(
    old_dir: Path,
    new_settings: MemorySettings,
    *,
    dry_run: bool = False,
) -> dict:
    """执行迁移。

    参数:
        old_dir: 旧 ReMe 目录 (e.g. .novaecho/reme)
        new_settings: 新记忆系统配置
        dry_run: 只预览不写入
    """
    memory_dir = old_dir / "memory"
    if not memory_dir.is_dir():
        print(f"[ERROR] 记忆目录不存在: {memory_dir}")
        return {"files": 0, "facts": 0, "preferences": 0, "reflections": 0}

    md_files = sorted(memory_dir.glob("*.md"))
    print(f"[INFO] 发现 {len(md_files)} 个记忆文件")

    # 1. 解析所有文件
    all_parsed = []
    for f in md_files:
        try:
            parsed = parse_daily_memory(f)
            all_parsed.append(parsed)
            print(f"  [{parsed['date']}] facts={len(parsed['facts'])} "
                  f"pref={len(parsed['preferences'])} "
                  f"refl={len(parsed['reflections'])}")
        except Exception as e:
            print(f"  [WARN] 解析失败 {f.name}: {e}")

    # 2. 汇总
    total_facts = []
    total_preferences = []
    total_reflections = []
    for p in all_parsed:
        total_facts.extend(p["facts"])
        total_preferences.extend(p["preferences"])
        total_reflections.extend(p["reflections"])

    # 3. 提取高价值用户偏好
    key_preferences = extract_user_preferences(
        [p["raw"] for p in all_parsed]
    )

    print(f"\n[SUMMARY]")
    print(f"  解析文件数: {len(all_parsed)}")
    print(f"  事实总数: {len(total_facts)}")
    print(f"  偏好总数: {len(total_preferences)}")
    print(f"  反思总数: {len(total_reflections)}")
    print(f"  关键偏好: {len(key_preferences)}")
    for pref in key_preferences[:10]:
        print(f"    - {pref[:80]}")

    if dry_run:
        print("\n[DRY-RUN] 跳过实际写入")
        return {
            "files": len(all_parsed),
            "facts": len(total_facts),
            "preferences": len(total_preferences),
            "reflections": len(total_reflections),
            "key_preferences": key_preferences,
        }

    # 4. 初始化新记忆系统
    service = MemoryService(new_settings)
    await service.ensure_started()

    # 5. 写入 Tier 3: 核心事实（用户偏好 + 关键记忆）
    core_count = 0
    for pref in key_preferences:
        await service.core.add_fact(
            pref,
            category="user_preference",
            source="remigration",
            weight=0.9,
        )
        core_count += 1

    # 写入其他高重要性偏好
    for pref in total_preferences:
        if len(pref.strip()) > 10:
            await service.core.add_fact(
                pref[:200],
                category="user_preference",
                source="remigration",
                weight=0.7,
            )
            core_count += 1

    print(f"\n[Tier 3] 写入 {core_count} 条核心事实")

    # 6. 写入 Tier 1: 每日记忆 (hierarchical)
    tier1_count = 0
    for parsed in all_parsed:
        # 将每日记忆的完整内容存储
        combined = []
        if parsed["facts"]:
            combined.append("### 事实记录\n" + "\n".join(parsed["facts"]))
        if parsed["reflections"]:
            combined.append("### 经验反思\n" + "\n".join(parsed["reflections"]))

        if combined:
            content = f"# {parsed['date']} 记忆\n\n" + "\n\n".join(combined)
            await service.hierarchical.store(
                content,
                importance=0.6,
                memory_type="conversation",
                scope="cross_session",
                source="remigration",
                tags=["migrated", parsed["date"]],
            )
            tier1_count += 1

    print(f"[Tier 1] 写入 {tier1_count} 条分层记忆")

    # 7. 迁移 MEMORY.md 到 Tier 3
    memory_md = old_dir / "MEMORY.md"
    if memory_md.exists():
        content = memory_md.read_text(encoding="utf-8").strip()
        if content and len(content) > 20:
            await service.core.add_fact(
                content[:500],
                category="identity",
                source="remigration",
                weight=1.0,
            )
            print(f"[Tier 3] 迁移 MEMORY.md 核心记忆")

    await service.close()

    result = {
        "files": len(all_parsed),
        "facts": len(total_facts),
        "preferences": len(total_preferences),
        "reflections": len(total_reflections),
        "key_preferences": key_preferences,
        "tier1_written": tier1_count,
        "tier3_written": core_count,
    }
    print(f"\n[DONE] 迁移完成!")
    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将旧 ReMe 记忆迁移至 4 层记忆系统"
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=None,
        help="旧 ReMe 目录 (默认: .novaecho/reme)",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=None,
        help="新记忆系统工作目录 (默认: .novaecho/memory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览不写入",
    )
    args = parser.parse_args()

    # 默认路径
    workspace = Path.cwd()
    old_dir = args.old_dir or workspace / ".novaecho" / "reme"
    new_working_dir = args.new_dir or workspace / ".novaecho" / "memory"

    if not old_dir.is_dir():
        print(f"[ERROR] 旧 ReMe 目录不存在: {old_dir}")
        return

    # 构建新配置
    settings = MemorySettings(
        working_dir=new_working_dir,
    )

    asyncio.run(migrate(old_dir, settings, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
