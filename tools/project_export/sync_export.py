#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "output",          # large renders
    "data",            # if you have big data
    "*.mp4",
    "*.mov",
    "*.avi",
]


def should_exclude(rel: str, patterns: List[str]) -> bool:
    # rel uses forward slashes
    for pat in patterns:
        if pat.startswith("*.") and fnmatch.fnmatch(Path(rel).name, pat):
            return True
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
            return True
        # directory prefix exclude
        if pat and (rel == pat or rel.startswith(pat.rstrip("/") + "/")):
            return True
    return False


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_tree(src: Path, dst: Path, excludes: List[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel_root = root_p.relative_to(src).as_posix()

        # prune excluded dirs in-place for os.walk
        pruned = []
        for d in list(dirs):
            rel = f"{rel_root}/{d}" if rel_root != "." else d
            if should_exclude(rel, excludes):
                pruned.append(d)
        for d in pruned:
            dirs.remove(d)

        # copy files
        for f in files:
            rel = f"{rel_root}/{f}" if rel_root != "." else f
            if should_exclude(rel, excludes):
                continue
            src_f = src / rel
            dst_f = dst / rel
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)


def render_tree(base: Path, excludes: List[str]) -> str:
    lines: List[str] = []
    base = base.resolve()
    lines.append(f"# Project Tree\n")
    lines.append(f"Root: `{base.name}`\n")

    def walk(dirpath: Path, prefix: str = ""):
        items = sorted(dirpath.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        # filter excluded
        filtered = []
        for p in items:
            rel = p.relative_to(base).as_posix()
            if should_exclude(rel, excludes):
                continue
            filtered.append(p)

        for i, p in enumerate(filtered):
            is_last = i == len(filtered) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir():
                extension = "    " if is_last else "│   "
                walk(p, prefix + extension)

    walk(base)
    lines.append("")
    return "\n".join(lines)


def make_zip(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(zip_path.with_suffix(""), "zip", root_dir=src_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a clean export folder + TREE.md (+ optional zip).")
    ap.add_argument("--repo", required=True, help="Path to your git repo root")
    ap.add_argument("--export", required=True, help="Path to write export folder")
    ap.add_argument("--zip", default="", help="Optional zip path, e.g. /tmp/project_upload.zip")
    ap.add_argument("--exclude", action="append", default=[], help="Additional exclude patterns")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    export = Path(args.export).expanduser().resolve()
    excludes = DEFAULT_EXCLUDES + list(args.exclude or [])

    if not repo.is_dir():
        print(f"ERROR: repo not found: {repo}", file=sys.stderr)
        return 2

    # wipe export (safe “delete old files”)
    safe_rmtree(export)
    export.mkdir(parents=True, exist_ok=True)

    # copy
    copy_tree(repo, export, excludes)

    # tree render
    tree_text = render_tree(export, excludes=[])
    (export / "TREE.md").write_text(tree_text, encoding="utf-8")

    # optional zip
    if args.zip:
        make_zip(export, Path(args.zip).expanduser().resolve())

    print(f"Export written to: {export}")
    if args.zip:
        print(f"Zip written to   : {Path(args.zip).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
