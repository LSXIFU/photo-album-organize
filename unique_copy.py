#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unique_copy.py — 找出 src 目录独有的图片(哈希不在参照库), 复制到 dst

参照库 = base 目录树下除 src 外的所有图片
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jfif',
        '.gif', '.mp4', '.webm'}


def md5(p: Path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(p, 'rb') as f:
            while b := f.read(chunk):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def scan_tree(root: Path, skip: Path):
    """遍历 root 下所有图片文件, 排除 skip 子树"""
    out = []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if skip is not None and (p == skip or skip in p.parents):
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description='独有图片筛选复制')
    ap.add_argument('--src', required=True, help='源目录(找独有)')
    ap.add_argument('--dst', required=True, help='目标目录(复制独有到此处)')
    ap.add_argument('--base', required=True, help='参照库根目录')
    ap.add_argument('--copy', action='store_true', help='执行复制(默认只出清单)')
    a = ap.parse_args()

    src = Path(a.src)
    dst = Path(a.dst)
    base = Path(a.base)

    src_files = scan_tree(src, None)
    print(f'[i] 源目录 {len(src_files)} 个文件')
    ref_files = scan_tree(base, src)
    print(f'[i] 参照库 {len(ref_files)} 个文件')

    # 参照库哈希集合
    ref_hashes = set()
    for i, p in enumerate(ref_files):
        h = md5(p)
        if h:
            ref_hashes.add(h)
    print(f'[i] 参照库哈希 {len(ref_hashes)} 个')

    # 源文件: 不在参照库的 = 独有
    unique = []
    for p in src_files:
        h = md5(p)
        if h and h not in ref_hashes:
            unique.append(p)
    print(f'[i] 独有图片 {len(unique)} 个 (其余 {len(src_files)-len(unique)} 个与图库重复)')

    for p in sorted(unique, key=lambda x: x.name):
        print(f'  独有  {p.name}  ({p.stat().st_size/1e6:.1f} MB)')

    if not a.copy:
        print('[i] 干跑模式 — 未复制。加 --copy 执行')
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in unique:
        shutil.copy2(p, dst / p.name)
        n += 1
    print(f'[✓] 已复制 {n} 个独有图片到 {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
