#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedupe.py — 文件夹内字节级完全相同的文件去重(按哈希)

流程:
  1) 扫描目录所有图片/视频文件, 计算 MD5
  2) 相同哈希分组, 每组保留一个(文件名排序后第一个), 其余待删
  3) 待删清单写 dedupe_map.csv, --delete 才真正删除

安全: 默认只出清单; --delete 执行删除(记录留档); 保留策略见下
保留策略: 每组按文件名排序, 优先保留非 Cache_/非 (1) 副本的原始名
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jfif',
        '.gif', '.mp4', '.webm', '.mov', '.mkv'}
CSV_NAME = 'dedupe_map.csv'

# 被认为"缓存/副本"的文件名特征, 优先删除
COPY_HINTS = ('cache_', '(1).', '(2).', '(3).', 'thumb', 'thumbnail')


def is_secondary(name: str) -> bool:
    """True 表示该文件名看起来是缓存/副本, 应优先删"""
    low = name.lower()
    return any(h in low for h in COPY_HINTS)


def file_md5(p: Path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(p, 'rb') as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description='字节级完全相同文件去重')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--delete', action='store_true', help='执行删除(默认只出清单)')
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_dir():
        print(f'[!] 目录不存在: {d}')
        return 1

    files = [p for p in sorted(d.iterdir())
             if p.suffix.lower() in EXTS and p.is_file()]
    print(f'[i] 扫描到 {len(files)} 个文件')

    by_hash = {}
    for p in files:
        md5 = file_md5(p)
        if md5:
            by_hash.setdefault(md5, []).append(p)

    dup_groups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
    if not dup_groups:
        print('[i] 没有字节级完全相同的文件, 无需去重')
        return 0

    # 每组选保留者 + 待删者
    keep, delete = [], []
    for h, ps in dup_groups.items():
        # 优先保留非副本名的; 全部是副本则留第一个
        primary = [p for p in ps if not is_secondary(p.name)]
        pool = primary if primary else ps
        k = pool[0]
        for p in ps:
            if p != k:
                delete.append(p)
            else:
                keep.append(p)
    delete.sort()

    print(f'[i] 重复组 {len(dup_groups)} 组, 待删 {len(delete)} 个文件, '
          f'可释放 {sum(p.stat().st_size for p in delete) / 1e6:.1f} MB')

    csv_path = d / CSV_NAME
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['md5', 'keep', 'delete', 'size_bytes'])
        kmap = {}
        for p in keep:
            kmap[p] = p.stat().st_size
        for h, ps in dup_groups.items():
            k = next(p for p in ps if p in [x for x in keep])
            for p in ps:
                if p != k:
                    w.writerow([h, k.name, p.name, p.stat().st_size])
    print(f'[i] 清单已写: {csv_path}')

    for p in delete:
        print(f'  删  {p.name}  ({p.stat().st_size/1e6:.1f} MB)')

    if not a.delete:
        print('[i] 干跑模式 — 未删除。加 --delete 执行')
        return 0

    n = 0
    for p in delete:
        try:
            p.unlink()
            n += 1
        except OSError as e:
            print(f'[!] 删除失败 {p.name}: {e}')
    print(f'[✓] 已删除 {n}/{len(delete)} 个重复文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
