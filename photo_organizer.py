#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo_organizer.py — 相册图片按内容分组排序重命名

管线（级联，成熟方案）:
  1) 粗筛: HSV 全局直方图(卡方距离) → 每张取 Top-K 候选对   [召回, 对裁剪鲁棒]
  2) 精判: ORB 特征匹配 + RANSAC 单应 → 判定"谁是谁的子图" [精确, 识别裁剪/缩放]
  3) 分组: 并查集合并子图关系
  4) 排序: 组内按像素数(宽×高)降序 → 完整版在前
  5) 命名: 组内最高清图的原名作前缀 + _001 序号; 单图不动

安全设计:
  - 默认干跑(只出报告 + rename_map.csv), 不碰任何文件
  - --apply 才真正重命名; --yes 跳过最终确认
  - 改名映射全量写入 rename_map.csv, --rollback 可一键回滚
  - 不删除/移动任何文件

用法:
  python photo_organizer.py --dir <相册目录>                 # 干跑预览
  python photo_organizer.py --dir <相册目录> --apply         # 执行重命名
  python photo_organizer.py --dir <相册目录> --rollback      # 按 rename_map.csv 回滚
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jfif'}
CSV_NAME = 'rename_map.csv'
LOWRES_CSV_NAME = 'lowres_map.csv'
VDUP_CSV_NAME = 'visual_dup_map.csv'

# ---- 可调参数(粗筛) ----
HIST_BINS = (8, 6, 3)      # HSV 量化 bin
TOP_K = 10                 # 每张图直方图最近候选数
HIST_THRESH = 120.0        # 卡方距离上限(放宽, 靠 Top-K 兜底召回)

# ---- 可调参数(精判 ORB) ----
ORB_MAX_EDGE = 512         # 特征统一在 512 缩略图上算(管线已缩, 不再二次缩放)
ORB_NFEATURES = 800        # 512 图上 800 特征足够匹配; 特征爆炸会让 BFMatcher O(n*m) 爆炸
MIN_MATCHES = 8            # 放宽召回(512缩略大图 vs 未缩放裁剪图 匹配少); 精确性由像素验证兜底
RATIO_TEST = 0.75          # Lowe 比例测试
MARGIN_RATIO = 0.05        # 角点判定边界容差(5%)
MAX_SIZE_RATIO = 10.0      # 对角线比超过则跳过(尺度差异过大 ORB 判不了)
PIX_THRESH = 0.90          # 像素级验证: warp 后重叠区纹理区灰度相似度下限
                           # (真裁剪≈0.95+, 暗色大场景误连≈0.82-0.84, 同场景不同图≈0.86-0.90)
SCALE_SIM_THRESH = 0.85    # 纯低清判定: 高清降采样到低清尺寸后的纹理区相似度下限
                           # (纯缩放≈0.90+, 裁剪/不同图≈0.6-0.8)
LOWRES_PIX_RATIO = 0.60    # 低清像素数 < 高清像素数的该比例 才判为低清
LOWRES_ASPECT_TOL = 0.03   # 宽高比容差(纯缩放必须一致)
SCALE_UP_DROP = 0.03       # 反向比较: sim(高清→低清) - sim(低清→高清) 超过该值
                           # → 低清只是高清的局部巧合(如拼贴图的一个元素), 拒绝
VISUAL_DUP_THRESH = 0.97   # 视觉重复: 分辨率接近(≥90%)且双向相似度 ≥ 该值 → 同图仅编码不同
VISUAL_DUP_PIX_RATIO = 0.90  # 分辨率接近阈值(像素比下限)
VISUAL_DUP_HIST_THRESH = 0.5  # HSV直方图卡方距离上限: 超过→色调不同(如暖/冷版本), 拒绝
                            # (同色重压缩≈0.0-0.1, 色调不同版本≈1.0-2.5)


# ------------------------------------------------------------------ IO
def exif_corrected_size(path: Path):
    """EXIF 旋转后的真实 (h, w); 读不到返回 (0,0)"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(274, 1) in (5, 6, 7, 8):
                w, h = h, w
            return h, w
    except Exception:
        img = cv2.imread(str(path))
        if img is None:
            return 0, 0
        return img.shape[:2]


def load_thumbnail(path: Path, max_edge=512):
    """PIL 单次解码: EXIF 方向修正 + 缩放到 max_edge → BGR numpy。
    JPEG 用 draft mode 只解到所需尺寸(不全解码), 大图提速关键"""
    try:
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = 200_000_000   # 本地相册可信, 放宽解压炸弹限制
        with Image.open(path) as im:
            im.draft('RGB', (max_edge, max_edge))   # JPEG 渐进解码(其他格式忽略)
            im = ImageOps.exif_transpose(im).convert('RGB')
            if max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image(path: Path):
    """读图(BGR) + EXIF 方向修正; 全失败返回 None。
    链路: cv2.imdecode 主路(快) → 失败退 PIL 兜底(编码/格式更全)
    Windows 中文路径必须 np.fromfile + imdecode, cv2.imread 读不了"""
    img = None
    # 1) cv2 主路
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # 2) cv2 失败 → PIL 兜底
    if img is None:
        try:
            from PIL import Image, ImageOps
            Image.MAX_IMAGE_PIXELS = 200_000_000   # 本地相册可信, 放宽解压炸弹限制
            with Image.open(path) as im:
                img = cv2.cvtColor(
                    np.array(ImageOps.exif_transpose(im).convert('RGB')),
                    cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            return None
    # 3) cv2 读到, 但需 PIL 做 EXIF 方向修正
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            if im.getexif().get(274, 1) != 1:
                arr = np.array(ImageOps.exif_transpose(im).convert('RGB'))
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return img


def scan_images(d: Path):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)


# ------------------------------------------------------------------ 特征
def compute_hist(img, bins=HIST_BINS):
    """HSV 全局直方图(先缩到 256 提速), 归一化后扁平。
    对缩放/压缩鲁棒, 裁剪部分鲁棒"""
    if max(img.shape[:2]) > 256:
        s = 256.0 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hb, sb, vb = bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [hb, sb, vb],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 主色调代号(2字母, Eng): 最大色相簇占比达标则取主色, 否则黑白灰/多色
TONE_RANGES = [
    (10, 20, 'RD'),    # 红
    (20, 45, 'OR'),    # 橙
    (45, 70, 'YW'),    # 黄
    (70, 165, 'GN'),   # 绿
    (165, 200, 'CY'),  # 青
    (200, 260, 'BL'),  # 蓝
    (260, 290, 'PP'),  # 紫
    (290, 350, 'PK'),  # 粉
]
TONE_MIN_RATIO = 0.25   # 主色簇最少占比, 不足判为 MX(多色)


def compute_tone(img):
    """返回 2 字母主色调代号 (RD/OR/YW/GN/CY/BL/PP/PK/WH/GY/BK/MX)"""
    small = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]   # H 0-180 -> 0-360

    # 黑白灰优先
    if v.mean() < 45:
        return 'BK'
    if s.mean() < 25:
        return 'WH' if v.mean() > 200 else 'GY'

    # 彩色: 统计色相簇占比
    colored = (s > 30) & (v > 40)
    if colored.sum() == 0:
        return 'GY'
    bins = np.digitize(h[colored] % 360, [r[1] for r in TONE_RANGES])
    bins = np.where(bins >= len(TONE_RANGES), 0, bins)   # 350°~360° 红色卷绕回 0
    counts = np.bincount(bins, minlength=len(TONE_RANGES))
    best = int(counts.argmax())
    if counts[best] / colored.sum() < TONE_MIN_RATIO:
        return 'MX'
    return TONE_RANGES[best][2]


def chi2_dist(a, b):
    """卡方距离: sum((a-b)^2 / (a+b+eps))"""
    s = a + b + 1e-6
    return float(np.sum((a - b) ** 2 / s))


def orb_features(img):
    """ORB 特征(缩放提速)。返回 (img_small, kp, des); 失败返回 (None,None,None)
    img_small 供像素级验证复用, 避免二次加载"""
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    if max(h, w) > ORB_MAX_EDGE:
        s = ORB_MAX_EDGE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, scaleFactor=1.2,
                         nlevels=8, fastThreshold=8, edgeThreshold=15)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(des) < 8:
        return None, None, None
    return img, kp, des


def find_subset(kp_a, des_a, size_a, kp_b, des_b, size_b):
    """B 是否为 A 的子图(裁剪/缩放关系)。
    返回 (relation, H): relation in ('B_in_A','A_in_B',None), H 为 B→A 单应"""
    if des_a is None or des_b is None:
        return None, None
    diag_a = (size_a[0] ** 2 + size_a[1] ** 2) ** 0.5
    diag_b = (size_b[0] ** 2 + size_b[1] ** 2) ** 0.5
    if max(diag_a, diag_b) / (min(diag_a, diag_b) + 1e-6) > MAX_SIZE_RATIO:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_b, des_a, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            good.append(pair[0])          # 只有 1 个近邻: 直接接受
        elif pair[0].distance < RATIO_TEST * pair[1].distance:
            good.append(pair[0])
    if len(good) < MIN_MATCHES:
        return None, None

    src = np.float32([kp_b[m.queryIdx].pt for m in good])
    dst = np.float32([kp_a[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
    inliers = int(mask.sum())
    if inliers < MIN_MATCHES:
        return None, None

    h_a, w_a = size_a
    h_b, w_b = size_b
    ma = MARGIN_RATIO * max(w_a, h_a)
    mb = MARGIN_RATIO * max(w_b, h_b)

    # B 的四个角 → A 平面
    cb = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    tb = cv2.perspectiveTransform(cb, H).reshape(-1, 2)
    if all(-ma <= x <= w_a + ma and -ma <= y <= h_a + ma for x, y in tb):
        return 'B_in_A', H

    # A 的四个角 → B 平面(逆变换)
    ret, Hinv = cv2.invert(H)
    if not ret:
        return None, None
    ca = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    ta = cv2.perspectiveTransform(ca, Hinv).reshape(-1, 2)
    if all(-mb <= x <= w_b + mb and -mb <= y <= h_b + mb for x, y in ta):
        return 'A_in_B', H
    return None, None


def pixel_similarity(img_full, img_sub, H, full_wh):
    """像素级验证: 用单应把子图 warp 到全图坐标系, 算重叠区灰度相似度。
    关键: 只统计纹理显著区域(Sobel 边缘)的差异——
    白底/纯色区对误判贡献大(构图相似的图白底都像), 排除后
    真裁剪: 内容区一致 → 高相似; 误连: 内容区错位 → 低相似 → 拒绝"""
    if img_full is None or img_sub is None:
        return 0.0
    w_f, h_f = full_wh
    warped = cv2.warpPerspective(img_sub, H, (w_f, h_f))
    gf = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
    gw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 统一降采样(平滑小裁剪图放大插值), 同尺度比较
    scale = 512.0 / max(w_f, h_f)
    if scale < 1.0:
        nw, nh = max(1, int(w_f * scale)), max(1, int(h_f * scale))
        gf = cv2.resize(gf, (nw, nh))
        gw = cv2.resize(gw, (nw, nh))
    gf = gf.astype(np.float32)
    gw = gw.astype(np.float32)
    # 纹理 mask: warp 覆盖区 ∩ 内容边缘区(排除白底)
    gx = cv2.Sobel(gw, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gw, cv2.CV_32F, 0, 1, ksize=3)
    tex = (np.hypot(gx, gy) > 12.0) & (gw > 1.0)
    if tex.sum() < 150:
        return 0.0
    diff = np.abs(gf - gw)[tex]
    return float(1.0 - diff.mean() / 255.0)


def scale_similarity(img_high, img_low):
    """高清降采样到低清尺寸后, 算纹理区灰度相似度(用于纯低清判定)。
    纯缩放(同内容): 降采样后内容全对齐 → 高相似;
    裁剪版: 内容区域错位/比例不同 → 低相似"""
    if img_high is None or img_low is None:
        return 0.0
    lh, lw = img_low.shape[:2]
    small = cv2.resize(img_high, (lw, lh), interpolation=cv2.INTER_AREA)
    gf = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gl = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 纹理 mask: 低清图内容边缘区(排除白底/纯色)
    gx = cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gl, cv2.CV_32F, 0, 1, ksize=3)
    tex = (np.hypot(gx, gy) > 12.0) & (gl > 1.0)
    if tex.sum() < 150:
        return 0.0
    diff = np.abs(gf - gl)[tex]
    return float(1.0 - diff.mean() / 255.0)


def scale_similarity_up(img_high, img_low):
    """反向比较: 低清放大到高清尺寸, 算纹理区灰度相似度。
    真低清(全图等比缩放): 放大后内容全图对应 → 高相似;
    局部巧合(低清是高清某区域的裁剪放大, 如拼贴图元素): 放大后只有局部匹配 → 低相似"""
    if img_high is None or img_low is None:
        return 0.0
    hh, hw = img_high.shape[:2]
    big = cv2.resize(img_low, (hw, hh), interpolation=cv2.INTER_CUBIC)
    gf = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 纹理 mask 用高清图(内容真实, 边缘可信)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    tex = (np.hypot(gx, gy) > 12.0) & (gf > 1.0)
    if tex.sum() < 150:
        return 0.0
    diff = np.abs(gf - gb)[tex]
    return float(1.0 - diff.mean() / 255.0)


def is_pure_downscale(img_high, size_h, img_low, size_l):
    """img_low 是否是 img_high 的纯低清版(等比缩放, 无裁剪无色差)"""
    px_h = size_h[0] * size_h[1]
    px_l = size_l[0] * size_l[1]
    if px_l >= px_h * LOWRES_PIX_RATIO:      # 像素没显著少
        return False
    aspect_h = size_h[1] / max(size_h[0], 1)
    aspect_l = size_l[1] / max(size_l[0], 1)
    if abs(aspect_l - aspect_h) / max(aspect_h, 1e-6) > LOWRES_ASPECT_TOL:
        return False                          # 宽高比不一致 → 裁剪/变形, 不是纯缩放
    sd = scale_similarity(img_high, img_low)
    if sd < SCALE_SIM_THRESH:
        return False
    # 反向比较: 局部巧合(低清只是高清某区域的放大)在放大方向会露馅
    su = scale_similarity_up(img_high, img_low)
    if sd - su > SCALE_UP_DROP:
        return False
    return True


def is_visual_dup(img_high, size_h, img_low, size_l):
    """img_low 是否与 img_high 视觉重复(同分辨率级别同内容同色, 仅编码不同)。
    JPEG/PNG 编码非唯一: 同像素内容不同压缩/保存 → 字节哈希不同但视觉一致"""
    px_h = size_h[0] * size_h[1]
    px_l = size_l[0] * size_l[1]
    if px_l < px_h * VISUAL_DUP_PIX_RATIO:   # 分辨率差太多(>10%) → 不是视觉重复
        return False
    sd = scale_similarity(img_high, img_low)
    su = scale_similarity_up(img_high, img_low)
    if min(sd, su) < VISUAL_DUP_THRESH:
        return False
    # 颜色一致性: 灰度构图一致≠视觉重复——色调不同(暖/冷版本)不算
    # HSV直方图距离: 同色重压缩≈0, 色调不同版本明显大
    d = chi2_dist(compute_hist(img_high), compute_hist(img_low))
    if d > VISUAL_DUP_HIST_THRESH:
        return False
    return True


def find_visual_dup(images, sizes, thumbs, groups):
    """返回 [(dup_idx, keep_idx)]: 组内两两比较, 分辨率接近且像素一致的(编码不同副本)。
    必须两两比较: 组内可能有多个同分辨率副本, 而最高清可能是另一内容(漏检源)"""
    dups = []
    for g in groups:
        if len(g) < 2:
            continue
        members = sorted(g, key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
        for k, low_i in enumerate(members):
            if thumbs[low_i] is None:
                continue
            for high_i in members[:k]:          # 比 low_i 更大的成员
                if thumbs[high_i] is None:
                    continue
                if is_visual_dup(thumbs[high_i], sizes[high_i],
                                 thumbs[low_i], sizes[low_i]):
                    dups.append((low_i, high_i))
                    break                        # 找到一张即可, 不重复标记
    return dups


def find_lowres(images, sizes, thumbs, groups):
    """返回 [(low_idx, keep_idx)]: 组内相对最高清为纯低清版的索引对。
    每个组以像素最大的成员为基准, 其余成员逐个判定"""
    lowres = []
    for g in groups:
        if len(g) < 2:
            continue
        members = sorted(g, key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
        keep = members[0]
        if thumbs[keep] is None:
            continue
        for i in members[1:]:
            if thumbs[i] is None:
                continue
            if is_pure_downscale(thumbs[keep], sizes[keep], thumbs[i], sizes[i]):
                lowres.append((i, keep))
    return lowres


# ------------------------------------------------------------------ 分组
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ------------------------------------------------------------------ 主流程
def purge_runner(d, images, sizes, thumbs, groups, find_fn, csv_name, subdir,
                 label, apply, yes, to_dir):
    """通用: 找出冗余版本(find_fn) → 干跑报告 → apply 移动到 subdir → CSV 记录(可回滚)"""
    hits = find_fn(images, sizes, thumbs, groups)
    if not hits:
        print(f"[i] 未发现{label}")
        return 0
    print(f"\n======== {label}清单 ({len(hits)} 个) ========")
    for low_i, keep_i in hits:
        lh, lw = sizes[low_i]
        sh, sw = sizes[keep_i]
        print(f"  [{label}] {images[low_i].name}  ({lw}x{lh})"
              f"  ← 保留 {images[keep_i].name}  ({sw}x{sh})")
    print("============================\n")
    csv_path = d / csv_name
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'kept_with', 'moved_to'])
        for low_i, keep_i in hits:
            w.writerow([images[low_i].name, images[keep_i].name, subdir])
    print(f"[i] {label}清单已写: {csv_path}")
    if not apply:
        print("[i] 干跑模式 — 未移动任何文件。确认后加 --apply 执行移动")
        return 0
    if not yes:
        r = input(f"确认移动 {len(hits)} 个{label}到 {to_dir or (subdir + '/')}? [y/N] ").strip().lower()
        if r != 'y':
            print("[i] 已取消")
            return 0
    target = Path(to_dir) if to_dir else (d / subdir)
    target.mkdir(parents=True, exist_ok=True)
    done = 0
    for low_i, keep_i in hits:
        src = d / images[low_i].name
        dst = target / images[low_i].name
        if not src.exists():
            print(f"[!] 源不存在, 跳过: {src.name}")
            continue
        if dst.exists():          # 目标重名保底: 加序号, 绝不覆盖
            stem, ext = src.stem, src.suffix
            k = 1
            while dst.exists():
                dst = target / f"{stem}_{k}{ext}"
                k += 1
        src.rename(dst)
        done += 1
    print(f"[✓] 已移动 {done}/{len(hits)} 个{label}到 {target}  (清单 {csv_name}, 可回滚)")
    return 0


def plan_renames(images, sizes, tones, groups):
    """全量重命名(不保留源文件名): {组号:03d}_{色调}_{序号:02d}.
    组内像素降序 → 完整版在前; 组号由调用方按原文件名排序分配"""
    plan = []
    stats = []
    used = set()
    for gi, g in enumerate(groups, 1):
        members = sorted(g, key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
        for k, i in enumerate(members, 1):
            ext = Path(images[i]).suffix.lower()
            tone = tones[i]
            new = f"{gi:03d}_{tone}_{k:02d}{ext}"
            n = 1
            while new.lower() in used:
                new = f"{gi:03d}_{tone}_{n:02d}_{k:02d}{ext}"
                n += 1
            used.add(new.lower())
            plan.append((images[i].name, new))   # 只存文件名, 目录由 --dir 决定
        stats.append((gi, len(members)))
    return plan, stats


def run_dir(d: Path, apply: bool, yes: bool, purge_lowres: bool = False,
            purge_visual_dup: bool = False, to_dir: str = None):
    images = scan_images(d)
    if not images:
        print(f"[!] 目录下没有图片: {d}")
        return 1
    n = len(images)
    print(f"[i] 扫描到 {n} 张图片")
    t0 = time.time()

    # --- 1) 缩略图 + 直方图粗筛 (并行解码, PIL/libjpeg 释放 GIL) ---
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        thumbs = list(ex.map(load_thumbnail, images))
    hists, sizes, tones = [], [], []
    for p, img in zip(images, thumbs):
        sizes.append(img.shape[:2] if img is not None else (0, 0))
        if img is None:
            hists.append(None)
            tones.append('MX')
            print(f"[!] 读图失败跳过: {p.name}")
        else:
            hists.append(compute_hist(img))
            tones.append(compute_tone(img))
    H = np.array([h if h is not None else np.zeros(np.prod(HIST_BINS))
                  for h in hists], dtype=np.float64)

    cand = set()
    for i in range(n):
        row = np.sum((H[i:i + 1] - H) ** 2 / (H[i:i + 1] + H + 1e-6), axis=1)
        row[i] = np.inf
        idx = np.argpartition(row, min(TOP_K, n - 1))[:TOP_K]
        for j in idx:
            if j > i and row[j] < HIST_THRESH:
                cand.add((i, int(j)))
    print(f"[i] 粗筛候选对 {len(cand)}  (直方图 Top-{TOP_K})  用时 {time.time()-t0:.1f}s")

    # --- 2) ORB 精判 (复用已解码缩略图) ---
    feats = {}          # index -> (img, kp, des)
    rel = []            # (i, j)  i 包含 j
    for k, (i, j) in enumerate(sorted(cand)):
        for x in (i, j):
            if x not in feats:
                feats[x] = orb_features(thumbs[x])
        img_i, kp_i, de_i = feats[i]
        img_j, kp_j, de_j = feats[j]
        if img_i is None or img_j is None:
            continue
        # 坐标系以 ORB 缩放后的实际图为准
        r, H = find_subset(kp_i, de_i, img_i.shape[:2],
                           kp_j, de_j, img_j.shape[:2])
        if r is None or H is None:
            continue
        # 像素级验证: 杀白底/纯色误匹配
        if r == 'B_in_A':
            sim = pixel_similarity(img_i, img_j, H,
                                   (img_i.shape[1], img_i.shape[0]))
            if sim >= PIX_THRESH:
                rel.append((i, j))
        else:  # A_in_B
            ret, Hinv = cv2.invert(H)
            if not ret:
                continue
            sim = pixel_similarity(img_j, img_i, Hinv,
                                   (img_j.shape[1], img_j.shape[0]))
            if sim >= PIX_THRESH:
                rel.append((j, i))
    print(f"[i] ORB 判定包含关系 {len(rel)} 对  用时 {time.time()-t0:.1f}s")

    # --- 3) 并查集分组 ---
    uf = UnionFind(n)
    for i, j in rel:
        uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    groups = list(groups.values())          # 含单图组(每张单图=一组)
    # 组号分配: 按组内最小文件名排序 → 保持原浏览顺序 + 内容聚合
    groups.sort(key=lambda g: min(Path(images[i]).name for i in g))
    print(f"[i] 形成 {len(groups)} 个组 (含单图)  用时 {time.time()-t0:.1f}s")

    # --- 3b) 冗余版本清除 (--purge-lowres / --purge-visual-dup, 独立于重命名) ---
    if purge_lowres or purge_visual_dup:
        # 像素比必须用原始尺寸(缩略图都压到512, 丢失像素关系)
        orig_sizes = [exif_corrected_size(p) for p in images]
        if purge_lowres:
            return purge_runner(d, images, orig_sizes, thumbs, groups,
                                find_lowres, LOWRES_CSV_NAME, '_lowres', '纯低清版',
                                apply, yes, to_dir)
        return purge_runner(d, images, orig_sizes, thumbs, groups,
                            find_visual_dup, VDUP_CSV_NAME, '_visual_dup', '视觉重复',
                            apply, yes, to_dir)

    # --- 4) 排序 + 命名计划 ---
    plan, stats = plan_renames(images, sizes, tones, groups)

    # --- 报告 ---
    print("\n======== 重命名计划 ========")
    multi = [(gi, c) for gi, c in stats if c > 1]
    singles = len(stats) - len(multi)
    print(f"  内容组(≥2张) {len(multi)} 个 / 单图组 {singles} 个 / 共 {len(plan)} 个文件全量重命名")
    if multi:
        shown = ", ".join(f"G{gi:03d}x{c}" for gi, c in multi[:15])
        print(f"  多图组: {shown}{' ...' if len(multi) > 15 else ''}")
    print("==========================\n")

    if not plan:
        print("[i] 没有需要重命名的组")
        return 0

    # 写 CSV(无论是否 apply 都写, 供预览/回滚)
    csv_path = d / CSV_NAME
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['old', 'new'])
        w.writerows(plan)
    print(f"[i] 改名映射已写: {csv_path}  ({len(plan)} 条)")

    if not apply:
        print("[i] 干跑模式 — 未改任何文件。确认无误后加 --apply 执行")
        return 0

    if not yes:
        r = input(f"确认重命名 {len(plan)} 个文件? [y/N] ").strip().lower()
        if r != 'y':
            print("[i] 已取消")
            return 0

    # 两阶段 rename: 先全部→唯一临时名, 再→最终名。
    # 防止"旧文件已用新格式命名"时源/目标名空间重叠导致跳过
    tmp = []
    for k, (old, new) in enumerate(plan):
        src = Path(d) / old
        if not src.exists():
            print(f"[!] 源不存在, 跳过: {old}")
            continue
        t = Path(d) / f"__tmp_{k:04d}{src.suffix.lower()}"
        src.rename(t)
        tmp.append((t, Path(d) / new))
    done = 0
    for t, dst in tmp:
        if dst.exists():
            # 目标冲突保底: 临时名保留, 绝不丢文件
            print(f"[!] 目标已存在, 保留临时名: {t.name} → {dst.name}")
            t.rename(Path(d) / f"__collision_{dst.name}")
            continue
        t.rename(dst)
        done += 1
    print(f"[✓] 已重命名 {done}/{len(plan)} 个文件 (映射存于 {CSV_NAME}, 可 --rollback)")
    return 0


def do_rollback(d: Path):
    csv_path = d / CSV_NAME
    if not csv_path.exists():
        print(f"[!] 找不到 {csv_path}, 无法回滚")
        return 1
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    done = 0
    for old, new in rows:
        src, dst = Path(d) / new, Path(d) / old   # 文件名 + 目录
        if src.exists() and not dst.exists():
            src.rename(dst)
            done += 1
    print(f"[✓] 已回滚 {done}/{len(rows)} 个文件")
    return 0


def do_rollback_purge(d: Path, csv_name: str):
    """按 purge csv (file, kept_with, moved_to) 把移走的文件移回原目录"""
    csv_path = d / csv_name
    if not csv_path.exists():
        print(f"[!] 找不到 {csv_path}, 无法回滚")
        return 1
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    done = 0
    for row in rows:
        if len(row) < 3:
            continue
        name, _kept, moved = row[0], row[1], row[2]
        src = d / moved / name
        dst = d / name
        if src.exists() and not dst.exists():
            src.rename(dst)
            done += 1
    print(f"[✓] 已移回 {done}/{len(rows)} 个文件")
    return 0


def main():
    ap = argparse.ArgumentParser(description='相册图片按内容分组排序重命名')
    ap.add_argument('--dir', required=True, help='图片目录')
    ap.add_argument('--apply', action='store_true', help='真正执行(默认干跑)')
    ap.add_argument('--yes', action='store_true', help='跳过最终确认')
    ap.add_argument('--rollback', action='store_true', help='按 rename_map.csv 回滚重命名')
    ap.add_argument('--purge-lowres', action='store_true',
                    help='清除纯低清版(同内容等比缩放, 移动而非删除)')
    ap.add_argument('--purge-visual-dup', action='store_true',
                    help='清除视觉重复(同分辨率同内容仅编码不同, 移动而非删除)')
    ap.add_argument('--to', default=None,
                    help='冗余文件移动目标目录(默认 <dir>/_lowres 或 <dir>/_visual_dup)')
    ap.add_argument('--rollback-lowres', action='store_true',
                    help='按 lowres_map.csv 把低清版移回原目录')
    ap.add_argument('--rollback-visual-dup', action='store_true',
                    help='按 visual_dup_map.csv 把视觉重复移回原目录')
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_dir():
        print(f"[!] 目录不存在: {d}")
        return 1
    # 回滚必须独立: 直接读 csv, 绝不能重跑分析管线(会覆盖 csv)
    if a.rollback:
        sys.exit(do_rollback(d))
    if a.rollback_lowres:
        sys.exit(do_rollback_purge(d, LOWRES_CSV_NAME))
    if a.rollback_visual_dup:
        sys.exit(do_rollback_purge(d, VDUP_CSV_NAME))
    sys.exit(run_dir(d, a.apply, a.yes, a.purge_lowres, a.purge_visual_dup, a.to))


if __name__ == '__main__':
    main()
