#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成合成测试图: 完整/裁剪/缩放/无关 混合, 验证分组管线"""
import random
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).parent / 'test_album'
OUT.mkdir(exist_ok=True)


def synth(w, h, seed, tag):
    """带特征丰富的合成图: 渐变底 + 随机几何 + 噪点, 保证 ORB 有匹配点"""
    rng = random.Random(seed)
    img = np.zeros((h, w, 3), np.uint8)
    # 渐变底
    for y in range(h):
        t = y / max(h - 1, 1)
        img[y, :] = (int(60 + 90 * t), int(90 + 60 * (1 - t)), int(120 * t + 40))
    # 随机几何
    for _ in range(rng.randint(25, 45)):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r = rng.randint(8, max(20, min(w, h) // 8))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        kind = rng.choice(['circle', 'rect', 'line'])
        if kind == 'circle':
            cv2.circle(img, (x, y), r, color, rng.choice([-1, 2]))
        elif kind == 'rect':
            cv2.rectangle(img, (x, y), (min(w - 1, x + r * 2), min(h - 1, y + r * 2)),
                          color, rng.choice([-1, 2]))
        else:
            cv2.line(img, (x, y), (min(w - 1, x + rng.randint(20, 120)),
                                   min(h - 1, y + rng.randint(20, 120))), color, 2)
    # 噪点
    noise = np.random.default_rng(seed).normal(0, 12, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 文字标签(便于人眼核对)
    cv2.putText(img, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def save(img, name):
    p = OUT / name
    cv2.imwrite(str(p), img)
    print(f"  {p.name}  {img.shape[1]}x{img.shape[0]}")


if __name__ == '__main__':
    print("生成测试图:")
    A = synth(1000, 700, seed=1, tag='A FULL')
    save(A, 'img_a_full.jpg')                 # 完整高清
    save(A[150:450, 250:700], 'img_b_crop.jpg')   # A 中部裁剪
    save(cv2.resize(A, (500, 350)), 'img_c_scaled.jpg')  # A 低清完整
    save(A[30:280, 40:340], 'img_e_crop2.jpg')    # A 左上裁剪
    save(cv2.resize(A[150:450, 250:700], (225, 150)), 'img_f_crop_small.jpg')  # B 的低清版
    B = synth(900, 640, seed=2, tag='D OTHER')
    save(B, 'img_d_other.jpg')                # 完全无关
    save(B[100:500, 200:700], 'img_g_crop_other.jpg')  # D 的裁剪
    print("完成")
