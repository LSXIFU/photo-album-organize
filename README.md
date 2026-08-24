# photo-album-organize

相册 / 图片文件夹整理工具集：**字节级去重 → 内容分组（识别同一张图的裁剪/缩放/多来源版本）→ 全量排序重命名**。

纯 Python，依赖 OpenCV + Pillow + NumPy。Windows / Linux / macOS 通用（中文路径已处理）。

## 功能

- **哈希去重**（`dedupe.py`）：字节级完全相同（MD5 一致）的重复文件，每组保留一个，删除副本，清单留档
- **内容分组**（`photo_organizer.py`）：识别"同一张图"的不同版本——高清完整版 vs 裁剪版、缩放版、不同来源下载、重压缩版，按内容归组并让它们在文件管理器中紧挨排列
- **独有筛选**（`unique_copy.py`）：找出某目录独有的图片（哈希不在整个图库其他位置），复制到目标目录
- **全量重命名**：不保留源文件名，统一 `{组号:03d}_{主色调}_{序号:02d}.ext` 格式；组内按像素数降序（完整版在前）；主色调用英文代号标注画面主色
- **安全设计**：默认干跑只出报告；执行前生成 `rename_map.csv` 映射可一键回滚；去重记录留档；不移动/不覆盖任何文件

## 快速开始

```bash
# 依赖（需要 Python 3.9+）
pip install opencv-python numpy pillow

# 1. 去重：先看清单，确认后加 --delete
python dedupe.py --dir "相册路径"

# 2. 内容分组 + 重命名：先干跑看分组报告
python photo_organizer.py --dir "相册路径"
# 确认无误后执行（--yes 跳过交互确认）
python photo_organizer.py --dir "相册路径" --apply --yes
# 后悔了？一键回滚
python photo_organizer.py --dir "相册路径" --rollback

# 3. 清除纯低清版 (同内容等比缩放, 无色差无裁剪, 仅分辨率低的版本)
python photo_organizer.py --dir "相册路径" --purge-lowres                    # 干跑: 列出低清清单
python photo_organizer.py --dir "相册路径" --purge-lowres --apply --yes      # 执行: 移动到 _lowres/
python photo_organizer.py --dir "相册路径" --purge-lowres --to "D:/低清回收" # 指定移动目标目录
python photo_organizer.py --dir "相册路径" --rollback-lowres                 # 按 lowres_map.csv 移回

# 4. 独有图片筛选复制（--base 是参照库根目录）
python unique_copy.py --src "源目录" --dst "目标目录" --base "图库根目录" --copy

# 5. 合成测试图验证（可选）
python make_test_images.py   # 生成 完整/裁剪/缩放/无关 混合测试集
```

## 分组管线原理

```
HSV 直方图粗筛(Top-K 候选) → ORB 特征 + RANSAC 单应判定"谁是谁的子图"
→ 纹理区像素验证(杀误连) → 并查集分组 → 组内像素降序 → 命名
```

- **粗筛**：全局 HSV 直方图（对缩放/压缩鲁棒，裁剪部分鲁棒），快速召回候选对
- **精判**：ORB 特征匹配 + RANSAC 求单应，判定一张图是否是另一张的子图（裁剪/缩放关系）
- **像素验证**：把子图 warp 到全图坐标系，只统计**纹理显著区域**的灰度相似度——真裁剪 ≈0.95+，白底/纯色/构图相似的误连 ≈0.82-0.84，`PIX_THRESH=0.90` 干净切割

## 关键参数（photo_organizer.py 头部常量）

| 参数 | 默认 | 含义 |
|---|---|---|
| `TOP_K` | 10 | 每张图直方图最近候选数 |
| `ORB_NFEATURES` | 800 | ORB 特征数（过大 BFMatcher O(n×m) 爆炸） |
| `MIN_MATCHES` | 8 | 单应内点下限（精确性由像素验证兜底） |
| `PIX_THRESH` | 0.90 | 像素级验证相似度阈值 |

## 性能

445 张图约 28 秒：512px 缩略图单次解码 + JPEG draft + 8 线程并行 + ORB 800 特征。

## 许可证

MIT © 2026 LSXIFU
