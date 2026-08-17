# -*- coding: utf-8 -*-
"""OCR 工具模块。

为所有视觉智能体提供统一的本地 OCR 能力（RapidOCR，内置中英文模型），
并提供按空间位置聚合成行的工具函数，供后续规则解析使用。
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

# 延迟加载 OCR 引擎，避免 import 时初始化过慢
_ENGINE = None


def get_engine():
    """获取 RapidOCR 单例引擎（懒加载，限制线程数降低内存占用）。"""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        # 限制推理线程数（默认 -1 用满所有线程，内存占用大）
        _ENGINE = RapidOCR(intra_op_num_threads=2, max_side_len=1500, rec_batch_num=1)
    return _ENGINE


def _load_optimized(image_path: str):
    """预处理图片：转 RGB、限制最长边，返回 numpy 数组（降低内存占用）。"""
    from PIL import Image
    import numpy as np
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    max_side = 1500  # 限制最长边，节省内存且不显著损失 OCR 精度
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(im)


# OCR 结果缓存路径（首次完整识别后缓存，后续直接复用，保证精度且省内存）
_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "ocr_cache.json",
)
_CACHE = None


def _load_cache() -> dict:
    """加载 OCR 缓存（懒加载）。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(_CACHE_PATH):
            try:
                import json
                with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                    _CACHE = json.load(f)
            except Exception:
                _CACHE = {}
    return _CACHE


def _save_cache():
    """保存 OCR 缓存。"""
    if _CACHE is None:
        return
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        import json
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False)
    except Exception:
        pass


def _cache_key(image_path: str) -> str:
    """构造缓存键：文件名 + 文件大小 + 修改时间（文件变化时自动失效）。"""
    st = os.stat(image_path)
    return f"{os.path.basename(image_path)}|{st.st_size}|{int(st.st_mtime)}"


def ocr_image(image_path: str) -> List[dict]:
    """对图片执行 OCR，返回按空间位置排序的文本项列表。

    每个文本项形如:
        {"text": 识别文本, "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
         "score": 置信度, "cx": 中心x, "cy": 中心y, "y": 顶部y, "x": 左侧x}
    优先使用缓存结果；无缓存时执行 OCR 并写入缓存（内存不足时自动降级重试）。
    """
    key = _cache_key(image_path)
    cache = _load_cache()
    if key in cache:
        return [dict(it) for it in cache[key]]

    engine = get_engine()
    items = []
    try:
        img_arr = _load_optimized(image_path)
        result, _ = engine(img_arr)
        items = _to_items(result)
    except Exception as e:
        # 内存不足等异常时，降级为更小尺寸重试一次
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(image_path).convert("RGB")
            im.thumbnail((700, 700), Image.LANCZOS)
            result, _ = engine(np.asarray(im))
            items = _to_items(result)
        except Exception:
            items = []
    cache[key] = items
    _save_cache()
    return items


def _to_items(result) -> List[dict]:
    """将 OCR 引擎原始输出转换为标准文本项列表。"""
    items = []
    if not result:
        return items
    for box, text, score in result:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append({
            "text": text,
            "box": box,
            "score": float(score),
            "cx": sum(xs) / 4.0,
            "cy": sum(ys) / 4.0,
            "x": min(xs),
            "y": min(ys),
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
        })
    items.sort(key=lambda t: (t["y"], t["x"]))
    return items
  
def cluster_lines(items: List[dict], tol: float = 12.0) -> List[str]:
    """将 OCR 文本项按垂直位置聚合成行文本列表。

    同一行内的文本项会按 x 从左到右拼接，便于规则解析。
    """
    if not items:
        return []
    lines = []
    cur_y = None
    cur_line = []
    for it in items:
        if cur_y is None or abs(it["y"] - cur_y) <= tol:
            cur_line.append(it)
        else:
            # 输出上一行
            cur_line.sort(key=lambda t: t["x"])
            lines.append(" ".join(t["text"] for t in cur_line))
            cur_line = [it]
        cur_y = it["y"]
    if cur_line:
        cur_line.sort(key=lambda t: t["x"])
        lines.append(" ".join(t["text"] for t in cur_line))
    return lines


def full_text(image_path: str) -> str:
    """返回图片的全部 OCR 文本（按行连接）。"""
    return "\n".join(cluster_lines(ocr_image(image_path)))
