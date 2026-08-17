# -*- coding: utf-8 -*-
"""视觉大模型后端（可选启用）。

通过 OpenAI 兼容的 Chat Completions 视觉接口调用大模型（GPT-4o / Qwen-VL /
GLM-4V / DeepSeek-VL 等），按各智能体的 Prompt 蓝图对图片做端到端识别。

启用方式（两种任选其一）：
    方式一（推荐）：编辑项目根目录 config.json 的 llm 段，填 api_key / base_url / model；
    方式二：设置环境变量 LLM_API_KEY（可选 LLM_BASE_URL / LLM_MODEL），优先级高于 config.json。

未启用（api_key 为空）时，系统自动使用本地 OCR + 规则版，保证零依赖可运行。
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

ENV_KEY = "LLM_API_KEY"
ENV_BASE = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"

# 默认值仅在“环境变量与 config.json 都未配置”时兜底；
# 配置优先级：环境变量 > config.json > 默认值。
# 本项目通过 config.json 配置 DeepSeek，因此实际运行走 DeepSeek。
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# 项目根目录下的本地配置文件（含 API Key 等，无需手动设置环境变量）
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.json",
)


def _load_config() -> dict:
    """读取项目根目录 config.json 中的 llm 配置；文件不存在或解析失败返回空字典。"""
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("llm", {}) or {}
    except Exception:
        pass
    return {}


def _get(key: str, env_name: str, default: str = "") -> str:
    """读取配置：环境变量优先，其次 config.json，最后默认值。"""
    val = os.environ.get(env_name, "").strip()
    if val:
        return val
    return str(_load_config().get(key, default) or default).strip()


# 图片扩展名 -> MIME 类型
_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "gif": "image/gif",
}


def is_enabled() -> bool:
    """是否启用了大模型后端（设置了 API Key：环境变量或 config.json）。"""
    return bool(_get("api_key", ENV_KEY))


def _image_data_url(image_path: str, max_side: int = 1600, quality: int = 88) -> str:
    """将图片编码为 data URL，供视觉接口使用。

    性能优化：图片尺寸超过 max_side（最长边）时，先用 PIL 等比缩放到
    max_side 并转 JPEG 压缩，显著减小上传体积与视觉模型的处理耗时；
    视觉模型对 1600px 以内的分辨率识别精度基本无损。
    任何异常（如缺少 PIL）回退到原始编码，保证功能可用。
    """
    try:
        from PIL import Image
        import io as _io
        im = Image.open(image_path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        # 兜底：原始编码
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = _MIME.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象（容忍 markdown 代码块等包裹）。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 代码块标记
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底：截取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _image_visual_description(image_path: str) -> str:
    """用图像分析计算图片的视觉特征描述（作为给 LLM 的视觉印象补充）。

    纯文本模型（如 DeepSeek）看不到像素，仅凭 OCR 文字层容易缺失版面/成像信息；
    这里用 PIL 计算亮度、色彩、白底占比、边缘密度等低层特征，生成一段
    “视觉印象”文本拼进 Prompt，帮助模型理解图片性质（照片/白板/文档/终端等）。
    这是感知层补充，不包含任何识别答案。
    """
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        a = np.asarray(im)
        gray = a.mean(axis=2)
        white_ratio = float((gray > 240).mean())
        dark_ratio = float((gray < 80).mean())
        colorfulness = float((a[:, :, 0].astype(int) - a[:, :, 2].astype(int)).std())
        brightness = float(gray.mean())
        # 边缘密度（粗略梯度）
        gx = float(np.abs(np.diff(gray, axis=1)).mean())
        gy = float(np.abs(np.diff(gray, axis=0)).mean())
        edge = round((gx + gy) / 2, 2)

        # 场景类型推断（启发式，仅作感知提示）
        if white_ratio > 0.85:
            if colorfulness < 8:
                scene = "白底文档/白板/手写材料（低色彩、高白底占比）"
            else:
                scene = "白底图表/表格类材料"
        elif dark_ratio > 0.5:
            scene = "深色背景界面/终端截图"
        elif colorfulness > 25:
            scene = "彩色实拍照片（色彩丰富）"
        else:
            scene = "普通图像（中等亮度）"
        return (
            f"【图片视觉特征】尺寸={w}x{h}；{scene}；"
            f"白底占比={white_ratio:.0%}；深色占比={dark_ratio:.0%}；"
            f"平均亮度={brightness:.0f}/255；色彩丰富度={colorfulness:.1f}；边缘密度={edge}"
        )
    except Exception:
        return ""


def llm_vision_extract(image_path: str, prompt: str, timeout: int = 90) -> Optional[dict]:
    """调用视觉大模型完成单张图片的结构化提取。

    返回解析后的 JSON（期望含 fields/numeric_fields/structure/confidence）；
    任何异常（网络/限流/解析失败）返回 None，由上层回退到规则版。
    """
    api_key = _get("api_key", ENV_KEY)
    base_url = _get("base_url", ENV_BASE, DEFAULT_BASE_URL).rstrip("/")
    model = _get("model", ENV_MODEL, DEFAULT_MODEL)
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ],
        }],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    # 支持代理（config.json 的 llm.proxy 或环境变量 HTTPS_PROXY/HTTP_PROXY）
    proxy = _get("proxy", "HTTPS_PROXY") or _get("proxy", "HTTP_PROXY")
    if proxy:
        ph = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(ph)
        urlopen = opener.open
    else:
        urlopen = urllib.request.urlopen
    # 自动重试：免费层限流(429)/服务端抖动(5xx)/超时/解析失败时重试最多 3 次
    import time as _time
    data = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 1:
                _time.sleep(2 + attempt * 3)
                continue
            return None
        except Exception:
            if attempt < 1:
                _time.sleep(2 + attempt * 3)
                continue
            return None
    if data is None:
        return None
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    parsed = _extract_json(content)
    # 解析失败也重试
    if parsed is None and content:
        for attempt in range(2):
            _time.sleep(3 + attempt * 3)
            try:
                with urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
            except Exception:
                continue
            parsed = _extract_json(content)
            if parsed:
                break
    return parsed


def _chat_text(prompt: str, timeout: int) -> Optional[str]:
    """调用纯文本 chat/completions，返回模型回复文本；失败返回 None。"""
    api_key = _get("api_key", ENV_KEY)
    base_url = _get("base_url", ENV_BASE, DEFAULT_BASE_URL).rstrip("/")
    model = _get("model", ENV_MODEL, DEFAULT_MODEL)
    if not api_key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    proxy = _get("proxy", "HTTPS_PROXY") or _get("proxy", "HTTP_PROXY")
    if proxy:
        ph = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(ph)
        urlopen = opener.open
    else:
        urlopen = urllib.request.urlopen
    import time as _time
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 1:
                _time.sleep(2 + attempt * 3)
                continue
            return None
        except Exception:
            if attempt < 1:
                _time.sleep(2 + attempt * 3)
                continue
            return None
    return None


def llm_text_extract(image_path: str, prompt: str, timeout: int = 90) -> Optional[dict]:
    """OCR 文本层 + 大模型语义理解通道（适配纯文本模型，如 DeepSeek）。

    流程：RapidOCR 提取图片文字层（含位置） -> 组织成文本描述 -> 送 LLM 结构化 -> 解析 JSON。
    任何环节失败返回 None，由上层回退规则版。
    """
    try:
        from .ocr_utils import ocr_image
        items = ocr_image(image_path)
        if not items:
            return None
        # 按阅读顺序组织 OCR 文本层（带空间位置，便于 LLM 理解版面）
        ocr_lines = [f"y={it['y']:.0f}, x={it['x']:.0f}: {it['text']}" for it in items]
        visual = _image_visual_description(image_path)
        full_prompt = (
            f"{prompt}\n\n"
            f"{visual}\n\n"
            "【图片 OCR 文本层（视觉感知结果，按空间位置从上到下排列，y/x 为坐标）】\n"
            f"{chr(10).join(ocr_lines)}\n\n"
            "请结合视觉特征与 OCR 文本层推断图片内容，输出严格 JSON，不要输出任何其他文字。"
        )
        content = _chat_text(full_prompt, timeout)
        if not content:
            return None
        return _extract_json(content)
    except Exception:
        return None


def _extract_once(image_path: str, prompt: str, timeout: int) -> Optional[dict]:
    """按当前生效配置调用一次（mode=vision 或 text）。"""
    mode = _get("mode", "LLM_MODE", "text").lower()
    if mode == "vision":
        return llm_vision_extract(image_path, prompt, timeout)
    return llm_text_extract(image_path, prompt, timeout)


def llm_extract(image_path: str, prompt: str, timeout: int = 90,
               use_fallback: bool = True, retry: int = 1) -> Optional[dict]:
    """按配置模式选择大模型通道，支持主/备双后端降级：
      - 主后端：config.json 的 llm 段（当前为 qwen 视觉，mode=vision）
      - 备后端：llm.fallback 段（当前为 DeepSeek，mode=text）
    主后端调用失败（限流/配额/超时/解析失败）时：
      - use_fallback=True：自动用备后端重试，保证系统始终可运行（适合演示）；
      - use_fallback=False：主后端重试 retry 次后返回 None（适合评估，避免文本
        后端基于 OCR 重构出的结果与视觉识别口径不一致）。
    """
    # 主后端调用（失败时重试 retry 次）
    raw = None
    for _ in range(retry + 1):
        raw = _extract_once(image_path, prompt, timeout)
        if raw is not None:
            return raw
    if not use_fallback:
        return None
    # 主后端失败 -> 降级到 fallback 后端
    fb = (_load_config().get("fallback") or {})
    if not fb.get("api_key"):
        return None
    saved = {k: os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_MODE")}
    try:
        # 环境变量优先级最高，临时切换配置到 fallback
        os.environ["LLM_API_KEY"] = str(fb["api_key"])
        os.environ["LLM_BASE_URL"] = str(fb.get("base_url") or DEFAULT_BASE_URL)
        os.environ["LLM_MODEL"] = str(fb.get("model") or DEFAULT_MODEL)
        os.environ["LLM_MODE"] = str(fb.get("mode", "text"))
        return _extract_once(image_path, prompt, timeout)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ----------------------------------------------------------------------
# LLM 识别结果缓存（保证评估可复现，避免大模型输出波动）
# ----------------------------------------------------------------------
_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "llm_cache.json",
)
_CACHE = None


def _load_cache() -> dict:
    """加载 LLM 结果缓存（懒加载）。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(_CACHE_PATH):
            try:
                with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                    _CACHE = json.load(f)
            except Exception:
                _CACHE = {}
    return _CACHE


def _save_cache():
    """保存 LLM 结果缓存。"""
    if _CACHE is None:
        return
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False)
    except Exception:
        pass


def get_model_name() -> str:
    """返回当前生效的模型名（用于缓存键）。"""
    return _get("model", ENV_MODEL, DEFAULT_MODEL)


def cache_key(image_path: str) -> str:
    """构造缓存键：图片文件名 + 模型名（换模型自动失效）。"""
    return f"{os.path.basename(image_path)}|{get_model_name()}"


def get_cached(image_path: str) -> Optional[dict]:
    """读取缓存中的识别结果；无缓存返回 None。"""
    return _load_cache().get(cache_key(image_path))


def put_cache(image_path: str, raw: dict):
    """写入识别结果到缓存。"""
    if not raw:
        return
    _load_cache()[cache_key(image_path)] = raw
    _save_cache()
