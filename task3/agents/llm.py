# -*- coding: utf-8 -*-
# agents/llm.py —— DeepSeek LLM 客户端（零第三方依赖，标准库 urllib 实现）
# 支持从环境变量覆盖配置；默认使用项目提供的 API Key。

import json
import os
import re
import time
import urllib.error
import urllib.request

# API 配置：优先读取环境变量，缺省使用项目提供的 key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d53a285e822e41dd9c65758ce4a18a24")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


class LLMError(RuntimeError):
    """LLM 调用异常"""


def chat(messages, temperature=0.0, max_tokens=8192, json_mode=True,
         timeout=300, logger=None, label="llm"):
    """调用 DeepSeek Chat Completions，返回 (content, usage)。

    json_mode=True 时使用 JSON 输出模式，要求模型返回合法 JSON 对象。
    """
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + DEEPSEEK_API_KEY,
        },
        method="POST",
    )

    t0 = time.time()
    est = _count_tokens(messages)
    if logger:
        logger.log("LLM", label,
                   f"调用 {DEEPSEEK_MODEL}（估算输入 {est} tokens，json_mode={json_mode}）")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:600]
        hint = ""
        if e.code in (401, 403):
            hint = "（提示：API Key 无效或已被撤销，请检查环境变量 DEEPSEEK_API_KEY 或 agents/llm.py 中的默认 key）"
        elif e.code == 402:
            hint = "（提示：DeepSeek 账户余额不足，请前往 platform.deepseek.com 充值）"
        elif e.code == 429:
            hint = "（提示：请求频率超限，请稍后重试）"
        raise LLMError("HTTP %s: %s %s" % (e.code, detail, hint)) from e
    except Exception as e:
        raise LLMError("网络异常: %s" % e) from e

    dt = time.time() - t0
    usage = result.get("usage", {})
    content = result["choices"][0]["message"]["content"]
    if logger:
        logger.log("LLM", label,
                   "返回完成，耗时 %.1fs，total_tokens=%s，回复 %d 字符"
                   % (dt, usage.get("total_tokens", "?"), len(content)))
    return content, usage


def extract_json(text):
    """从 LLM 回复中稳健提取 JSON 对象（兼容 markdown 代码块包裹与前后杂质）"""
    if not text:
        raise LLMError("LLM 返回空内容")
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        text = text[s:e + 1]
    return json.loads(text)


def chat_json(messages, logger=None, label="llm", attempts=3, **kw):
    """调用 chat 并解析 JSON；失败自动重试（网络抖动/限流/JSON 解析失败），
    认证类错误（401/403）不重试，直接抛出。"""
    last_err = None
    for i in range(attempts):
        try:
            content, usage = chat(messages, logger=logger, label=label, **kw)
            data = extract_json(content)
            return data, usage
        except LLMError as e:
            last_err = e
            if "401" in str(e) or "403" in str(e):
                raise
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
        if logger:
            logger.log("WARN", label, f"第 {i + 1} 次尝试失败（{last_err}），{2 * (i + 1)} 秒后重试…")
        time.sleep(2 * (i + 1))
    raise LLMError(f"LLM 调用重试 {attempts} 次仍失败：{last_err}")


def normalize_time(text):
    """把时间描述归一为 (年, 月[, 日])；无法解析返回 None"""
    if not text:
        return None
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(20\d{2})\s*年\s*第([一二三四])季度", text)
    if m:
        qmap = {"一": 3, "二": 6, "三": 9, "四": 12}
        return (int(m.group(1)), qmap[m.group(2)])
    return None


def _count_tokens(messages):
    """粗略估算输入 token 数（中文字符约 1.5 token，ASCII 约 0.3 token）"""
    total = 0
    for msg in messages:
        text = msg.get("content", "") or ""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        total += int(cjk * 1.5 + (len(text) - cjk) * 0.3)
    return total
