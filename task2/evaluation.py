# -*- coding: utf-8 -*-
"""视觉识别准确率评估模块。

定义并计算 4 个量化评估指标（题目要求至少 3 个）：
    1. 字段提取准确率 Field Accuracy：语义字段值提取的正确比例
    2. 数值识别误差率 Numeric Error Rate：数值字段的平均绝对百分比误差(MAPE)
    3. 图表结构还原正确率 Structure Accuracy：图表/版面结构元素还原命中率
    4. 类型路由准确率 Routing Accuracy：材料类型判定正确比例（系统级指标）
并用 8 张测试图片评测，输出结果表格。
"""

from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from typing import Dict, List

from agents.base_agent import AgentResult


def norm_str(s) -> str:
    """规范化字符串用于比较。"""
    return str(s).strip().lower().replace(" ", "").replace("\u3000", "")


def _sim(a: str, b: str) -> float:
    """计算两个规范化字符串的相似度。"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def field_correct(pred, gt) -> bool:
    """判断单个字段提取是否正确（含 OCR 噪声容忍规则）。"""
    # 字符串：规范化后完全相等 / 互为包含 / 相似度 >= 0.75（容忍 OCR 单字误差）
    if isinstance(gt, str):
        p, g = norm_str(pred), norm_str(gt)
        if p == g or (g and (g in p or p in g)):
            return True
        return _sim(p, g) >= 0.75
    # 列表：元素级比较（GT 元素在提取中被完全/包含/近似匹配的比例 >= 0.5）
    if isinstance(gt, list):
        if not isinstance(pred, list):
            pred = [pred] if pred else []
        gp = [norm_str(x) for x in gt]
        pp = [norm_str(x) for x in pred]
        if not gp:
            return True
        hit = 0
        for g in gp:
            if any(g == q or (g and (g in q or q in g)) or _sim(g, q) >= 0.8 for q in pp):
                hit += 1
        return hit / len(gp) >= 0.5
    # 字典：键值对逐个匹配，键需对应，值用包含/相似判断，命中率 >= 0.5
    if isinstance(gt, dict):
        if not isinstance(pred, dict):
            return False
        gp_keys = list(gt.keys())
        if not gp_keys:
            return True
        hit = 0
        for k in gp_keys:
            gv = norm_str(gt[k])
            if not gv:
                hit += 1
                continue
            # 找到提取字典中键相似或值能匹配的条目
            found = False
            for pk, pv in pred.items():
                kv_sim = _sim(norm_str(pk), norm_str(k))
                pv_n = norm_str(pv)
                if kv_sim >= 0.6 and (gv == pv_n or (gv and (gv in pv_n or pv_n in gv)) or _sim(gv, pv_n) >= 0.75):
                    found = True
                    break
            if found:
                hit += 1
        return hit / len(gp_keys) >= 0.5
    # 数值/布尔
    if isinstance(gt, bool):
        return bool(pred) == gt
    if isinstance(gt, (int, float)):
        try:
            return abs(float(pred) - float(gt)) <= max(1e-6, abs(float(gt)) * 0.05)
        except (TypeError, ValueError):
            return False
    return False


def field_accuracy_per_image(gt_fields: dict, pred_fields: dict) -> tuple:
    """计算单张图的字段提取准确率，返回 (正确数, 总字段数, 明细)。"""
    correct = 0
    total = 0
    detail = []
    for key, gt_val in gt_fields.items():
        total += 1
        pred_val = pred_fields.get(key)
        ok = field_correct(pred_val, gt_val)
        if ok:
            correct += 1
        detail.append({
            "字段": key, "期望": gt_val, "提取": pred_val, "正确": ok,
        })
    return correct, total, detail


def numeric_error_rate_per_image(gt_nums: dict, pred_nums: dict) -> tuple:
    """计算单张图的数值识别误差率(MAPE)，缺失按 100% 误差计。"""
    if not gt_nums:
        return None, 0, []
    errs = []
    for key, gt_val in gt_nums.items():
        pred = pred_nums.get(key)
        if pred is None:
            errs.append({"字段": key, "期望": gt_val, "提取": None, "相对误差": 1.0})
            continue
        gt = float(gt_val)
        p = float(pred)
        rel = abs(p - gt) / abs(gt) if gt != 0 else (0.0 if p == 0 else 1.0)
        errs.append({"字段": key, "期望": gt, "提取": p, "相对误差": round(rel, 4)})
    mape = sum(e["相对误差"] for e in errs) / len(errs)
    return mape, len(errs), errs


def structure_accuracy_per_image(gt_struct: dict, pred_struct: dict) -> tuple:
    """计算单张图的结构还原正确率。"""
    correct = 0
    total = 0
    detail = []
    for key, gt_val in gt_struct.items():
        total += 1
        pred = pred_struct.get(key)
        ok = field_correct(pred, gt_val)
        if ok:
            correct += 1
        detail.append({"结构元素": key, "期望": gt_val, "提取": pred, "正确": ok})
    acc = correct / total if total else None
    return acc, total, detail


def evaluate(results: List[AgentResult], gt_path: str = None) -> dict:
    """对全部识别结果执行评估，返回汇总报告。"""
    if gt_path is None:
        gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_by_file = {g["file"]: g for g in gt_data["images"]}

    rows = []
    total_f_correct = total_f_all = 0
    total_n_err = total_n_count = 0
    total_s_correct = total_s_all = 0
    total_routing_ok = 0
    n_images = len(results)

    for r in results:
        gt = gt_by_file.get(r.image)
        if gt is None:
            continue
        # 字段准确率
        f_c, f_t, f_d = field_accuracy_per_image(gt["ground_truth"], r.fields)
        total_f_correct += f_c
        total_f_all += f_t
        # 数值误差率
        n_mape, n_cnt, n_d = numeric_error_rate_per_image(gt.get("numeric_fields", {}), r.numeric_fields)
        if n_mape is not None:
            total_n_err += n_mape * n_cnt
            total_n_count += n_cnt
        # 结构还原率
        s_acc, s_t, s_d = structure_accuracy_per_image(gt.get("structure", {}), r.structure)
        total_s_correct += (s_acc * s_t if s_acc is not None else 0)
        total_s_all += s_t
        # 路由准确率
        routing_ok = (r.material_type_id == gt["material_type_id"])
        total_routing_ok += 1 if routing_ok else 0

        rows.append({
            "图片": r.image,
            "材料类型": gt["material_type"],
            "字段准确率": round(f_c / f_t, 3) if f_t else None,
            "数值MAPE": round(n_mape, 4) if n_mape is not None else None,
            "结构还原率": round(s_acc, 3) if s_acc is not None else None,
            "路由正确": routing_ok,
            "置信度": r.confidence,
            "耗时_s": r.processing_time,
        })

    summary = {
        "图片数": n_images,
        "字段提取准确率": round(total_f_correct / total_f_all, 4) if total_f_all else None,
        "数值识别误差率MAPE": round(total_n_err / total_n_count, 4) if total_n_count else None,
        "图表结构还原正确率": round(total_s_correct / total_s_all, 4) if total_s_all else None,
        "类型路由准确率": round(total_routing_ok / n_images, 4) if n_images else None,
    }
    return {"summary": summary, "rows": rows, "details": {
        "字段明细": [f_d for r in results for f_d in [field_accuracy_per_image(
            gt_by_file.get(r.image, {}).get("ground_truth", {}), r.fields)[2]]],
        "数值明细": [n_d for r in results for n_d in [numeric_error_rate_per_image(
            gt_by_file.get(r.image, {}).get("numeric_fields", {}), r.numeric_fields)[2]]],
    }}


def print_report(report: dict):
    """打印评估结果表格。"""
    print("=" * 110)
    print("视觉识别准确率评估结果（8 张测试图片）")
    print("=" * 110)
    header = f"{'图片':<42}{'材料类型':<14}{'字段准确率':<10}{'数值MAPE':<10}{'结构还原率':<10}{'路由':<6}{'置信度':<8}"
    print(header)
    print("-" * 110)
    for row in report["rows"]:
        print(f"{row['图片']:<42}{row['材料类型']:<14}"
              f"{str(row['字段准确率']):<10}{str(row['数值MAPE']):<10}{str(row['结构还原率']):<10}"
              f"{'√' if row['路由正确'] else '×':<6}{row['置信度']:<8}")
    print("-" * 110)
    s = report["summary"]
    print(f"汇总: 字段提取准确率={s['字段提取准确率']} | 数值识别误差率(MAPE)={s['数值识别误差率MAPE']} | "
          f"图表结构还原正确率={s['图表结构还原正确率']} | 类型路由准确率={s['类型路由准确率']}")
    print("=" * 110)


def main():
    """命令行入口：运行协调器后执行评估并保存结果。"""
    from coordinator import Coordinator
    coord = Coordinator()
    coord.run_pipeline()
    report = evaluate(coord.results)
    print_report(report)
    # 保存评估结果
    os.makedirs(coord.logs_dir, exist_ok=True)
    out = os.path.join(coord.logs_dir, "evaluation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[评估] 结果已保存 -> {out}")
    return report


if __name__ == "__main__":
    main()
