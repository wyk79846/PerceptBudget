#!/usr/bin/env python3
"""只读检查本地设置；不启动推理或创建目录。"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "python_executable": "executable", "flashrt_root": "directory",
    "openpi_root": "directory", "checkpoint_path": "directory",
    "model_path": "file", "norm_stats_path": "file", "source_document": "file",
}
FILES = {"model_path": "model.safetensors", "norm_stats_path": "norm_stats.json"}
REQUIRED = set(PATHS) | {"model_id", "output_root", "candidate_q", "H", "N", "precision", "state_enabled"}


def add(results, name, status, message):
    results.append({"name": name, "status": status, "message": message})


def under(path, root):
    return path == root or root in path.parents


def check_path(config, key, kind, results):
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        add(results, key, "error", "必须是非空绝对路径")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        message = "必须是绝对路径"
    elif not path.exists():
        message = "路径不存在"
    elif kind == "directory" and not path.is_dir():
        message = "必须是目录"
    elif kind != "directory" and not path.is_file():
        message = "必须是文件"
    elif key in FILES and path.name != FILES[key]:
        message = "文件名应为 " + FILES[key]
    elif kind == "executable" and not os.access(path, os.X_OK):
        message = "文件不可执行"
    else:
        add(results, key, "ok", "路径及类型正常")
        return path
    add(results, key, "error", message)
    return None


def validate_config(config):
    results = []
    missing = sorted(REQUIRED - config.keys())
    if missing:
        add(results, "config", "error", "缺少配置项：" + ", ".join(missing))
        return results
    paths = {key: check_path(config, key, kind, results) for key, kind in PATHS.items()}
    checkpoint, model, stats = (paths[k] for k in ("checkpoint_path", "model_path", "norm_stats_path"))
    model_id = config["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        add(results, "model_id", "error", "必须是非空模型标识")
    elif checkpoint and checkpoint.name != model_id:
        add(results, "model_id", "error", "应与 checkpoint 目录名一致")
    if checkpoint and model and model.resolve().parent != checkpoint.resolve():
        add(results, "model_path", "error", "model.safetensors 应位于 checkpoint 目录")
    if checkpoint and stats and not under(stats.resolve(), checkpoint.resolve()):
        add(results, "norm_stats_path", "error", "norm_stats.json 应位于 checkpoint 目录内")

    q = config["candidate_q"]
    q_ok = isinstance(q, list) and bool(q) and all(type(v) is int and v > 0 for v in q) and len(q) == len(set(q))
    add(results, "candidate_q", "ok" if q_ok else "error", "正整数且不重复" if q_ok else "必须是非空、不重复的正整数列表")
    h = config["H"]
    h_ok = type(h) is int and h > 0 and (not q_ok or max(q) <= h)
    add(results, "H", "ok" if h_ok else "error", "正整数且不小于最大 q" if h_ok else "必须为正整数且不小于最大 q")
    for key, expected in (("N", 1), ("precision", "fp16"), ("state_enabled", False)):
        valid = type(config[key]) is type(expected) and config[key] == expected
        add(results, key, "ok" if valid else "error", "符合固定配置" if valid else "不符合固定配置：" + key)

    value = config["output_root"]
    if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
        add(results, "output_root", "error", "必须是项目目录外的绝对路径")
    else:
        output = Path(value).expanduser()
        if under(output.resolve(), PROJECT_ROOT):
            add(results, "output_root", "error", "必须位于项目目录之外")
        elif output.exists() and not output.is_dir():
            add(results, "output_root", "error", "已存在但不是目录")
        else:
            add(results, "output_root", "ok" if output.exists() else "warning", "外部目录可用" if output.exists() else "尚不存在；不会创建")
    if "baseline_status" in config:
        add(results, "baseline_status", "info", "状态记录：" + str(config["baseline_status"]))
    return results


def main():
    parser = argparse.ArgumentParser(description="只读检查配置、依赖路径和基线参数。")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json", help="配置文件路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    results = []
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        add(results, "config", "error", "无法读取配置：" + str(exc))
    else:
        if not isinstance(config, dict):
            add(results, "config", "error", "配置顶层必须是 JSON 对象")
        else:
            add(results, "config", "ok", "JSON 配置可读")
            results.extend(validate_config(config))
    errors = sum(item["status"] == "error" for item in results)
    report = {"ok": errors == 0, "error_count": errors, "results": results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print("[{}] {}: {}".format(item["status"], item["name"], item["message"]))
        print("结论：{}（错误 {}）".format("通过" if not errors else "未通过", errors))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
