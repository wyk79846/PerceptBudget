#!/usr/bin/env python3
"""只读检查本地设置；不启动推理或创建目录。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRT_REVISION = "e4fe1232ee49f8d3cd96e32706114fd960b320fb"
FLASHRT_ENTRY = PROJECT_ROOT / "scripts" / "run_flashrt.py"
PATHS = {
    "python_executable": "executable", "flashrt_root": "directory",
    "openpi_root": "directory", "checkpoint_path": "directory",
    "model_path": "file", "norm_stats_path": "file", "source_document": "file",
}
FILES = {"model_path": "model.safetensors", "norm_stats_path": "norm_stats.json"}
REQUIRED = set(PATHS) | {"model_id", "flashrt_revision", "output_root", "candidate_q", "H", "N", "precision", "state_enabled"}


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
    revision = config["flashrt_revision"]
    revision_ok = revision == FLASHRT_REVISION
    add(results, "flashrt_revision", "ok" if revision_ok else "error",
        "固定快照 " + FLASHRT_REVISION if revision_ok else "必须固定为 " + FLASHRT_REVISION)
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
    n = config["N"]
    n_ok = type(n) is int and n > 0
    add(results, "N", "ok" if n_ok else "error",
        "正整数（当前值是基线候选）" if n_ok else "必须是正整数")
    precision = config["precision"]
    precision_ok = isinstance(precision, str) and precision in {"fp16", "bfloat16"}
    add(results, "precision", "ok" if precision_ok else "error",
        "支持值 fp16/bfloat16（当前值是基线候选）" if precision_ok
        else "必须是支持的精度：fp16 或 bfloat16")
    state_enabled = config["state_enabled"]
    state_ok = type(state_enabled) is bool
    add(results, "state_enabled", "ok" if state_ok else "error",
        "布尔值（config 默认关闭）" if state_ok else "必须是布尔值")

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


def check_flashrt_checkout(config, results):
    root_value = config.get("flashrt_root")
    if not isinstance(root_value, str) or not root_value.strip():
        return
    root = Path(root_value).expanduser()
    if not root.is_dir():
        return

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            text=True, capture_output=True, check=False,
        )

    head = git("rev-parse", "HEAD")
    if head.returncode:
        add(results, "flashrt_git_revision", "error", "无法读取 Git HEAD：" + head.stderr.strip())
    else:
        actual = head.stdout.strip()
        expected = config.get("flashrt_revision")
        valid = actual == expected == FLASHRT_REVISION
        add(results, "flashrt_git_revision", "ok" if valid else "error",
            actual + ("（与配置固定版本一致）" if valid else "（与配置固定版本不一致）"))

    status = git("status", "--porcelain", "--untracked-files=all")
    if status.returncode:
        add(results, "flashrt_git_clean", "error", "无法检查工作树：" + status.stderr.strip())
    else:
        clean = not status.stdout.strip()
        add(results, "flashrt_git_clean", "ok" if clean else "error",
            "工作树干净" if clean else "工作树有未提交或未跟踪文件")

    origin = git("remote", "get-url", "origin")
    if origin.returncode:
        add(results, "flashrt_origin", "error", "无法读取 origin：" + origin.stderr.strip())
    else:
        url = origin.stdout.strip()
        expected_url = "https://github.com/flashrt-project/FlashRT.git"
        add(results, "flashrt_origin", "ok" if url == expected_url else "error",
            url + ("（官方 origin）" if url == expected_url else "（应为 " + expected_url + "）"))


def check_flashrt_import(config, results):
    root_value = config.get("flashrt_root")
    executable_value = config.get("python_executable")
    if not isinstance(root_value, str) or not root_value.strip():
        add(results, "flashrt_import", "error", "flashrt_root 必须是非空路径")
        return
    if not isinstance(executable_value, str) or not executable_value.strip():
        add(results, "flashrt_import", "error", "python_executable 必须是非空路径")
        return
    root = Path(root_value).expanduser()
    executable = Path(executable_value).expanduser()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        add(results, "flashrt_import", "error", "配置的 Python 不存在或不可执行")
        return
    try:
        completed = subprocess.run(
            [str(executable), str(FLASHRT_ENTRY), "--root", str(root), "--check-import"],
            cwd=PROJECT_ROOT, text=True, capture_output=True, check=False, timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        add(results, "flashrt_import", "error", "隔离导入进程失败：" + str(exc))
        return

    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        detail = (completed.stderr or completed.stdout).strip()[-1200:]
        add(results, "flashrt_import", "error", "入口未返回导入报告：" + detail)
        return

    imported = payload.get("ok") is True and completed.returncode == 0
    if imported:
        add(results, "flashrt_import", "ok",
            "flash_rt.__file__: {}; FLASHRT_ROOT: {}; OPENPI_ROOT: {}".format(
                payload.get("flash_rt_file"), payload.get("FLASHRT_ROOT"),
                payload.get("OPENPI_ROOT")))
        add(results, "flashrt_import_dependencies", "info",
            "未发现 import 阶段缺项；模型运行依赖尚未验证")
    else:
        detail = payload.get("error") or "; ".join(payload.get("issues", [])) or "导入失败"
        if payload.get("missing_dependency"):
            detail = "缺少依赖/模块 " + str(payload["missing_dependency"]) + ": " + detail
            add(results, "flashrt_import_dependencies", "error", detail)
        add(results, "flashrt_import", "error", detail)


def main():
    parser = argparse.ArgumentParser(description="只读检查配置、依赖路径和基线参数。")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json", help="配置文件路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--check-import", action="store_true",
                        help="经进程入口在配置的 Python 中导入 FlashRT，不加载模型")
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
            check_flashrt_checkout(config, results)
            if args.check_import:
                check_flashrt_import(config, results)
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
